#ml-service/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import joblib
import sqlite3
import os
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import requests
import threading
import time

# App configuration
app = Flask(__name__)
CORS(app)

# Configuration
DATABASE_PATH = os.environ.get('DATABASE_PATH', '../backend/recommendations.db')
MODEL_PATH = os.environ.get('MODEL_PATH', './models/')
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:5000')

# Ensure model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

# Global variables for models and data
user_item_matrix = None
content_features = None
tfidf_vectorizer = None
svd_model = None
scaler = None
last_model_update = None
recommendation_cache = {}

class RecommendationEngine:
    def __init__(self):
        self.user_profiles = {}
        self.content_similarity = {}
        self.interaction_weights = {
            'rating': 1.0,
            'favorite': 0.8,
            'watchlist': 0.6,
            'viewed': 0.4
        }
        self.diversity_factor = 0.3
        self.load_models()
    
    def load_models(self):
        """Load existing models or create new ones"""
        global tfidf_vectorizer, svd_model, scaler, last_model_update
        
        try:
            if os.path.exists(f"{MODEL_PATH}/tfidf_vectorizer.pkl"):
                tfidf_vectorizer = joblib.load(f"{MODEL_PATH}/tfidf_vectorizer.pkl")
                logging.info("Loaded TF-IDF vectorizer")
            
            if os.path.exists(f"{MODEL_PATH}/svd_model.pkl"):
                svd_model = joblib.load(f"{MODEL_PATH}/svd_model.pkl")
                logging.info("Loaded SVD model")
            
            if os.path.exists(f"{MODEL_PATH}/scaler.pkl"):
                scaler = joblib.load(f"{MODEL_PATH}/scaler.pkl")
                logging.info("Loaded scaler")
            
            last_model_update = datetime.utcnow()
            
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            self.initialize_models()
    
    def initialize_models(self):
        """Initialize new models"""
        global tfidf_vectorizer, svd_model, scaler
        
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        svd_model = TruncatedSVD(
            n_components=50,
            random_state=42
        )
        
        scaler = StandardScaler()
        
        logging.info("Initialized new models")
    
    def save_models(self):
        """Save trained models to disk"""
        global tfidf_vectorizer, svd_model, scaler
        
        try:
            if tfidf_vectorizer:
                joblib.dump(tfidf_vectorizer, f"{MODEL_PATH}/tfidf_vectorizer.pkl")
            
            if svd_model:
                joblib.dump(svd_model, f"{MODEL_PATH}/svd_model.pkl")
            
            if scaler:
                joblib.dump(scaler, f"{MODEL_PATH}/scaler.pkl")
            
            logging.info("Models saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving models: {e}")
    
    def get_database_connection(self):
        """Get database connection"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logging.error(f"Database connection error: {e}")
            return None
    
    def load_user_data(self, user_id):
        """Load user interaction data"""
        conn = self.get_database_connection()
        if not conn:
            return {}
        
        try:
            query = """
            SELECT ui.*, c.title, c.genres, c.overview, c.content_type, c.rating as content_rating
            FROM user_interaction ui
            JOIN content c ON ui.content_id = c.id
            WHERE ui.user_id = ?
            ORDER BY ui.timestamp DESC
            """
            
            cursor = conn.execute(query, (user_id,))
            interactions = cursor.fetchall()
            
            user_data = {
                'interactions': [],
                'preferences': {
                    'genres': defaultdict(float),
                    'content_types': defaultdict(float),
                    'rating_patterns': []
                }
            }
            
            for interaction in interactions:
                interaction_data = {
                    'content_id': interaction['content_id'],
                    'interaction_type': interaction['interaction_type'],
                    'value': interaction['value'] or 0,
                    'timestamp': interaction['timestamp'],
                    'content': {
                        'title': interaction['title'],
                        'genres': json.loads(interaction['genres']) if interaction['genres'] else [],
                        'overview': interaction['overview'],
                        'content_type': interaction['content_type'],
                        'rating': interaction['content_rating']
                    }
                }
                
                user_data['interactions'].append(interaction_data)
                
                # Update preferences
                weight = self.interaction_weights.get(interaction['interaction_type'], 0.1)
                
                # Genre preferences
                for genre in interaction_data['content']['genres']:
                    if isinstance(genre, dict):
                        genre_name = genre.get('name', '')
                    else:
                        genre_name = str(genre)
                    
                    if genre_name:
                        user_data['preferences']['genres'][genre_name] += weight
                
                # Content type preferences
                user_data['preferences']['content_types'][interaction['content_type']] += weight
                
                # Rating patterns
                if interaction['interaction_type'] == 'rating' and interaction['value']:
                    user_data['preferences']['rating_patterns'].append(interaction['value'])
            
            conn.close()
            return user_data
            
        except Exception as e:
            logging.error(f"Error loading user data: {e}")
            conn.close()
            return {}
    
    def load_content_data(self, limit=1000):
        """Load content data for similarity calculations"""
        conn = self.get_database_connection()
        if not conn:
            return []
        
        try:
            query = """
            SELECT id, title, overview, genres, content_type, rating, metadata
            FROM content
            WHERE cached_at > datetime('now', '-30 days')
            ORDER BY rating DESC
            LIMIT ?
            """
            
            cursor = conn.execute(query, (limit,))
            content_data = cursor.fetchall()
            
            content_list = []
            for content in content_data:
                content_item = {
                    'id': content['id'],
                    'title': content['title'],
                    'overview': content['overview'] or '',
                    'genres': json.loads(content['genres']) if content['genres'] else [],
                    'content_type': content['content_type'],
                    'rating': content['rating'] or 0,
                    'metadata': json.loads(content['metadata']) if content['metadata'] else {}
                }
                content_list.append(content_item)
            
            conn.close()
            return content_list
            
        except Exception as e:
            logging.error(f"Error loading content data: {e}")
            conn.close()
            return []
    
    def build_content_features(self, content_data):
        """Build content feature matrix"""
        global tfidf_vectorizer, content_features
        
        try:
            # Prepare text features
            text_features = []
            for content in content_data:
                genres_text = ' '.join([g['name'] if isinstance(g, dict) else str(g) for g in content['genres']])
                overview_text = content['overview']
                combined_text = f"{genres_text} {overview_text}"
                text_features.append(combined_text)
            
            # Fit TF-IDF vectorizer
            if not tfidf_vectorizer:
                self.initialize_models()
            
            tfidf_matrix = tfidf_vectorizer.fit_transform(text_features)
            
            # Add numerical features
            numerical_features = []
            for content in content_data:
                features = [
                    content['rating'],
                    len(content['genres']),
                    len(content['overview'].split()) if content['overview'] else 0,
                    1 if content['content_type'] == 'movie' else 0,
                    1 if content['content_type'] == 'tv' else 0,
                    1 if content['content_type'] == 'anime' else 0
                ]
                numerical_features.append(features)
            
            numerical_features = np.array(numerical_features)
            
            # Scale numerical features
            if scaler:
                numerical_features = scaler.fit_transform(numerical_features)
            
            # Combine features
            content_features = np.hstack([tfidf_matrix.toarray(), numerical_features])
            
            logging.info(f"Built content features matrix: {content_features.shape}")
            return content_features
            
        except Exception as e:
            logging.error(f"Error building content features: {e}")
            return None
    
    def calculate_content_similarity(self, content_data):
        """Calculate content similarity matrix"""
        try:
            if content_features is None:
                self.build_content_features(content_data)
            
            if content_features is not None:
                similarity_matrix = cosine_similarity(content_features)
                
                # Store similarity for quick lookup
                self.content_similarity = {}
                for i, content in enumerate(content_data):
                    self.content_similarity[content['id']] = {
                        'similarities': similarity_matrix[i],
                        'content_ids': [c['id'] for c in content_data]
                    }
                
                logging.info(f"Calculated content similarity matrix: {similarity_matrix.shape}")
                return similarity_matrix
            
        except Exception as e:
            logging.error(f"Error calculating content similarity: {e}")
            return None
    
    def get_collaborative_recommendations(self, user_id, num_recommendations=10):
        """Get collaborative filtering recommendations"""
        try:
            # Load interaction data for collaborative filtering
            conn = self.get_database_connection()
            if not conn:
                return []
            
            query = """
            SELECT user_id, content_id, interaction_type, value
            FROM user_interaction
            WHERE interaction_type IN ('rating', 'favorite', 'watchlist')
            """
            
            cursor = conn.execute(query)
            interactions = cursor.fetchall()
            
            if not interactions:
                return []
            
            # Build user-item matrix
            user_item_data = defaultdict(lambda: defaultdict(float))
            
            for interaction in interactions:
                weight = self.interaction_weights.get(interaction['interaction_type'], 0.1)
                value = interaction['value'] or 1.0
                user_item_data[interaction['user_id']][interaction['content_id']] = weight * value
            
            # Convert to matrix format
            users = list(user_item_data.keys())
            items = set()
            for user_items in user_item_data.values():
                items.update(user_items.keys())
            items = list(items)
            
            if user_id not in users or len(items) < 10:
                return []
            
            # Create user-item matrix
            matrix = np.zeros((len(users), len(items)))
            user_idx = {user: i for i, user in enumerate(users)}
            item_idx = {item: i for i, item in enumerate(items)}
            
            for user, user_items in user_item_data.items():
                for item, rating in user_items.items():
                    matrix[user_idx[user], item_idx[item]] = rating
            
            # Apply SVD
            if svd_model:
                reduced_matrix = svd_model.fit_transform(matrix)
                
                # Find similar users
                target_user_idx = user_idx[user_id]
                user_similarities = cosine_similarity([reduced_matrix[target_user_idx]], reduced_matrix)[0]
                
                # Get recommendations from similar users
                similar_users = np.argsort(user_similarities)[::-1][1:6]  # Top 5 similar users
                
                recommendations = []
                target_user_items = set(user_item_data[user_id].keys())
                
                for similar_user_idx in similar_users:
                    similar_user_id = users[similar_user_idx]
                    similar_user_items = user_item_data[similar_user_id]
                    
                    for item_id, rating in similar_user_items.items():
                        if item_id not in target_user_items and rating > 0.5:
                            recommendations.append({
                                'content_id': item_id,
                                'score': rating * user_similarities[similar_user_idx],
                                'reason': 'Users with similar taste also liked this'
                            })
                
                # Sort by score and return top recommendations
                recommendations.sort(key=lambda x: x['score'], reverse=True)
                
                conn.close()
                return recommendations[:num_recommendations]
            
            conn.close()
            return []
            
        except Exception as e:
            logging.error(f"Error in collaborative recommendations: {e}")
            return []
    
    def get_content_based_recommendations(self, user_id, num_recommendations=10):
        """Get content-based recommendations"""
        try:
            user_data = self.load_user_data(user_id)
            if not user_data['interactions']:
                return []
            
            # Get user's favorite content
            favorite_content = []
            for interaction in user_data['interactions']:
                if interaction['interaction_type'] in ['favorite', 'rating']:
                    if interaction['interaction_type'] == 'rating' and interaction['value'] >= 4.0:
                        favorite_content.append(interaction['content_id'])
                    elif interaction['interaction_type'] == 'favorite':
                        favorite_content.append(interaction['content_id'])
            
            if not favorite_content:
                return []
            
            # Get similar content
            recommendations = []
            for content_id in favorite_content[:5]:  # Limit to top 5 favorites
                if content_id in self.content_similarity:
                    similarities = self.content_similarity[content_id]['similarities']
                    content_ids = self.content_similarity[content_id]['content_ids']
                    
                    # Get top similar content
                    similar_indices = np.argsort(similarities)[::-1][1:6]  # Skip self
                    
                    for idx in similar_indices:
                        similar_content_id = content_ids[idx]
                        if similar_content_id not in favorite_content:
                            recommendations.append({
                                'content_id': similar_content_id,
                                'score': similarities[idx],
                                'reason': 'Similar to content you liked'
                            })
            
            # Remove duplicates and sort
            unique_recommendations = {}
            for rec in recommendations:
                content_id = rec['content_id']
                if content_id not in unique_recommendations or rec['score'] > unique_recommendations[content_id]['score']:
                    unique_recommendations[content_id] = rec
            
            recommendations = list(unique_recommendations.values())
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logging.error(f"Error in content-based recommendations: {e}")
            return []
    
    def get_hybrid_recommendations(self, user_id, num_recommendations=20):
            """Get hybrid recommendations combining collaborative and content-based"""
            try:
                # Get recommendations from both approaches
                collaborative_recs = self.get_collaborative_recommendations(user_id, num_recommendations)
                content_based_recs = self.get_content_based_recommendations(user_id, num_recommendations)
                
                # Combine and weight recommendations
                combined_recs = {}
                
                # Weight collaborative filtering recommendations
                for rec in collaborative_recs:
                    content_id = rec['content_id']
                    combined_recs[content_id] = {
                        'content_id': content_id,
                        'score': rec['score'] * 0.6,  # 60% weight for collaborative
                        'reason': rec['reason'],
                        'sources': ['collaborative']
                    }
                
                # Weight content-based recommendations
                for rec in content_based_recs:
                    content_id = rec['content_id']
                    if content_id in combined_recs:
                        combined_recs[content_id]['score'] += rec['score'] * 0.4  # 40% weight for content-based
                        combined_recs[content_id]['sources'].append('content-based')
                    else:
                        combined_recs[content_id] = {
                            'content_id': content_id,
                            'score': rec['score'] * 0.4,
                            'reason': rec['reason'],
                            'sources': ['content-based']
                        }
                
                # Apply diversity factor
                final_recs = list(combined_recs.values())
                final_recs.sort(key=lambda x: x['score'], reverse=True)
                
                # Add diversity by content type and genres
                diversified_recs = self.apply_diversity_filter(final_recs, user_id)
                
                return diversified_recs[:num_recommendations]
                
            except Exception as e:
                logging.error(f"Error in hybrid recommendations: {e}")
                return []
    
    def apply_diversity_filter(self, recommendations, user_id):
        """Apply diversity filter to recommendations"""
        try:
            user_data = self.load_user_data(user_id)
            user_genres = user_data['preferences']['genres']
            user_content_types = user_data['preferences']['content_types']
            
            # Get content details for recommendations
            content_details = {}
            conn = self.get_database_connection()
            if conn:
                content_ids = [rec['content_id'] for rec in recommendations]
                placeholders = ','.join(['?' for _ in content_ids])
                
                query = f"""
                SELECT id, genres, content_type, title
                FROM content
                WHERE id IN ({placeholders})
                """
                
                cursor = conn.execute(query, content_ids)
                results = cursor.fetchall()
                
                for row in results:
                    content_details[row['id']] = {
                        'genres': json.loads(row['genres']) if row['genres'] else [],
                        'content_type': row['content_type'],
                        'title': row['title']
                    }
                
                conn.close()
            
            # Apply diversity scoring
            diversified_recs = []
            genre_counts = defaultdict(int)
            type_counts = defaultdict(int)
            
            for rec in recommendations:
                content_id = rec['content_id']
                if content_id in content_details:
                    content = content_details[content_id]
                    
                    # Calculate diversity penalty
                    diversity_penalty = 0
                    
                    # Genre diversity
                    for genre in content['genres']:
                        genre_name = genre['name'] if isinstance(genre, dict) else str(genre)
                        diversity_penalty += genre_counts[genre_name] * 0.1
                    
                    # Content type diversity
                    diversity_penalty += type_counts[content['content_type']] * 0.2
                    
                    # Apply diversity penalty
                    adjusted_score = rec['score'] * (1 - diversity_penalty * self.diversity_factor)
                    
                    rec['score'] = adjusted_score
                    rec['title'] = content['title']
                    diversified_recs.append(rec)
                    
                    # Update counts
                    for genre in content['genres']:
                        genre_name = genre['name'] if isinstance(genre, dict) else str(genre)
                        genre_counts[genre_name] += 1
                    type_counts[content['content_type']] += 1
                else:
                    diversified_recs.append(rec)
            
            # Sort by adjusted score
            diversified_recs.sort(key=lambda x: x['score'], reverse=True)
            
            return diversified_recs
            
        except Exception as e:
            logging.error(f"Error applying diversity filter: {e}")
            return recommendations
    
    def get_trending_based_recommendations(self, user_id, num_recommendations=10):
        """Get recommendations based on trending content"""
        try:
            user_data = self.load_user_data(user_id)
            user_genres = user_data['preferences']['genres']
            user_content_types = user_data['preferences']['content_types']
            
            # Get trending content from the last 7 days
            conn = self.get_database_connection()
            if not conn:
                return []
            
            query = """
            SELECT c.id, c.title, c.genres, c.content_type, c.rating,
                   COUNT(ui.id) as interaction_count
            FROM content c
            LEFT JOIN user_interaction ui ON c.id = ui.content_id
            WHERE c.cached_at > datetime('now', '-7 days')
            GROUP BY c.id
            ORDER BY interaction_count DESC, c.rating DESC
            LIMIT 50
            """
            
            cursor = conn.execute(query)
            trending_content = cursor.fetchall()
            
            recommendations = []
            for content in trending_content:
                genres = json.loads(content['genres']) if content['genres'] else []
                
                # Calculate relevance score based on user preferences
                relevance_score = 0
                
                # Genre relevance
                for genre in genres:
                    genre_name = genre['name'] if isinstance(genre, dict) else str(genre)
                    if genre_name in user_genres:
                        relevance_score += user_genres[genre_name] * 0.3
                
                # Content type relevance
                if content['content_type'] in user_content_types:
                    relevance_score += user_content_types[content['content_type']] * 0.2
                
                # Popularity boost
                popularity_score = content['interaction_count'] * 0.1
                
                # Rating boost
                rating_score = (content['rating'] or 0) * 0.1
                
                total_score = relevance_score + popularity_score + rating_score
                
                if total_score > 0.5:  # Minimum relevance threshold
                    recommendations.append({
                        'content_id': content['id'],
                        'score': total_score,
                        'reason': 'Trending content matching your interests',
                        'title': content['title']
                    })
            
            conn.close()
            
            # Sort by score and return top recommendations
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logging.error(f"Error in trending-based recommendations: {e}")
            return []
    
    def update_user_profile(self, user_id, interaction_data):
        """Update user profile based on new interaction"""
        try:
            # This method would be called when new interactions are recorded
            # to update the user's preference profile in real-time
            
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = self.load_user_data(user_id)
            
            # Update preferences based on new interaction
            content_id = interaction_data['content_id']
            interaction_type = interaction_data['interaction_type']
            value = interaction_data.get('value', 1.0)
            
            # Get content details
            conn = self.get_database_connection()
            if conn:
                cursor = conn.execute("""
                SELECT genres, content_type FROM content WHERE id = ?
                """, (content_id,))
                
                content = cursor.fetchone()
                if content:
                    genres = json.loads(content['genres']) if content['genres'] else []
                    content_type = content['content_type']
                    
                    weight = self.interaction_weights.get(interaction_type, 0.1)
                    
                    # Update genre preferences
                    for genre in genres:
                        genre_name = genre['name'] if isinstance(genre, dict) else str(genre)
                        self.user_profiles[user_id]['preferences']['genres'][genre_name] += weight * value
                    
                    # Update content type preferences
                    self.user_profiles[user_id]['preferences']['content_types'][content_type] += weight * value
                
                conn.close()
            
            # Clear cache for this user
            if user_id in recommendation_cache:
                del recommendation_cache[user_id]
            
            logging.info(f"Updated user profile for user {user_id}")
            
        except Exception as e:
            logging.error(f"Error updating user profile: {e}")


# Initialize recommendation engine
recommendation_engine = RecommendationEngine()

# Background tasks
def retrain_models():
    """Retrain models periodically"""
    while True:
        try:
            logging.info("Starting model retraining...")
            
            # Load fresh content data
            content_data = recommendation_engine.load_content_data(2000)
            
            if content_data:
                # Rebuild content features
                recommendation_engine.build_content_features(content_data)
                
                # Recalculate content similarity
                recommendation_engine.calculate_content_similarity(content_data)
                
                # Save updated models
                recommendation_engine.save_models()
                
                global last_model_update
                last_model_update = datetime.utcnow()
                
                logging.info("Model retraining completed")
            
            # Sleep for 6 hours
            time.sleep(6 * 60 * 60)
            
        except Exception as e:
            logging.error(f"Error in model retraining: {e}")
            time.sleep(30 * 60)  # Retry in 30 minutes

# Start background training thread
training_thread = threading.Thread(target=retrain_models, daemon=True)
training_thread.start()

# API Routes
@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'models_loaded': {
            'tfidf_vectorizer': tfidf_vectorizer is not None,
            'svd_model': svd_model is not None,
            'scaler': scaler is not None
        },
        'last_model_update': last_model_update.isoformat() if last_model_update else None
    })

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Get recommendations for a user"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        preferences = data.get('preferences', {})
        num_recommendations = data.get('num_recommendations', 20)
        
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        # Check cache first
        cache_key = f"{user_id}_{num_recommendations}"
        if cache_key in recommendation_cache:
            cached_result = recommendation_cache[cache_key]
            if (datetime.utcnow() - cached_result['timestamp']).seconds < 1800:  # 30 minutes
                return jsonify(cached_result['recommendations'])
        
        # Get hybrid recommendations
        recommendations = recommendation_engine.get_hybrid_recommendations(user_id, num_recommendations)
        
        # Add trending recommendations if we don't have enough
        if len(recommendations) < num_recommendations:
            trending_recs = recommendation_engine.get_trending_based_recommendations(
                user_id, num_recommendations - len(recommendations)
            )
            recommendations.extend(trending_recs)
        
        # Get content details for recommendations
        if recommendations:
            content_ids = [rec['content_id'] for rec in recommendations]
            conn = recommendation_engine.get_database_connection()
            
            if conn:
                placeholders = ','.join(['?' for _ in content_ids])
                query = f"""
                SELECT id, title, overview, poster_path, backdrop_path, rating, 
                       genres, content_type, source
                FROM content
                WHERE id IN ({placeholders})
                """
                
                cursor = conn.execute(query, content_ids)
                content_details = {}
                
                for row in cursor.fetchall():
                    content_details[row['id']] = {
                        'id': row['id'],
                        'title': row['title'],
                        'overview': row['overview'],
                        'poster_path': row['poster_path'],
                        'backdrop_path': row['backdrop_path'],
                        'rating': row['rating'],
                        'genres': json.loads(row['genres']) if row['genres'] else [],
                        'content_type': row['content_type'],
                        'source': row['source']
                    }
                
                conn.close()
                
                # Add content details to recommendations
                for rec in recommendations:
                    content_id = rec['content_id']
                    if content_id in content_details:
                        rec.update(content_details[content_id])
        
        # Cache the result
        result = {
            'recommendations': recommendations,
            'timestamp': datetime.utcnow(),
            'user_id': user_id
        }
        
        recommendation_cache[cache_key] = result
        
        return jsonify({'recommendations': recommendations})
        
    except Exception as e:
        logging.error(f"Error getting recommendations: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@app.route('/api/learn', methods=['POST'])
def learn_from_interaction():
    """Learn from user interaction"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        content_id = data.get('content_id')
        interaction_type = data.get('interaction_type')
        value = data.get('value')
        
        if not all([user_id, content_id, interaction_type]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Update user profile
        interaction_data = {
            'content_id': content_id,
            'interaction_type': interaction_type,
            'value': value
        }
        
        recommendation_engine.update_user_profile(user_id, interaction_data)
        
        return jsonify({'message': 'Learning completed'})
        
    except Exception as e:
        logging.error(f"Error learning from interaction: {e}")
        return jsonify({'error': 'Failed to learn from interaction'}), 500

@app.route('/api/similar', methods=['POST'])
def get_similar_content():
    """Get similar content based on content ID"""
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        num_similar = data.get('num_similar', 10)
        
        if not content_id:
            return jsonify({'error': 'Content ID is required'}), 400
        
        similar_content = []
        
        if content_id in recommendation_engine.content_similarity:
            similarities = recommendation_engine.content_similarity[content_id]['similarities']
            content_ids = recommendation_engine.content_similarity[content_id]['content_ids']
            
            # Get top similar content
            similar_indices = np.argsort(similarities)[::-1][1:num_similar+1]  # Skip self
            
            for idx in similar_indices:
                similar_content_id = content_ids[idx]
                similar_content.append({
                    'content_id': similar_content_id,
                    'similarity_score': similarities[idx]
                })
        
        # Get content details
        if similar_content:
            content_ids = [item['content_id'] for item in similar_content]
            conn = recommendation_engine.get_database_connection()
            
            if conn:
                placeholders = ','.join(['?' for _ in content_ids])
                query = f"""
                SELECT id, title, overview, poster_path, rating, content_type
                FROM content
                WHERE id IN ({placeholders})
                """
                
                cursor = conn.execute(query, content_ids)
                content_details = {}
                
                for row in cursor.fetchall():
                    content_details[row['id']] = {
                        'id': row['id'],
                        'title': row['title'],
                        'overview': row['overview'],
                        'poster_path': row['poster_path'],
                        'rating': row['rating'],
                        'content_type': row['content_type']
                    }
                
                conn.close()
                
                # Add content details
                for item in similar_content:
                    content_id = item['content_id']
                    if content_id in content_details:
                        item.update(content_details[content_id])
        
        return jsonify({'similar_content': similar_content})
        
    except Exception as e:
        logging.error(f"Error getting similar content: {e}")
        return jsonify({'error': 'Failed to get similar content'}), 500

@app.route('/api/user-profile/<int:user_id>')
def get_user_profile(user_id):
    """Get user preference profile"""
    try:
        user_data = recommendation_engine.load_user_data(user_id)
        
        profile = {
            'user_id': user_id,
            'preferences': {
                'top_genres': dict(sorted(user_data['preferences']['genres'].items(), 
                                        key=lambda x: x[1], reverse=True)[:10]),
                'content_types': dict(user_data['preferences']['content_types']),
                'average_rating': np.mean(user_data['preferences']['rating_patterns']) if user_data['preferences']['rating_patterns'] else 0
            },
            'interaction_stats': {
                'total_interactions': len(user_data['interactions']),
                'interaction_types': defaultdict(int)
            }
        }
        
        # Calculate interaction type stats
        for interaction in user_data['interactions']:
            profile['interaction_stats']['interaction_types'][interaction['interaction_type']] += 1
        
        return jsonify(profile)
        
    except Exception as e:
        logging.error(f"Error getting user profile: {e}")
        return jsonify({'error': 'Failed to get user profile'}), 500

@app.route('/api/retrain', methods=['POST'])
def trigger_retrain():
    """Manually trigger model retraining"""
    try:
        # This would typically be protected by admin authentication
        
        # Load fresh content data
        content_data = recommendation_engine.load_content_data(2000)
        
        if content_data:
            # Rebuild content features
            recommendation_engine.build_content_features(content_data)
            
            # Recalculate content similarity
            recommendation_engine.calculate_content_similarity(content_data)
            
            # Save updated models
            recommendation_engine.save_models()
            
            global last_model_update
            last_model_update = datetime.utcnow()
            
            # Clear recommendation cache
            recommendation_cache.clear()
            
            return jsonify({
                'message': 'Model retraining completed',
                'timestamp': last_model_update.isoformat()
            })
        else:
            return jsonify({'error': 'No content data available'}), 400
            
    except Exception as e:
        logging.error(f"Error in manual retraining: {e}")
        return jsonify({'error': 'Failed to retrain models'}), 500

@app.route('/api/search-suggestions', methods=['POST'])
def get_search_suggestions():
    """Get ML-powered search suggestions"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        user_id = data.get('user_id')
        
        if not query:
            return jsonify({'suggestions': []})
        
        suggestions = []
        
        # Get content matching the query
        conn = recommendation_engine.get_database_connection()
        if conn:
            search_query = f"%{query}%"
            cursor = conn.execute("""
            SELECT id, title, overview, content_type, rating
            FROM content
            WHERE title LIKE ? OR overview LIKE ?
            ORDER BY rating DESC
            LIMIT 10
            """, (search_query, search_query))
            
            results = cursor.fetchall()
            
            for row in results:
                suggestions.append({
                    'id': row['id'],
                    'title': row['title'],
                    'content_type': row['content_type'],
                    'rating': row['rating']
                })
            
            conn.close()
        
        # If user is logged in, personalize suggestions
        if user_id:
            user_data = recommendation_engine.load_user_data(user_id)
            user_genres = user_data['preferences']['genres']
            user_content_types = user_data['preferences']['content_types']
            
            # Score suggestions based on user preferences
            for suggestion in suggestions:
                relevance_score = 0
                
                # Content type preference
                if suggestion['content_type'] in user_content_types:
                    relevance_score += user_content_types[suggestion['content_type']] * 0.3
                
                # Rating boost
                relevance_score += (suggestion['rating'] or 0) * 0.1
                
                suggestion['relevance_score'] = relevance_score
            
            # Sort by relevance
            suggestions.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return jsonify({'suggestions': suggestions})
        
    except Exception as e:
        logging.error(f"Error getting search suggestions: {e}")
        return jsonify({'error': 'Failed to get search suggestions'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Initialize on startup
@app.before_first_request
def initialize_service():
    """Initialize the ML service"""
    try:
        # Load initial content data and build models
        content_data = recommendation_engine.load_content_data(1000)
        
        if content_data:
            recommendation_engine.build_content_features(content_data)
            recommendation_engine.calculate_content_similarity(content_data)
            recommendation_engine.save_models()
            
            logging.info("ML service initialized successfully")
        else:
            logging.warning("No content data found for initialization")
            
    except Exception as e:
        logging.error(f"Error initializing ML service: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host='0.0.0.0', port=5001)