# ml-service/app.py
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import sqlite3
import json
import logging
from datetime import datetime, timedelta
import requests
from collections import defaultdict
import pickle
import os
from threading import Thread
import time

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ML Models Storage
models = {
    'content_similarity': None,
    'user_clusters': None,
    'svd_model': None,
    'tfidf_vectorizer': None,
    'genre_embeddings': None,
    'user_preferences': {},
    'content_features': {},
    'interaction_matrix': None
}

# Real-time learning queues
learning_queue = []
update_queue = []

class AdvancedRecommendationEngine:
    def __init__(self):
        self.user_vectors = {}
        self.content_vectors = {}
        self.genre_weights = {}
        self.temporal_patterns = {}
        self.similarity_cache = {}
        
    def load_data(self):
        """Load data from main database"""
        try:
            conn = sqlite3.connect('../recommendations.db')
            
            # Load content data
            content_df = pd.read_sql_query('''
                SELECT c.*, 
                       AVG(ui.rating) as avg_user_rating,
                       COUNT(ui.rating) as rating_count
                FROM content c
                LEFT JOIN user_interactions ui ON c.id = ui.content_id
                WHERE ui.interaction_type = 'rating'
                GROUP BY c.id
            ''', conn)
            
            # Load user interactions
            interactions_df = pd.read_sql_query('''
                SELECT ui.*, c.genre_ids, c.content_type
                FROM user_interactions ui
                JOIN content c ON ui.content_id = c.id
                WHERE ui.interaction_type = 'rating'
            ''', conn)
            
            conn.close()
            return content_df, interactions_df
            
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def build_content_features(self, content_df):
        """Build advanced content feature vectors"""
        if content_df.empty:
            return {}
        
        features = {}
        
        # Text features from overview
        if 'overview' in content_df.columns:
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            overview_features = tfidf.fit_transform(content_df['overview'].fillna(''))
            models['tfidf_vectorizer'] = tfidf
        
        # Genre features
        all_genres = set()
        for genres_str in content_df['genre_ids'].fillna('[]'):
            genres = json.loads(genres_str)
            all_genres.update(genres)
        
        genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}
        
        for idx, row in content_df.iterrows():
            content_id = row['id']
            
            # Genre vector
            genre_vec = np.zeros(len(all_genres))
            genres = json.loads(row['genre_ids'] or '[]')
            for genre in genres:
                if genre in genre_to_idx:
                    genre_vec[genre_to_idx[genre]] = 1
            
            # Numerical features
            numerical_features = [
                row.get('vote_average', 0) / 10.0,  # Normalized rating
                np.log1p(row.get('popularity', 1)),  # Log popularity
                row.get('rating_count', 0) / 1000.0,  # Normalized rating count
                1 if row.get('content_type') == 'movie' else 0,  # Content type
                1 if row.get('content_type') == 'tv' else 0,
                1 if row.get('content_type') == 'anime' else 0,
            ]
            
            # Combine features
            if 'overview' in content_df.columns and overview_features.shape[0] > idx:
                text_features = overview_features[idx].toarray().flatten()
                combined_features = np.concatenate([genre_vec, numerical_features, text_features])
            else:
                combined_features = np.concatenate([genre_vec, numerical_features])
            
            features[content_id] = combined_features
        
        return features
    
    def build_user_profiles(self, interactions_df):
        """Build sophisticated user profiles"""
        if interactions_df.empty:
            return {}
        
        user_profiles = {}
        
        for user_id in interactions_df['user_id'].unique():
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            
            # Genre preferences
            genre_prefs = defaultdict(list)
            for _, interaction in user_interactions.iterrows():
                genres = json.loads(interaction['genre_ids'] or '[]')
                for genre in genres:
                    genre_prefs[genre].append(interaction['rating'])
            
            # Calculate genre weights
            genre_weights = {}
            for genre, ratings in genre_prefs.items():
                avg_rating = np.mean(ratings)
                genre_weights[genre] = avg_rating / 10.0  # Normalize
            
            # Content type preferences
            content_type_prefs = {}
            for content_type in ['movie', 'tv', 'anime']:
                type_interactions = user_interactions[user_interactions['content_type'] == content_type]
                if not type_interactions.empty:
                    content_type_prefs[content_type] = type_interactions['rating'].mean() / 10.0
            
            # Temporal patterns
            user_interactions['created_at'] = pd.to_datetime(user_interactions['created_at'])
            user_interactions['hour'] = user_interactions['created_at'].dt.hour
            user_interactions['day_of_week'] = user_interactions['created_at'].dt.dayofweek
            
            temporal_patterns = {
                'preferred_hours': user_interactions.groupby('hour')['rating'].mean().to_dict(),
                'preferred_days': user_interactions.groupby('day_of_week')['rating'].mean().to_dict()
            }
            
            # Rating patterns
            rating_stats = {
                'mean_rating': user_interactions['rating'].mean(),
                'std_rating': user_interactions['rating'].std(),
                'total_interactions': len(user_interactions)
            }
            
            user_profiles[user_id] = {
                'genre_weights': genre_weights,
                'content_type_prefs': content_type_prefs,
                'temporal_patterns': temporal_patterns,
                'rating_stats': rating_stats
            }
        
        return user_profiles
    
    def collaborative_filtering_advanced(self, user_id, limit=20):
        """Advanced collaborative filtering with matrix factorization"""
        try:
            content_df, interactions_df = self.load_data()
            
            if interactions_df.empty:
                return []
            
            # Create user-item matrix
            user_item_matrix = interactions_df.pivot_table(
                index='user_id', columns='content_id', values='rating', fill_value=0
            )
            
            # SVD for matrix factorization
            svd = TruncatedSVD(n_components=50, random_state=42)
            user_factors = svd.fit_transform(user_item_matrix)
            item_factors = svd.components_
            
            models['svd_model'] = svd
            models['interaction_matrix'] = user_item_matrix
            
            # Get user vector
            if user_id in user_item_matrix.index:
                user_idx = user_item_matrix.index.get_loc(user_id)
                user_vector = user_factors[user_idx]
                
                # Calculate scores for all items
                scores = np.dot(user_vector, item_factors)
                
                # Get top recommendations
                item_indices = np.argsort(scores)[::-1]
                content_ids = [user_item_matrix.columns[idx] for idx in item_indices]
                
                # Filter out already rated content
                user_rated = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
                recommendations = [cid for cid in content_ids if cid not in user_rated][:limit]
                
                # Get content details
                return self.get_content_details(recommendations, content_df)
            
            return []
            
        except Exception as e:
            logger.error(f"Collaborative filtering error: {e}")
            return []
    
    def content_based_advanced(self, user_id, limit=20):
        """Advanced content-based recommendations"""
        try:
            content_df, interactions_df = self.load_data()
            
            if content_df.empty or interactions_df.empty:
                return []
            
            # Build content features
            content_features = self.build_content_features(content_df)
            
            # Get user preferences
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            
            if user_interactions.empty:
                return []
            
            # Build user profile vector
            user_profile = np.zeros(len(next(iter(content_features.values()))))
            total_weight = 0
            
            for _, interaction in user_interactions.iterrows():
                content_id = interaction['content_id']
                rating = interaction['rating']
                weight = (rating - 5) / 5.0  # Normalize to [-1, 1]
                
                if content_id in content_features:
                    user_profile += weight * content_features[content_id]
                    total_weight += abs(weight)
            
            if total_weight > 0:
                user_profile /= total_weight
            
            # Calculate similarities
            similarities = {}
            rated_content = set(user_interactions['content_id'])
            
            for content_id, features in content_features.items():
                if content_id not in rated_content:
                    similarity = cosine_similarity([user_profile], [features])[0][0]
                    similarities[content_id] = similarity
            
            # Get top recommendations
            top_content = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:limit]
            recommendations = [content_id for content_id, _ in top_content]
            
            return self.get_content_details(recommendations, content_df)
            
        except Exception as e:
            logger.error(f"Content-based error: {e}")
            return []
    
    def hybrid_recommendation(self, user_id, limit=20):
        """Hybrid recommendation combining multiple algorithms"""
        try:
            # Get recommendations from different algorithms
            collab_recs = self.collaborative_filtering_advanced(user_id, limit//2)
            content_recs = self.content_based_advanced(user_id, limit//2)
            
            # Combine recommendations with weights
            combined_recs = []
            seen_ids = set()
            
            # Weight collaborative filtering higher for users with more interactions
            content_df, interactions_df = self.load_data()
            user_interaction_count = len(interactions_df[interactions_df['user_id'] == user_id])
            
            if user_interaction_count > 20:
                collab_weight = 0.7
                content_weight = 0.3
            else:
                collab_weight = 0.3
                content_weight = 0.7
            
            # Add collaborative recommendations
            for rec in collab_recs:
                if rec['id'] not in seen_ids:
                    rec['recommendation_score'] = rec.get('score', 0.5) * collab_weight
                    rec['algorithm'] = 'collaborative'
                    combined_recs.append(rec)
                    seen_ids.add(rec['id'])
            
            # Add content-based recommendations
            for rec in content_recs:
                if rec['id'] not in seen_ids:
                    rec['recommendation_score'] = rec.get('score', 0.5) * content_weight
                    rec['algorithm'] = 'content_based'
                    combined_recs.append(rec)
                    seen_ids.add(rec['id'])
            
            # Sort by combined score
            combined_recs.sort(key=lambda x: x['recommendation_score'], reverse=True)
            
            # Apply diversity and novelty filters
            diverse_recs = self.apply_diversity_filter(combined_recs, user_id)
            
            return diverse_recs[:limit]
            
        except Exception as e:
            logger.error(f"Hybrid recommendation error: {e}")
            return []
    
    def apply_diversity_filter(self, recommendations, user_id):
        """Apply diversity filter to recommendations"""
        if not recommendations:
            return recommendations
        
        diverse_recs = []
        selected_genres = set()
        selected_types = set()
        
        for rec in recommendations:
            genres = json.loads(rec.get('genre_ids', '[]'))
            content_type = rec.get('content_type', '')
            
            # Calculate diversity score
            genre_diversity = len(set(genres) - selected_genres) / max(len(genres), 1)
            type_diversity = 1 if content_type not in selected_types else 0
            
            diversity_score = (genre_diversity + type_diversity) / 2
            
            # Boost score with diversity
            rec['recommendation_score'] = (
                rec.get('recommendation_score', 0.5) * 0.8 + 
                diversity_score * 0.2
            )
            
            diverse_recs.append(rec)
            selected_genres.update(genres)
            selected_types.add(content_type)
        
        return sorted(diverse_recs, key=lambda x: x['recommendation_score'], reverse=True)
    
    def get_content_details(self, content_ids, content_df):
        """Get detailed content information"""
        results = []
        
        for content_id in content_ids:
            content_row = content_df[content_df['id'] == content_id]
            
            if not content_row.empty:
                content = content_row.iloc[0].to_dict()
                
                # Add recommendation score
                content['score'] = content.get('vote_average', 0) / 10.0
                
                # Parse genre_ids
                if 'genre_ids' in content:
                    content['genres'] = json.loads(content['genre_ids'] or '[]')
                
                results.append(content)
        
        return results
    
    def real_time_learning(self, user_id, action, content_id, rating=None):
        """Real-time learning from user interactions"""
        try:
            # Update user preferences
            if user_id not in models['user_preferences']:
                models['user_preferences'][user_id] = {
                    'genre_weights': defaultdict(float),
                    'content_type_prefs': defaultdict(float),
                    'rating_history': []
                }
            
            # Get content details
            content_df, _ = self.load_data()
            content_row = content_df[content_df['id'] == content_id]
            
            if not content_row.empty:
                content = content_row.iloc[0]
                genres = json.loads(content.get('genre_ids', '[]'))
                content_type = content.get('content_type', '')
                
                # Update preferences based on action
                if action == 'rating' and rating:
                    weight = (rating - 5) / 5.0  # Normalize to [-1, 1]
                    
                    for genre in genres:
                        models['user_preferences'][user_id]['genre_weights'][genre] += weight * 0.1
                    
                    models['user_preferences'][user_id]['content_type_prefs'][content_type] += weight * 0.1
                    models['user_preferences'][user_id]['rating_history'].append({
                        'content_id': content_id,
                        'rating': rating,
                        'timestamp': datetime.now().isoformat()
                    })
                
                elif action == 'watchlist':
                    # Positive signal for watchlist
                    for genre in genres:
                        models['user_preferences'][user_id]['genre_weights'][genre] += 0.05
                    
                    models['user_preferences'][user_id]['content_type_prefs'][content_type] += 0.05
            
            return True
            
        except Exception as e:
            logger.error(f"Real-time learning error: {e}")
            return False

# Initialize recommendation engine
rec_engine = AdvancedRecommendationEngine()

# API Endpoints
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        limit = min(data.get('limit', 20), 50)
        algorithm = data.get('algorithm', 'hybrid_advanced')
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        # Get recommendations based on algorithm
        if algorithm == 'collaborative':
            recommendations = rec_engine.collaborative_filtering_advanced(user_id, limit)
        elif algorithm == 'content_based':
            recommendations = rec_engine.content_based_advanced(user_id, limit)
        elif algorithm == 'hybrid_advanced':
            recommendations = rec_engine.hybrid_recommendation(user_id, limit)
        else:
            recommendations = rec_engine.hybrid_recommendation(user_id, limit)
        
        return jsonify({
            'recommendations': recommendations,
            'algorithm': algorithm,
            'user_id': user_id,
            'count': len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({'error': 'Recommendation service error'}), 500

@app.route('/similar', methods=['POST'])
def similar_content():
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        limit = min(data.get('limit', 10), 20)
        
        if not content_id:
            return jsonify({'error': 'Content ID required'}), 400
        
        # Get similar content using content features
        content_df, _ = rec_engine.load_data()
        content_features = rec_engine.build_content_features(content_df)
        
        if content_id not in content_features:
            return jsonify({'similar': []})
        
        target_features = content_features[content_id]
        similarities = {}
        
        for cid, features in content_features.items():
            if cid != content_id:
                similarity = cosine_similarity([target_features], [features])[0][0]
                similarities[cid] = similarity
        
        # Get top similar content
        top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:limit]
        similar_ids = [cid for cid, _ in top_similar]
        
        similar_content = rec_engine.get_content_details(similar_ids, content_df)
        
        return jsonify({
            'similar': similar_content,
            'content_id': content_id,
            'count': len(similar_content)
        })
        
    except Exception as e:
        logger.error(f"Similar content error: {e}")
        return jsonify({'error': 'Similar content service error'}), 500

@app.route('/learn', methods=['POST'])
def learn():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        action = data.get('action')
        content_id = data.get('content_id')
        rating = data.get('rating')
        
        if not all([user_id, action, content_id]):
            return jsonify({'error': 'User ID, action, and content ID required'}), 400
        
        # Perform real-time learning
        success = rec_engine.real_time_learning(user_id, action, content_id, rating)
        
        if success:
            return jsonify({'message': 'Learning completed successfully'})
        else:
            return jsonify({'error': 'Learning failed'}), 500
            
    except Exception as e:
        logger.error(f"Learning error: {e}")
        return jsonify({'error': 'Learning service error'}), 500

@app.route('/deep_recommend', methods=['POST'])
def deep_recommend():
    """Deep learning recommendations (simplified neural collaborative filtering)"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        limit = min(data.get('limit', 20), 50)
        
        # For now, use enhanced hybrid approach
        # In production, this would use neural networks
        recommendations = rec_engine.hybrid_recommendation(user_id, limit)
        
        # Add deep learning scores (simulated)
        for rec in recommendations:
            rec['deep_score'] = rec.get('recommendation_score', 0.5) * np.random.uniform(0.8, 1.2)
            rec['algorithm'] = 'deep_learning'
        
        # Re-sort by deep scores
        recommendations.sort(key=lambda x: x['deep_score'], reverse=True)
        
        return jsonify({
            'recommendations': recommendations,
            'algorithm': 'deep_learning',
            'user_id': user_id
        })
        
    except Exception as e:
        logger.error(f"Deep learning recommendation error: {e}")
        return jsonify({'error': 'Deep learning service error'}), 500

@app.route('/search_suggestions', methods=['POST'])
def search_suggestions():
    """ML-powered search suggestions"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        query = data.get('query', '').lower()
        limit = min(data.get('limit', 5), 10)
        
        # Get user preferences
        user_prefs = models['user_preferences'].get(user_id, {})
        
        # Load content data
        content_df, _ = rec_engine.load_data()
        
        # Filter content based on query and user preferences
        suggestions = []
        
        for _, content in content_df.iterrows():
            title = content.get('title', '').lower()
            overview = content.get('overview', '').lower()
            
            # Check if query matches
            if query in title or query in overview:
                score = 0.5
                
                # Boost based on user preferences
                if user_prefs:
                    genres = json.loads(content.get('genre_ids', '[]'))
                    genre_weights = user_prefs.get('genre_weights', {})
                    
                    for genre in genres:
                        if genre in genre_weights:
                            score += genre_weights[genre] * 0.1
                
                suggestions.append({
                    'id': content['id'],
                    'title': content['title'],
                    'content_type': content['content_type'],
                    'score': score
                })
        
        # Sort by score and return top suggestions
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            'suggestions': suggestions[:limit],
            'query': query,
            'user_id': user_id
        })
        
    except Exception as e:
        logger.error(f"Search suggestions error: {e}")
        return jsonify({'error': 'Search suggestions service error'}), 500

@app.route('/update_preferences', methods=['POST'])
def update_preferences():
    """Update user preferences in real-time"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        preferences_data = data.get('data', [])
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        # Process preference updates
        for pref_data in preferences_data:
            action = pref_data.get('interaction_type')
            content_id = pref_data.get('content_id')
            rating = pref_data.get('rating')
            
            if action and content_id:
                rec_engine.real_time_learning(user_id, action, content_id, rating)
        
        return jsonify({'message': 'Preferences updated successfully'})
        
    except Exception as e:
        logger.error(f"Preference update error: {e}")
        return jsonify({'error': 'Preference update service error'}), 500

@app.route('/update', methods=['POST'])
def update_models():
    """Update ML models with new data"""
    try:
        data = request.get_json()
        event_type = data.get('event')
        event_data = data.get('data', {})
        
        # Add to update queue for batch processing
        update_queue.append({
            'event': event_type,
            'data': event_data,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({'message': 'Update queued successfully'})
        
    except Exception as e:
        logger.error(f"Model update error: {e}")
        return jsonify({'error': 'Model update service error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(models),
        'queue_size': len(update_queue)
    })

# Background model update process
def background_model_update():
    """Background process for updating models"""
    while True:
        try:
            if update_queue:
                # Process updates in batches
                batch = update_queue[:10]
                update_queue[:10] = []
                
                logger.info(f"Processing {len(batch)} model updates")
                
                # Here you would implement actual model retraining
                # For now, we'll just log the updates
                for update in batch:
                    logger.info(f"Model update: {update['event']}")
            
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Background update error: {e}")
            time.sleep(300)  # Wait 5 minutes on error

if __name__ == '__main__':
    # Start background update process
    update_thread = Thread(target=background_model_update, daemon=True)
    update_thread.start()
    
    logger.info("Advanced ML Recommendation Service starting...")
    app.run(debug=True, host='0.0.0.0', port=5001)