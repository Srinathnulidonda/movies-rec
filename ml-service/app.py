from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
import os
from datetime import datetime
import joblib
from collections import defaultdict
import psycopg2
from urllib.parse import urlparse

app = Flask(__name__)

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://movies_rec_mqs4_user:IhZQfhvaLsJ4O0bXGznHfSZjr0rym5QJ@dpg-d1l0vdvdiees73f2olp0-a/movies_rec_mqs4')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://')

def get_db_connection():
    """Get database connection"""
    result = urlparse(DATABASE_URL)
    conn = psycopg2.connect(
        database=result.path[1:],
        user=result.username,
        password=result.password,
        host=result.hostname,
        port=result.port
    )
    return conn

# ML Models Manager
class MLModelsManager:
    def __init__(self):
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.svd_model = None
        self.neural_model = None
        self.content_features = None
        self.user_features = None
        self.scaler = StandardScaler()
        self.knn_model = None
        
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models if they exist"""
        try:
            # Load SVD model
            svd_path = os.path.join(self.models_dir, 'svd_model.pkl')
            if os.path.exists(svd_path):
                self.svd_model = joblib.load(svd_path)
            
            # Load neural model
            neural_path = os.path.join(self.models_dir, 'neural_model.h5')
            if os.path.exists(neural_path):
                self.neural_model = tf.keras.models.load_model(neural_path)
            
            # Load features
            features_path = os.path.join(self.models_dir, 'features.pkl')
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    data = pickle.load(f)
                    self.content_features = data.get('content_features')
                    self.user_features = data.get('user_features')
            
            # Load KNN model
            knn_path = os.path.join(self.models_dir, 'knn_model.pkl')
            if os.path.exists(knn_path):
                self.knn_model = joblib.load(knn_path)
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def save_models(self):
        """Save trained models"""
        try:
            # Save SVD
            if self.svd_model:
                joblib.dump(self.svd_model, os.path.join(self.models_dir, 'svd_model.pkl'))
            
            # Save neural model
            if self.neural_model:
                self.neural_model.save(os.path.join(self.models_dir, 'neural_model.h5'))
            
            # Save features
            with open(os.path.join(self.models_dir, 'features.pkl'), 'wb') as f:
                pickle.dump({
                    'content_features': self.content_features,
                    'user_features': self.user_features
                }, f)
            
            # Save KNN
            if self.knn_model:
                joblib.dump(self.knn_model, os.path.join(self.models_dir, 'knn_model.pkl'))
                
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def build_neural_collaborative_filtering(self, num_users, num_items, embedding_dim=50):
        """Build neural collaborative filtering model"""
        # User input
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(num_users, embedding_dim, name='user_embedding')(user_input)
        user_vec = Flatten(name='user_flatten')(user_embedding)
        
        # Item input
        item_input = Input(shape=(1,), name='item_input')
        item_embedding = Embedding(num_items, embedding_dim, name='item_embedding')(item_input)
        item_vec = Flatten(name='item_flatten')(item_embedding)
        
        # Concatenate features
        concat = Concatenate()([user_vec, item_vec])
        
        # Deep layers
        fc1 = Dense(128, activation='relu')(concat)
        fc1 = Dropout(0.2)(fc1)
        fc2 = Dense(64, activation='relu')(fc1)
        fc2 = Dropout(0.2)(fc2)
        fc3 = Dense(32, activation='relu')(fc2)
        
        # Output
        output = Dense(1, activation='sigmoid')(fc3)
        
        # Create model
        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def extract_features(self):
        """Extract features from database"""
        conn = get_db_connection()
        
        try:
            # Get user-item interactions
            query = """
                SELECT user_id, content_id, 
                       CASE 
                           WHEN interaction_type = 'favorite' THEN 5
                           WHEN interaction_type = 'like' THEN 4
                           WHEN interaction_type = 'rating' THEN rating
                           WHEN interaction_type = 'view' THEN 3
                           ELSE 2
                       END as score
                FROM user_interaction
            """
            interactions_df = pd.read_sql(query, conn)
            
            # Get content features
            content_query = """
                SELECT id, genres, language, rating, popularity, runtime
                FROM content
            """
            content_df = pd.read_sql(content_query, conn)
            
            # Get user features
            user_query = """
                SELECT id, preferences, location
                FROM "user"
            """
            user_df = pd.read_sql(user_query, conn)
            
            conn.close()
            
            # Process features
            self.process_content_features(content_df)
            self.process_user_features(user_df)
            
            return interactions_df
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            if conn:
                conn.close()
            return None
    
    def process_content_features(self, content_df):
        """Process content features for ML"""
        features = []
        
        for _, row in content_df.iterrows():
            feature_vec = []
            
            # Genre features (one-hot encoding)
            genres = row['genres'] if row['genres'] else []
            genre_features = np.zeros(20)  # Top 20 genres
            if genres:
                for i, genre in enumerate(genres[:20]):
                    genre_features[i] = 1
            feature_vec.extend(genre_features)
            
            # Numerical features
            feature_vec.append(row['rating'] if row['rating'] else 0)
            feature_vec.append(row['popularity'] if row['popularity'] else 0)
            feature_vec.append(row['runtime'] if row['runtime'] else 0)
            
            # Language feature
            lang_map = {'en': 0, 'hi': 1, 'te': 2, 'ta': 3, 'kn': 4}
            lang_feature = lang_map.get(row['language'], 5)
            feature_vec.append(lang_feature)
            
            features.append(feature_vec)
        
        self.content_features = np.array(features)
        self.content_features = self.scaler.fit_transform(self.content_features)
    
    def process_user_features(self, user_df):
        """Process user features for ML"""
        features = []
        
        for _, row in user_df.iterrows():
            feature_vec = []
            
            # Extract preferences
            prefs = row['preferences'] if row['preferences'] else {}
            
            # Genre preferences
            genre_weights = prefs.get('genre_weights', {})
            genre_vec = np.zeros(20)
            for i, (genre, weight) in enumerate(list(genre_weights.items())[:20]):
                genre_vec[i] = weight
            feature_vec.extend(genre_vec)
            
            # Location feature
            location_map = {'US': 0, 'IN': 1, 'UK': 2, 'CA': 3, 'AU': 4}
            location = location_map.get(row['location'], 5)
            feature_vec.append(location)
            
            features.append(feature_vec)
        
        self.user_features = np.array(features)
    
    def train_svd(self, interactions_df):
        """Train SVD model"""
        # Create user-item matrix
        user_item_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='content_id',
            values='score',
            fill_value=0
        )
        
        # Train SVD
        self.svd_model = TruncatedSVD(n_components=50, random_state=42)
        user_factors = self.svd_model.fit_transform(user_item_matrix)
        
        # Store item factors
        self.item_factors = self.svd_model.components_.T
        self.user_factors = user_factors
        
        # Build KNN model for similar items
        self.knn_model = NearestNeighbors(n_neighbors=20, metric='cosine')
        self.knn_model.fit(self.item_factors)
        
        return True
    
    def train_neural_model(self, interactions_df):
        """Train neural collaborative filtering model"""
        # Prepare data
        num_users = interactions_df['user_id'].nunique()
        num_items = interactions_df['content_id'].nunique()
        
        # Create user and item mappings
        user_ids = interactions_df['user_id'].unique()
        item_ids = interactions_df['content_id'].unique()
        
        self.user_id_map = {uid: i for i, uid in enumerate(user_ids)}
        self.item_id_map = {iid: i for i, iid in enumerate(item_ids)}
        
        # Map IDs
        interactions_df['user_idx'] = interactions_df['user_id'].map(self.user_id_map)
        interactions_df['item_idx'] = interactions_df['content_id'].map(self.item_id_map)
        
        # Prepare training data
        X_user = interactions_df['user_idx'].values
        X_item = interactions_df['item_idx'].values
        y = (interactions_df['score'] > 3).astype(int).values
        
        # Build and train model
        self.neural_model = self.build_neural_collaborative_filtering(num_users, num_items)
        
        # Train with validation split
        history = self.neural_model.fit(
            [X_user, X_item],
            y,
            batch_size=64,
            epochs=10,
            validation_split=0.2,
            verbose=1
        )
        
        return True
    
    def get_svd_recommendations(self, user_id, limit=20):
        """Get recommendations using SVD"""
        if self.svd_model is None or self.user_factors is None:
            return []
        
        try:
            # Get user index
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get user's position in the matrix
            cursor.execute("""
                SELECT DISTINCT user_id FROM user_interaction ORDER BY user_id
            """)
            user_ids = [row[0] for row in cursor.fetchall()]
            
            if user_id not in user_ids:
                conn.close()
                return []
            
            user_idx = user_ids.index(user_id)
            
            # Get user's factor vector
            user_vec = self.user_factors[user_idx]
            
            # Calculate scores for all items
            scores = np.dot(self.item_factors, user_vec)
            
            # Get already interacted items
            cursor.execute("""
                SELECT DISTINCT content_id FROM user_interaction WHERE user_id = %s
            """, (user_id,))
            interacted_items = set(row[0] for row in cursor.fetchall())
            
            # Get all content IDs
            cursor.execute("""
                SELECT DISTINCT content_id FROM user_interaction ORDER BY content_id
            """)
            content_ids = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            # Sort and filter recommendations
            item_scores = [(content_ids[i], scores[i]) for i in range(len(scores)) 
                          if i < len(content_ids) and content_ids[i] not in interacted_items]
            
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [item_id for item_id, _ in item_scores[:limit]]
            
        except Exception as e:
            print(f"SVD recommendation error: {e}")
            return []
    
    def get_neural_recommendations(self, user_id, limit=20):
        """Get recommendations using neural network"""
        if self.neural_model is None:
            return []
        
        try:
            # Get user index
            if user_id not in self.user_id_map:
                return []
            
            user_idx = self.user_id_map[user_id]
            
            # Get predictions for all items
            all_items = np.array(list(self.item_id_map.values()))
            user_array = np.full(len(all_items), user_idx)
            
            predictions = self.neural_model.predict([user_array, all_items])
            
            # Get already interacted items
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT content_id FROM user_interaction WHERE user_id = %s
            """, (user_id,))
            interacted_items = set(row[0] for row in cursor.fetchall())
            conn.close()
            
            # Sort predictions
            item_predictions = []
            for item_id, item_idx in self.item_id_map.items():
                if item_id not in interacted_items:
                    score = predictions[item_idx][0]
                    item_predictions.append((item_id, score))
            
            item_predictions.sort(key=lambda x: x[1], reverse=True)
            
            return [item_id for item_id, _ in item_predictions[:limit]]
            
        except Exception as e:
            print(f"Neural recommendation error: {e}")
            return []
    
    def get_content_similarity_recommendations(self, content_id, limit=10):
        """Get similar content using KNN"""
        if self.knn_model is None or self.item_factors is None:
            return []
        
        try:
            # Get content index
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT content_id FROM user_interaction ORDER BY content_id
            """)
            content_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if content_id not in content_ids:
                return []
            
            content_idx = content_ids.index(content_id)
            
            # Find similar items
            distances, indices = self.knn_model.kneighbors(
                self.item_factors[content_idx].reshape(1, -1),
                n_neighbors=limit + 1
            )
            
            # Get content IDs (excluding the input content)
            similar_items = []
            for idx in indices[0][1:]:  # Skip first item (itself)
                if idx < len(content_ids):
                    similar_items.append(content_ids[idx])
            
            return similar_items
            
        except Exception as e:
            print(f"Content similarity error: {e}")
            return []
    
    def update_user_embedding(self, user_id, interaction_data):
        """Update user embeddings based on new interactions"""
        # This would be called after each interaction to update the model
        # In production, this might be done in batches
        pass

# Initialize ML manager
ml_manager = MLModelsManager()

# API Routes
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Get ML-based recommendations for a user"""
    data = request.get_json()
    user_id = data.get('user_id')
    limit = data.get('limit', 20)
    method = data.get('method', 'hybrid')  # svd, neural, hybrid
    
    if not user_id:
        return jsonify({'error': 'User ID required'}), 400
    
    recommendations = []
    
    if method == 'svd' or method == 'hybrid':
        svd_recs = ml_manager.get_svd_recommendations(user_id, limit)
        recommendations.extend(svd_recs)
    
    if method == 'neural' or method == 'hybrid':
        neural_recs = ml_manager.get_neural_recommendations(user_id, limit)
        recommendations.extend(neural_recs)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_recommendations = []
    for item in recommendations:
        if item not in seen:
            seen.add(item)
            unique_recommendations.append(item)
    
    return jsonify({
        'recommendations': unique_recommendations[:limit],
        'method': method
    })

@app.route('/similar/<int:content_id>', methods=['GET'])
def get_similar_content(content_id):
    """Get similar content based on content features"""
    limit = request.args.get('limit', 10, type=int)
    
    similar_items = ml_manager.get_content_similarity_recommendations(content_id, limit)
    
    return jsonify({
        'content_id': content_id,
        'similar': similar_items
    })

@app.route('/learn', methods=['POST'])
def learn_from_interaction():
    """Update models based on user interaction"""
    data = request.get_json()
    user_id = data.get('user_id')
    content_id = data.get('content_id')
    interaction_type = data.get('interaction_type')
    rating = data.get('rating')
    
    # In a production system, you would:
    # 1. Store this interaction
    # 2. Periodically retrain models
    # 3. Update user embeddings
    
    return jsonify({'status': 'recorded'})

@app.route('/train', methods=['POST'])
def train_models():
    """Train or retrain ML models"""
    try:
        # Extract features
        interactions_df = ml_manager.extract_features()
        
        if interactions_df is None or interactions_df.empty:
            return jsonify({'error': 'No data to train on'}), 400
        
        # Train models
        results = {}
        
        # Train SVD
        if ml_manager.train_svd(interactions_df):
            results['svd'] = 'trained'
        
        # Train neural model
        if ml_manager.train_neural_model(interactions_df):
            results['neural'] = 'trained'
        
        # Save models
        ml_manager.save_models()
        
        return jsonify({
            'status': 'success',
            'models': results,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_rating', methods=['POST'])
def predict_rating():
    """Predict user rating for a content item"""
    data = request.get_json()
    user_id = data.get('user_id')
    content_id = data.get('content_id')
    
    if not user_id or not content_id:
        return jsonify({'error': 'User ID and Content ID required'}), 400
    
    # Use neural model to predict rating
    if ml_manager.neural_model and user_id in ml_manager.user_id_map and content_id in ml_manager.item_id_map:
        user_idx = ml_manager.user_id_map[user_id]
        item_idx = ml_manager.item_id_map[content_id]
        
        prediction = ml_manager.neural_model.predict([[user_idx], [item_idx]])[0][0]
        
        # Convert to 1-5 scale
        rating = 1 + (prediction * 4)
        
        return jsonify({
            'user_id': user_id,
            'content_id': content_id,
            'predicted_rating': round(rating, 1)
        })
    
    return jsonify({'error': 'Unable to predict rating'}), 400

@app.route('/diversity_score', methods=['POST'])
def calculate_diversity_score():
    """Calculate diversity score for recommendations"""
    data = request.get_json()
    content_ids = data.get('content_ids', [])
    
    if not content_ids:
        return jsonify({'error': 'Content IDs required'}), 400
    
    # Get content features
    conn = get_db_connection()
    cursor = conn.cursor()
    
    placeholders = ','.join(['%s'] * len(content_ids))
    query = f"""
        SELECT id, genres, language, content_type 
        FROM content 
        WHERE id IN ({placeholders})
    """
    
    cursor.execute(query, content_ids)
    content_data = cursor.fetchall()
    conn.close()
    
    # Calculate diversity metrics
    genres = set()
    languages = set()
    types = set()
    
    for _, genre_list, language, content_type in content_data:
        if genre_list:
            for genre in genre_list:
                if isinstance(genre, dict):
                    genres.add(genre.get('name', ''))
                else:
                    genres.add(str(genre))
        languages.add(language)
        types.add(content_type)
    
    diversity_score = {
        'genre_diversity': len(genres) / max(len(content_ids), 1),
        'language_diversity': len(languages) / max(len(content_ids), 1),
        'type_diversity': len(types) / 3,  # Assuming 3 types: movie, tv, anime
        'overall_score': (len(genres) + len(languages) + len(types)) / (len(content_ids) + 6)
    }
    
    return jsonify(diversity_score)

@app.route('/trending_analysis', methods=['GET'])
def trending_analysis():
    """Analyze trending patterns"""
    days = request.args.get('days', 7, type=int)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get trending content based on recent interactions
    query = """
        SELECT c.id, c.title, c.genres, COUNT(ui.id) as interaction_count
        FROM content c
        JOIN user_interaction ui ON c.id = ui.content_id
        WHERE ui.created_at >= CURRENT_DATE - INTERVAL '%s days'
        GROUP BY c.id, c.title, c.genres
        ORDER BY interaction_count DESC
        LIMIT 20
    """
    
    cursor.execute(query, (days,))
    trending_content = cursor.fetchall()
    
    # Analyze genre trends
    genre_counts = defaultdict(int)
    for _, _, genres, count in trending_content:
        if genres:
            for genre in genres:
                if isinstance(genre, dict):
                    genre_counts[genre.get('name', 'Unknown')] += count
                else:
                    genre_counts[str(genre)] += count
    
    conn.close()
    
    return jsonify({
        'trending_content': [
            {
                'id': content_id,
                'title': title,
                'interactions': count
            }
            for content_id, title, _, count in trending_content
        ],
        'genre_trends': dict(genre_counts),
        'analysis_period': f'{days} days'
    })

@app.route('/cold_start/<int:user_id>', methods=['GET'])
def cold_start_recommendations(user_id):
    """Get recommendations for new users with little/no history"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get user's location
    cursor.execute("SELECT location FROM \"user\" WHERE id = %s", (user_id,))
    result = cursor.fetchone()
    user_location = result[0] if result else None
    
    # Get popular content in user's region
    if user_location:
        # Map location to language
        location_language_map = {
            'IN': ['hi', 'te', 'ta', 'kn'],
            'US': ['en'],
            'UK': ['en'],
            'JP': ['ja'],
            'KR': ['ko']
        }
        
        languages = location_language_map.get(user_location, ['en'])
        
        placeholders = ','.join(['%s'] * len(languages))
        query = f"""
            SELECT c.id
            FROM content c
            WHERE c.language IN ({placeholders})
            ORDER BY c.popularity DESC
            LIMIT 20
        """
        cursor.execute(query, languages)
    else:
        # Global popular content
        cursor.execute("""
            SELECT id FROM content 
            ORDER BY popularity DESC 
            LIMIT 20
        """)
    
    recommendations = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({
        'recommendations': recommendations,
        'type': 'cold_start',
        'based_on': 'location' if user_location else 'global_popularity'
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = {
        'svd': 'loaded' if ml_manager.svd_model is not None else 'not_loaded',
        'neural': 'loaded' if ml_manager.neural_model is not None else 'not_loaded',
        'knn': 'loaded' if ml_manager.knn_model is not None else 'not_loaded'
    }
    
    return jsonify({
        'status': 'healthy',
        'models': model_status,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/')
def index():
    return jsonify({
        'service': 'ML Recommendation Service',
        'version': '1.0',
        'endpoints': {
            'recommendations': '/recommend',
            'similar_content': '/similar/<content_id>',
            'train': '/train',
            'predict_rating': '/predict_rating',
            'diversity_score': '/diversity_score',
            'trending_analysis': '/trending_analysis',
            'cold_start': '/cold_start/<user_id>',
            'health': '/health'
        }
    })

if __name__ == '__main__':
    # Try to load or train models on startup
    if not ml_manager.svd_model and not ml_manager.neural_model:
        print("No pre-trained models found. Run /train endpoint to train models.")
    
    app.run(debug=True, host='0.0.0.0', port=5001)