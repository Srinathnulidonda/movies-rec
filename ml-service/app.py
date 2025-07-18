## ml-service/app.py

import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from sqlalchemy import create_engine, text
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import faiss
import joblib
from scipy.sparse import csr_matrix, lil_matrix
from implicit import als, bpr, lmf
import json
import redis
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///movie_recommendations.db')
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Database connection
engine = create_engine(DATABASE_URL)

# Model storage
models = {
    'svd': None,
    'als': None,
    'bpr': None,
    'neural': None,
    'content_vectorizer': None,
    'content_features': None,
    'user_embeddings': None,
    'item_embeddings': None,
    'faiss_index': None
}

# Neural Network Architecture for Recommendations
class RecommendationNet(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=50, hidden_layers=[100, 50]):
        super(RecommendationNet, self).__init__()
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        self.fc_layers = nn.ModuleList()
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_layers:
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
            
        self.output = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        
        x = torch.cat([user_embeds, item_embeds], dim=1)
        
        for layer in self.fc_layers:
            x = layer(x)
            
        output = self.sigmoid(self.output(x))
        return output.squeeze()

class RatingDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)
        
    def __len__(self):
        return len(self.ratings)
        
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

# Recommendation Engine Class
class RecommendationEngine:
    def __init__(self):
        self.user_item_matrix = None
        self.content_features = None
        self.user_features = None
        self.item_mapping = {}
        self.user_mapping = {}
        self.reverse_item_mapping = {}
        self.reverse_user_mapping = {}
        
    def load_data(self):
        """Load data from database"""
        # Load ratings
        ratings_query = """
        SELECT user_id, content_id, rating 
        FROM ratings
        """
        ratings_df = pd.read_sql(ratings_query, engine)
        
        # Load content features
        content_query = """
        SELECT id, title, synopsis, genres, language, content_type, popularity_score
        FROM content
        """
        content_df = pd.read_sql(content_query, engine)
        
        # Load user features
        user_query = """
        SELECT id, preferred_languages, preferred_genres, region
        FROM users
        """
        user_df = pd.read_sql(user_query, engine)
        
        return ratings_df, content_df, user_df
        
    def create_user_item_matrix(self, ratings_df):
        """Create sparse user-item matrix"""
        # Create mappings
        unique_users = ratings_df['user_id'].unique()
        unique_items = ratings_df['content_id'].unique()
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Create sparse matrix
        row_indices = [self.user_mapping[user] for user in ratings_df['user_id']]
        col_indices = [self.item_mapping[item] for item in ratings_df['content_id']]
        
        self.user_item_matrix = csr_matrix(
            (ratings_df['rating'], (row_indices, col_indices)),
            shape=(len(unique_users), len(unique_items))
        )
        
        return self.user_item_matrix
        
    def create_content_features(self, content_df):
        """Create content feature matrix"""
        # Process genres
        content_df['genres'] = content_df['genres'].apply(
            lambda x: ' '.join(json.loads(x)) if x else ''
        )
        
        # Combine text features
        content_df['combined_features'] = (
            content_df['title'].fillna('') + ' ' +
            content_df['synopsis'].fillna('') + ' ' +
            content_df['genres'].fillna('') + ' ' +
            content_df['content_type'].fillna('')
        )
        
        # Create TF-IDF features
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_features = tfidf.fit_transform(content_df['combined_features'])
        
        # Add numerical features
        numerical_features = StandardScaler().fit_transform(
            content_df[['popularity_score']].fillna(0)
        )
        
        # Combine features
        from scipy.sparse import hstack
        self.content_features = hstack([tfidf_features, numerical_features])
        
        models['content_vectorizer'] = tfidf
        models['content_features'] = self.content_features
        
        return self.content_features
        
    def train_svd(self):
        """Train SVD model"""
        svd = TruncatedSVD(n_components=50, random_state=42)
        self.user_embeddings_svd = svd.fit_transform(self.user_item_matrix)
        self.item_embeddings_svd = svd.components_.T
        
        models['svd'] = svd
        models['user_embeddings'] = self.user_embeddings_svd
        models['item_embeddings'] = self.item_embeddings_svd
        
    def train_als(self):
        """Train Alternating Least Squares model"""
        als_model = als.AlternatingLeastSquares(
            factors=50, 
            regularization=0.1, 
            iterations=20,
            random_state=42
        )
        als_model.fit(self.user_item_matrix.T)
        
        models['als'] = als_model
        
    def train_bpr(self):
        """Train Bayesian Personalized Ranking model"""
        bpr_model = bpr.BayesianPersonalizedRanking(
            factors=50,
            learning_rate=0.01,
            regularization=0.01,
            iterations=100,
            random_state=42
        )
        bpr_model.fit(self.user_item_matrix.T)
        
        models['bpr'] = bpr_model
        
    def train_neural_network(self, ratings_df):
        """Train neural collaborative filtering model"""
        # Prepare data
        user_ids = [self.user_mapping[user] for user in ratings_df['user_id']]
        item_ids = [self.item_mapping[item] for item in ratings_df['content_id']]
        ratings = ratings_df['rating'].values / 10.0  # Normalize to 0-1
        
        # Create dataset
        dataset = RatingDataset(user_ids, item_ids, ratings)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        # Initialize model
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)
        model = RecommendationNet(n_users, n_items)
        
        # Training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(20):
            total_loss = 0
            for batch_users, batch_items, batch_ratings in dataloader:
                optimizer.zero_grad()
                predictions = model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        models['neural'] = model
        
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        # Use item embeddings from SVD
        item_embeddings = self.item_embeddings_svd.astype('float32')
        
        # Build index
        index = faiss.IndexFlatIP(item_embeddings.shape[1])
        faiss.normalize_L2(item_embeddings)
        index.add(item_embeddings)
        
        models['faiss_index'] = index
        
    def get_user_recommendations(self, user_id, n_recommendations=20):
        """Get recommendations for a user using hybrid approach"""
        recommendations = defaultdict(list)
        
        # Check if user exists
        if user_id not in self.user_mapping:
            # Cold start - return popular items
            popular_items = self.get_popular_items(n_recommendations)
            return popular_items
            
        user_idx = self.user_mapping[user_id]
        
        # SVD recommendations
        if models['svd']:
            user_embedding = models['user_embeddings'][user_idx]
            scores = np.dot(models['item_embeddings'], user_embedding)
            top_items = np.argsort(scores)[::-1][:n_recommendations * 2]
            
            for idx in top_items:
                item_id = self.reverse_item_mapping[idx]
                recommendations['svd'].append((item_id, float(scores[idx])))
                
        # ALS recommendations
        if models['als']:
            ids, scores = models['als'].recommend(
                user_idx, 
                self.user_item_matrix[user_idx], 
                N=n_recommendations * 2
            )
            
            for idx, score in zip(ids, scores):
                item_id = self.reverse_item_mapping[idx]
                recommendations['als'].append((item_id, float(score)))
                
        # Neural network recommendations
        if models['neural']:
            model = models['neural']
            model.eval()
            
            with torch.no_grad():
                user_tensor = torch.LongTensor([user_idx] * len(self.item_mapping))
                item_tensor = torch.LongTensor(list(range(len(self.item_mapping))))
                scores = model(user_tensor, item_tensor).numpy()
                
            top_items = np.argsort(scores)[::-1][:n_recommendations * 2]
            
            for idx in top_items:
                item_id = self.reverse_item_mapping[idx]
                recommendations['neural'].append((item_id, float(scores[idx])))
                
        # Combine recommendations with weighted ensemble
        final_scores = defaultdict(float)
        weights = {'svd': 0.3, 'als': 0.3, 'neural': 0.4}
        
        for method, items in recommendations.items():
            weight = weights.get(method, 0.33)
            for item_id, score in items:
                final_scores[item_id] += weight * score
                
        # Sort and return top recommendations
        sorted_recommendations = sorted(
            final_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n_recommendations]
        
        return [item_id for item_id, _ in sorted_recommendations]
        
    def get_similar_items(self, item_id, n_similar=10):
        """Get similar items using content-based and collaborative filtering"""
        similar_items = []
        
        if item_id not in self.item_mapping:
            return []
            
        item_idx = self.item_mapping[item_id]
        
        # Content-based similarity
        if models['content_features'] is not None:
            item_features = models['content_features'][item_idx]
            similarities = cosine_similarity(item_features, models['content_features'])[0]
            top_similar = np.argsort(similarities)[::-1][1:n_similar + 1]
            
            content_similar = [(self.reverse_item_mapping.get(idx, idx), float(similarities[idx])) 
                             for idx in top_similar if idx in self.reverse_item_mapping]
            
        # Collaborative filtering similarity using FAISS
        if models['faiss_index'] and models['item_embeddings'] is not None:
            item_embedding = models['item_embeddings'][item_idx:item_idx+1].astype('float32')
            faiss.normalize_L2(item_embedding)
            
            distances, indices = models['faiss_index'].search(item_embedding, n_similar + 1)
            
            collab_similar = [(self.reverse_item_mapping.get(idx, idx), float(dist)) 
                            for idx, dist in zip(indices[0][1:], distances[0][1:])
                            if idx in self.reverse_item_mapping]
            
        # Combine results
        all_similar = defaultdict(float)
        
        if content_similar:
            for item, score in content_similar:
                all_similar[item] += 0.5 * score
                
        if collab_similar:
            for item, score in collab_similar:
                all_similar[item] += 0.5 * score
                
        # Sort and return
        sorted_similar = sorted(all_similar.items(), key=lambda x: x[1], reverse=True)[:n_similar]
        
        return [item for item, _ in sorted_similar]
        
    def get_popular_items(self, n_items=20, region=None, language=None):
        """Get popular items for cold start"""
        query = """
        SELECT id 
        FROM content 
        WHERE 1=1
        """
        params = {}
        
        if region:
            query += " AND region = :region"
            params['region'] = region
            
        if language:
            query += " AND language = :language"
            params['language'] = language
            
        query += " ORDER BY popularity_score DESC LIMIT :limit"
        params['limit'] = n_items
        
        result = engine.execute(text(query), params)
        return [row[0] for row in result]
        
    def update_user_preference(self, user_id, item_id, rating):
        """Update user preferences in real-time"""
        # Add to user-item matrix
        if user_id in self.user_mapping and item_id in self.item_mapping:
            user_idx = self.user_mapping[user_id]
            item_idx = self.item_mapping[item_id]
            self.user_item_matrix[user_idx, item_idx] = rating
            
            # Clear user's cache
            redis_client.delete(f"ml_recommendations:{user_id}")

# Initialize recommendation engine
rec_engine = RecommendationEngine()

# Training endpoint
@app.route('/train', methods=['POST'])
def train_models():
    """Train all recommendation models"""
    try:
        # Load data
        ratings_df, content_df, user_df = rec_engine.load_data()
        
        if ratings_df.empty:
            return jsonify({'message': 'No ratings data available'}), 400
            
        # Create matrices
        rec_engine.create_user_item_matrix(ratings_df)
        rec_engine.create_content_features(content_df)
        
        # Train models
        rec_engine.train_svd()
        rec_engine.train_als()
        rec_engine.train_bpr()
        rec_engine.train_neural_network(ratings_df)
        rec_engine.build_faiss_index()
        
        # Save models
        save_models()
        
        return jsonify({'message': 'Models trained successfully'}), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# Recommendation endpoints
@app.route('/recommend/user', methods=['POST'])
def recommend_for_user():
    """Get recommendations for a specific user"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        n_recommendations = data.get('n_recommendations', 20)
        
        # Check cache
        cache_key = f"ml_recommendations:{user_id}"
        cached = redis_client.get(cache_key)
        if cached:
            return jsonify({'recommendations': json.loads(cached)}), 200
            
        # Get recommendations
        recommendations = rec_engine.get_user_recommendations(user_id, n_recommendations)
        
        # Cache results
        redis_client.setex(cache_key, 3600, json.dumps(recommendations))
        
        return jsonify({'recommendations': recommendations}), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/recommend/similar', methods=['POST'])
def recommend_similar_items():
    """Get similar items"""
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        n_similar = data.get('n_similar', 10)
        
        # Check cache
        cache_key = f"ml_similar:{content_id}"
        cached = redis_client.get(cache_key)
        if cached:
            return jsonify({'recommendations': json.loads(cached)}), 200
            
        # Get similar items
        similar_items = rec_engine.get_similar_items(content_id, n_similar)
        
        # Cache results
        redis_client.setex(cache_key, 3600, json.dumps(similar_items))
        
        return jsonify({'recommendations': similar_items}), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/recommend/popular', methods=['POST'])
def recommend_popular():
    """Get popular recommendations"""
    try:
        data = request.get_json()
        region = data.get('region')
        language = data.get('language')
        n_items = data.get('n_items', 20)
        
        popular_items = rec_engine.get_popular_items(n_items, region, language)
        
        return jsonify({'recommendations': popular_items}), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/learn/rating', methods=['POST'])
def learn_from_rating():
    """Learn from user rating in real-time"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        content_id = data.get('content_id')
        rating = data.get('rating')
        
        # Update user preference
        rec_engine.update_user_preference(user_id, content_id, rating)
        
        return jsonify({'message': 'Preference updated'}), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# Model management
def save_models():
    """Save trained models to disk"""
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save scikit-learn models
    if models['svd']:
        joblib.dump(models['svd'], os.path.join(model_dir, 'svd_model.pkl'))
    if models['content_vectorizer']:
        joblib.dump(models['content_vectorizer'], os.path.join(model_dir, 'content_vectorizer.pkl'))
        
    # Save embeddings
    if models['user_embeddings'] is not None:
        np.save(os.path.join(model_dir, 'user_embeddings.npy'), models['user_embeddings'])
    if models['item_embeddings'] is not None:
        np.save(os.path.join(model_dir, 'item_embeddings.npy'), models['item_embeddings'])
        
    # Save implicit models
    if models['als']:
        pickle.dump(models['als'], open(os.path.join(model_dir, 'als_model.pkl'), 'wb'))
    if models['bpr']:
        pickle.dump(models['bpr'], open(os.path.join(model_dir, 'bpr_model.pkl'), 'wb'))
        
    # Save neural network
    if models['neural']:
        torch.save(models['neural'].state_dict(), os.path.join(model_dir, 'neural_model.pth'))
        
    # Save FAISS index
    if models['faiss_index']:
        faiss.write_index(models['faiss_index'], os.path.join(model_dir, 'faiss_index.idx'))
        
    # Save mappings
    mappings = {
        'user_mapping': rec_engine.user_mapping,
        'item_mapping': rec_engine.item_mapping,
        'reverse_user_mapping': rec_engine.reverse_user_mapping,
        'reverse_item_mapping': rec_engine.reverse_item_mapping
    }
    pickle.dump(mappings, open(os.path.join(model_dir, 'mappings.pkl'), 'wb'))

def load_models():
    """Load pre-trained models from disk"""
    model_dir = 'models'
    
    try:
        # Load scikit-learn models
        if os.path.exists(os.path.join(model_dir, 'svd_model.pkl')):
            models['svd'] = joblib.load(os.path.join(model_dir, 'svd_model.pkl'))
        if os.path.exists(os.path.join(model_dir, 'content_vectorizer.pkl')):
            models['content_vectorizer'] = joblib.load(os.path.join(model_dir, 'content_vectorizer.pkl'))
            
        # Load embeddings
        if os.path.exists(os.path.join(model_dir, 'user_embeddings.npy')):
            models['user_embeddings'] = np.load(os.path.join(model_dir, 'user_embeddings.npy'))
        if os.path.exists(os.path.join(model_dir, 'item_embeddings.npy')):
            models['item_embeddings'] = np.load(os.path.join(model_dir, 'item_embeddings.npy'))
            
        # Load implicit models
        if os.path.exists(os.path.join(model_dir, 'als_model.pkl')):
            models['als'] = pickle.load(open(os.path.join(model_dir, 'als_model.pkl'), 'rb'))
        if os.path.exists(os.path.join(model_dir, 'bpr_model.pkl')):
            models['bpr'] = pickle.load(open(os.path.join(model_dir, 'bpr_model.pkl'), 'rb'))
            
        # Load FAISS index
        if os.path.exists(os.path.join(model_dir, 'faiss_index.idx')):
            models['faiss_index'] = faiss.read_index(os.path.join(model_dir, 'faiss_index.idx'))
            
        # Load mappings
        if os.path.exists(os.path.join(model_dir, 'mappings.pkl')):
            mappings = pickle.load(open(os.path.join(model_dir, 'mappings.pkl'), 'rb'))
            rec_engine.user_mapping = mappings['user_mapping']
            rec_engine.item_mapping = mappings['item_mapping']
            rec_engine.reverse_user_mapping = mappings['reverse_user_mapping']
            rec_engine.reverse_item_mapping = mappings['reverse_item_mapping']
            
        # Load neural network
        if os.path.exists(os.path.join(model_dir, 'neural_model.pth')):
            # Need to know model dimensions
            if rec_engine.user_mapping and rec_engine.item_mapping:
                n_users = len(rec_engine.user_mapping)
                n_items = len(rec_engine.item_mapping)
                model = RecommendationNet(n_users, n_items)
                model.load_state_dict(torch.load(os.path.join(model_dir, 'neural_model.pth')))
                model.eval()
                models['neural'] = model
                
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'svd': models['svd'] is not None,
            'als': models['als'] is not None,
            'bpr': models['bpr'] is not None,
            'neural': models['neural'] is not None,
            'faiss': models['faiss_index'] is not None
        }
    }), 200

# Initialize models on startup
@app.before_first_request
def initialize():
    """Initialize models on startup"""
    # Try to load existing models
    if not load_models():
        # If no models exist, train them
        try:
            ratings_df, content_df, user_df = rec_engine.load_data()
            if not ratings_df.empty:
                rec_engine.create_user_item_matrix(ratings_df)
                rec_engine.create_content_features(content_df)
                rec_engine.train_svd()
                rec_engine.build_faiss_index()
                save_models()
        except Exception as e:
            print(f"Error initializing models: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)