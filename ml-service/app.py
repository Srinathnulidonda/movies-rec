#ml-service/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
import pickle
import json

app = Flask(__name__)
CORS(app)

# Sample movie data for recommendations (in production, this would be from a database)
SAMPLE_MOVIES = [
    {
        "id": 1,
        "title": "The Shawshank Redemption",
        "genres": "Drama",
        "overview": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
        "rating": 9.3,
        "type": "movie"
    },
    {
        "id": 2,
        "title": "The Godfather",
        "genres": "Crime Drama",
        "overview": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
        "rating": 9.2,
        "type": "movie"
    },
    {
        "id": 3,
        "title": "The Dark Knight",
        "genres": "Action Crime Drama",
        "overview": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests.",
        "rating": 9.0,
        "type": "movie"
    },
    {
        "id": 4,
        "title": "Spirited Away",
        "genres": "Animation Family Fantasy",
        "overview": "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits.",
        "rating": 9.3,
        "type": "anime"
    },
    {
        "id": 5,
        "title": "Breaking Bad",
        "genres": "Crime Drama Thriller",
        "overview": "A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing and selling methamphetamine.",
        "rating": 9.5,
        "type": "series"
    },
    {
        "id": 6,
        "title": "Game of Thrones",
        "genres": "Action Adventure Drama",
        "overview": "Nine noble families fight for control over the lands of Westeros, while an ancient enemy returns after being dormant for millennia.",
        "rating": 9.3,
        "type": "series"
    },
    {
        "id": 7,
        "title": "Attack on Titan",
        "genres": "Animation Action Drama",
        "overview": "After his hometown is destroyed and his mother is killed, young Eren Jaeger vows to cleanse the earth of the giant humanoid Titans.",
        "rating": 9.0,
        "type": "anime"
    },
    {
        "id": 8,
        "title": "Pulp Fiction",
        "genres": "Crime Drama",
        "overview": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
        "rating": 8.9,
        "type": "movie"
    },
    {
        "id": 9,
        "title": "The Matrix",
        "genres": "Action Sci-Fi",
        "overview": "A computer programmer is led to fight an underground war against powerful computers who have constructed his entire reality with a system called the Matrix.",
        "rating": 8.7,
        "type": "movie"
    },
    {
        "id": 10,
        "title": "Death Note",
        "genres": "Animation Crime Drama",
        "overview": "An intelligent high school student goes on a secret crusade to eliminate criminals from the world after discovering a notebook capable of killing anyone.",
        "rating": 9.0,
        "type": "anime"
    }
]

class MovieRecommender:
    def __init__(self):
        self.movies_df = pd.DataFrame(SAMPLE_MOVIES)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.train_model()
    
    def train_model(self):
        # Combine genres and overview for content-based filtering
        self.movies_df['content'] = self.movies_df['genres'] + ' ' + self.movies_df['overview']
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df['content'])
    
    def get_content_recommendations(self, movie_ids, num_recommendations=5):
        if not movie_ids:
            # Return popular movies if no history
            return self.movies_df.nlargest(num_recommendations, 'rating').to_dict('records')
        
        # Get average feature vector for user's movies
        user_profile = np.zeros(self.tfidf_matrix.shape[1])
        count = 0
        
        for movie_id in movie_ids:
            movie_idx = self.movies_df[self.movies_df['id'] == movie_id].index
            if not movie_idx.empty:
                movie_vector = self.tfidf_matrix[movie_idx[0]].toarray().flatten()
                user_profile += movie_vector
                count += 1
        
        if count > 0:
            user_profile /= count
        
        # Calculate similarity with all movies
        similarities = cosine_similarity([user_profile], self.tfidf_matrix).flatten()
        
        # Get recommendations excluding already seen movies
        movie_scores = list(enumerate(similarities))
        movie_scores = [(i, score) for i, score in movie_scores 
                       if self.movies_df.iloc[i]['id'] not in movie_ids]
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        recommendations = []
        for i, score in movie_scores[:num_recommendations]:
            movie = self.movies_df.iloc[i].to_dict()
            movie['similarity_score'] = float(score)
            recommendations.append(movie)
        
        return recommendations
    
    def get_collaborative_recommendations(self, user_favorites, num_recommendations=5):
        # Simple collaborative filtering based on favorites
        if not user_favorites:
            return self.get_content_recommendations([], num_recommendations)
        
        # Find movies with similar genres to favorites
        favorite_genres = set()
        for movie_id in user_favorites:
            movie = self.movies_df[self.movies_df['id'] == movie_id]
            if not movie.empty:
                genres = movie.iloc[0]['genres'].split()
                favorite_genres.update(genres)
        
        # Score movies based on genre overlap
        recommendations = []
        for _, movie in self.movies_df.iterrows():
            if movie['id'] not in user_favorites:
                movie_genres = set(movie['genres'].split())
                overlap = len(favorite_genres.intersection(movie_genres))
                if overlap > 0:
                    movie_dict = movie.to_dict()
                    movie_dict['genre_score'] = overlap
                    recommendations.append(movie_dict)
        
        # Sort by genre overlap and rating
        recommendations.sort(key=lambda x: (x['genre_score'], x['rating']), reverse=True)
        return recommendations[:num_recommendations]

# Initialize recommender
recommender = MovieRecommender()

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        watch_history = data.get('watch_history', [])
        favorites = data.get('favorites', [])
        
        # Get content-based recommendations
        content_recs = recommender.get_content_recommendations(watch_history + favorites, 5)
        
        # Get collaborative recommendations
        collab_recs = recommender.get_collaborative_recommendations(favorites, 3)
        
        # Combine and deduplicate
        all_recs = content_recs + collab_recs
        seen_ids = set()
        final_recs = []
        
        for rec in all_recs:
            if rec['id'] not in seen_ids:
                seen_ids.add(rec['id'])
                final_recs.append(rec)
        
        return jsonify({
            'recommendations': final_recs[:8],
            'user_id': user_id,
            'total_count': len(final_recs)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'ml-recommendation'})

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': 'Content-based + Collaborative Filtering',
        'total_movies': len(SAMPLE_MOVIES),
        'features': 'TF-IDF on genres and overview',
        'algorithms': ['Cosine Similarity', 'Genre Overlap']
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))