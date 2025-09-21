#backend/ml_services/algorithm.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jaccard
from scipy import stats
import json
import math
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
import hashlib

logger = logging.getLogger(__name__)

class UserBehaviorTracker:
    """Advanced user behavior tracking system"""
    
    def __init__(self, db, models):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.behavior_weights = {
            'search': 0.8,      # High intent signal
            'view': 1.5,        # Engagement signal
            'like': 3.0,        # Positive preference
            'favorite': 5.0,    # Strong preference
            'watchlist': 3.5,   # Future intent
            'rating': lambda r: r * 1.2,  # Scaled by rating value
            'rewatch': 4.0,     # Strong satisfaction signal
            'share': 3.5,       # Social validation
            'dislike': -3.0,    # Negative preference
            'skip': -1.0        # Weak negative signal
        }
        self.recency_decay_factor = 0.95  # Daily decay
        self.user_profiles_cache = {}
        
    def get_comprehensive_user_profile(self, user_id):
        """Build comprehensive user profile from all interactions"""
        
        if user_id in self.user_profiles_cache:
            cache_time = self.user_profiles_cache[user_id].get('timestamp')
            if cache_time and (datetime.utcnow() - cache_time).seconds < 3600:
                return self.user_profiles_cache[user_id]
        
        profile = {
            'user_id': user_id,
            'search_keywords': [],
            'viewed_content': {},
            'genre_preferences': defaultdict(float),
            'language_preferences': defaultdict(float),
            'content_type_preferences': defaultdict(float),
            'actor_preferences': defaultdict(float),
            'director_preferences': defaultdict(float),
            'time_preferences': defaultdict(float),
            'rating_distribution': defaultdict(int),
            'interaction_patterns': defaultdict(list),
            'temporal_patterns': defaultdict(float),
            'quality_preferences': {
                'min_rating': 0,
                'avg_rating': 0,
                'prefer_new': False,
                'prefer_popular': False
            },
            'content_features': {
                'preferred_runtime': [],
                'preferred_years': [],
                'preferred_studios': []
            },
            'negative_preferences': {
                'avoided_genres': set(),
                'avoided_languages': set(),
                'avoided_actors': set()
            },
            'timestamp': datetime.utcnow()
        }
        
        # Get all user interactions
        interactions = self.UserInteraction.query.filter_by(
            user_id=user_id
        ).order_by(self.UserInteraction.timestamp.desc()).all()
        
        if not interactions:
            self.user_profiles_cache[user_id] = profile
            return profile
        
        # Process each interaction
        for interaction in interactions:
            content = self.Content.query.get(interaction.content_id)
            if not content:
                continue
            
            # Calculate time-based weight
            days_ago = (datetime.utcnow() - interaction.timestamp).days
            time_weight = self.recency_decay_factor ** days_ago
            
            # Get base weight for interaction type
            if interaction.interaction_type == 'rating' and interaction.rating:
                base_weight = self.behavior_weights['rating'](interaction.rating)
            else:
                base_weight = self.behavior_weights.get(interaction.interaction_type, 1.0)
            
            final_weight = base_weight * time_weight
            
            # Track search patterns
            if interaction.interaction_type == 'search':
                if hasattr(interaction, 'interaction_metadata') and interaction.interaction_metadata:
                    search_query = interaction.interaction_metadata.get('query', '')
                    if search_query:
                        profile['search_keywords'].append({
                            'keyword': search_query,
                            'timestamp': interaction.timestamp,
                            'weight': final_weight
                        })
            
            # Track viewed content with engagement metrics
            if interaction.interaction_type in ['view', 'rewatch']:
                profile['viewed_content'][content.id] = {
                    'title': content.title,
                    'engagement_score': final_weight,
                    'timestamp': interaction.timestamp,
                    'watch_count': profile['viewed_content'].get(content.id, {}).get('watch_count', 0) + 1
                }
            
            # Process content features
            self._process_content_features(content, profile, final_weight, interaction)
            
            # Track temporal patterns (when user interacts)
            hour = interaction.timestamp.hour
            day_of_week = interaction.timestamp.weekday()
            profile['temporal_patterns'][f'hour_{hour}'] += final_weight
            profile['temporal_patterns'][f'day_{day_of_week}'] += final_weight
            
            # Track rating distribution
            if interaction.rating:
                profile['rating_distribution'][int(interaction.rating)] += 1
        
        # Calculate quality preferences
        self._calculate_quality_preferences(profile, interactions)
        
        # Identify negative preferences
        self._identify_negative_preferences(profile, interactions)
        
        # Normalize preference scores
        self._normalize_preferences(profile)
        
        self.user_profiles_cache[user_id] = profile
        return profile
    
    def _process_content_features(self, content, profile, weight, interaction):
        """Extract and process content features for profile building"""
        
        # Handle negative signals
        if weight < 0:
            try:
                genres = json.loads(content.genres or '[]')
                for genre in genres:
                    profile['negative_preferences']['avoided_genres'].add(genre)
            except:
                pass
            return
        
        # Genre preferences
        try:
            genres = json.loads(content.genres or '[]')
            for genre in genres:
                profile['genre_preferences'][genre] += weight
        except:
            pass
        
        # Language preferences
        try:
            languages = json.loads(content.languages or '[]')
            for language in languages:
                profile['language_preferences'][language] += weight
        except:
            pass
        
        # Content type preferences
        profile['content_type_preferences'][content.content_type] += weight
        
        # Cast and crew preferences
        if hasattr(content, 'cast_crew'):
            for person in content.cast_crew:
                if person.role_type == 'cast':
                    profile['actor_preferences'][person.person_id] += weight * 0.7
                elif person.role_type == 'crew' and person.job == 'Director':
                    profile['director_preferences'][person.person_id] += weight * 0.5
        
        # Runtime preferences
        if content.runtime:
            profile['content_features']['preferred_runtime'].append({
                'runtime': content.runtime,
                'weight': weight
            })
        
        # Release year preferences
        if content.release_date:
            profile['content_features']['preferred_years'].append({
                'year': content.release_date.year,
                'weight': weight
            })
    
    def _calculate_quality_preferences(self, profile, interactions):
        """Calculate user's quality preferences"""
        
        ratings = []
        viewed_content_ratings = []
        
        for interaction in interactions:
            if interaction.rating:
                ratings.append(interaction.rating)
            
            if interaction.interaction_type in ['view', 'like', 'favorite']:
                content = self.Content.query.get(interaction.content_id)
                if content and content.rating:
                    viewed_content_ratings.append(content.rating)
        
        if ratings:
            profile['quality_preferences']['avg_user_rating'] = np.mean(ratings)
            profile['quality_preferences']['min_acceptable_rating'] = np.percentile(ratings, 25)
        
        if viewed_content_ratings:
            profile['quality_preferences']['avg_content_rating'] = np.mean(viewed_content_ratings)
            profile['quality_preferences']['prefer_highly_rated'] = np.mean(viewed_content_ratings) > 7.0
        
        # Check for recency preference
        recent_interactions = [i for i in interactions if (datetime.utcnow() - i.timestamp).days < 30]
        if recent_interactions:
            recent_years = []
            for interaction in recent_interactions[:20]:
                content = self.Content.query.get(interaction.content_id)
                if content and content.release_date:
                    recent_years.append(content.release_date.year)
            
            if recent_years:
                avg_year = np.mean(recent_years)
                profile['quality_preferences']['prefer_new'] = avg_year > (datetime.now().year - 3)
    
    def _identify_negative_preferences(self, profile, interactions):
        """Identify what user tends to avoid"""
        
        # Find content that was started but not finished (skipped)
        viewed_but_not_liked = set()
        liked_content = set()
        
        for interaction in interactions:
            if interaction.interaction_type in ['like', 'favorite', 'rating']:
                if interaction.interaction_type == 'rating' and interaction.rating and interaction.rating >= 3:
                    liked_content.add(interaction.content_id)
                elif interaction.interaction_type in ['like', 'favorite']:
                    liked_content.add(interaction.content_id)
            elif interaction.interaction_type == 'view':
                viewed_but_not_liked.add(interaction.content_id)
        
        skipped_content = viewed_but_not_liked - liked_content
        
        # Analyze skipped content for patterns
        for content_id in skipped_content:
            content = self.Content.query.get(content_id)
            if content:
                try:
                    genres = json.loads(content.genres or '[]')
                    for genre in genres:
                        profile['negative_preferences']['avoided_genres'].add(genre)
                except:
                    pass
    
    def _normalize_preferences(self, profile):
        """Normalize preference scores to 0-1 range"""
        
        for pref_type in ['genre_preferences', 'language_preferences', 
                         'content_type_preferences', 'actor_preferences', 
                         'director_preferences']:
            if profile[pref_type]:
                max_val = max(profile[pref_type].values())
                if max_val > 0:
                    for key in profile[pref_type]:
                        profile[pref_type][key] = profile[pref_type][key] / max_val

class AdvancedCollaborativeFiltering:
    """Enhanced collaborative filtering with multiple similarity metrics"""
    
    def __init__(self, db, models):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.behavior_tracker = UserBehaviorTracker(db, models)
        self.similarity_cache = {}
        
    def calculate_user_similarity_advanced(self, user_id, target_user_id):
        """Calculate similarity between two users using multiple metrics"""
        
        cache_key = f"{user_id}_{target_user_id}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        user1_profile = self.behavior_tracker.get_comprehensive_user_profile(user_id)
        user2_profile = self.behavior_tracker.get_comprehensive_user_profile(target_user_id)
        
        similarities = []
        
        # Genre similarity
        genre_sim = self._calculate_preference_similarity(
            user1_profile['genre_preferences'],
            user2_profile['genre_preferences']
        )
        similarities.append(genre_sim * 0.3)
        
        # Language similarity
        lang_sim = self._calculate_preference_similarity(
            user1_profile['language_preferences'],
            user2_profile['language_preferences']
        )
        similarities.append(lang_sim * 0.2)
        
        # Content type similarity
        type_sim = self._calculate_preference_similarity(
            user1_profile['content_type_preferences'],
            user2_profile['content_type_preferences']
        )
        similarities.append(type_sim * 0.15)
        
        # Viewed content overlap
        viewed1 = set(user1_profile['viewed_content'].keys())
        viewed2 = set(user2_profile['viewed_content'].keys())
        if viewed1 and viewed2:
            jaccard_sim = len(viewed1 & viewed2) / len(viewed1 | viewed2)
            similarities.append(jaccard_sim * 0.25)
        
        # Rating pattern similarity
        rating_sim = self._calculate_rating_similarity(
            user1_profile['rating_distribution'],
            user2_profile['rating_distribution']
        )
        similarities.append(rating_sim * 0.1)
        
        final_similarity = sum(similarities)
        self.similarity_cache[cache_key] = final_similarity
        
        return final_similarity
    
    def _calculate_preference_similarity(self, pref1, pref2):
        """Calculate similarity between two preference dictionaries"""
        
        if not pref1 or not pref2:
            return 0.0
        
        all_keys = set(pref1.keys()) | set(pref2.keys())
        if not all_keys:
            return 0.0
        
        vec1 = np.array([pref1.get(key, 0) for key in all_keys])
        vec2 = np.array([pref2.get(key, 0) for key in all_keys])
        
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _calculate_rating_similarity(self, ratings1, ratings2):
        """Calculate similarity in rating patterns"""
        
        if not ratings1 or not ratings2:
            return 0.0
        
        # Convert to probability distributions
        total1 = sum(ratings1.values())
        total2 = sum(ratings2.values())
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        prob1 = {k: v/total1 for k, v in ratings1.items()}
        prob2 = {k: v/total2 for k, v in ratings2.items()}
        
        # Calculate KL divergence
        kl_div = 0
        for rating in range(1, 11):
            p1 = prob1.get(rating, 0.001)
            p2 = prob2.get(rating, 0.001)
            if p1 > 0 and p2 > 0:
                kl_div += p1 * math.log(p1 / p2)
        
        # Convert to similarity (inverse of divergence)
        return 1 / (1 + kl_div)
    
    def get_user_based_recommendations(self, user_id, limit=50):
        """Get recommendations based on similar users' preferences"""
        
        user_profile = self.behavior_tracker.get_comprehensive_user_profile(user_id)
        
        # Find all users
        all_users = self.User.query.filter(self.User.id != user_id).all()
        
        # Calculate similarities with all users
        user_similarities = []
        for other_user in all_users:
            similarity = self.calculate_user_similarity_advanced(user_id, other_user.id)
            if similarity > 0.1:  # Minimum similarity threshold
                user_similarities.append((other_user.id, similarity))
        
        # Sort by similarity
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar_users = user_similarities[:min(50, len(user_similarities))]
        
        # Get recommendations from similar users
        recommendations = defaultdict(float)
        user_seen_content = set(user_profile['viewed_content'].keys())
        
        for similar_user_id, similarity in top_similar_users:
            # Get similar user's highly rated content
            similar_user_interactions = self.UserInteraction.query.filter_by(
                user_id=similar_user_id
            ).filter(
                self.UserInteraction.interaction_type.in_(['like', 'favorite', 'rating', 'watchlist'])
            ).all()
            
            for interaction in similar_user_interactions:
                if interaction.content_id not in user_seen_content:
                    # Calculate recommendation score
                    if interaction.interaction_type == 'rating' and interaction.rating:
                        score = similarity * interaction.rating / 10.0
                    elif interaction.interaction_type == 'favorite':
                        score = similarity * 1.0
                    elif interaction.interaction_type == 'like':
                        score = similarity * 0.8
                    elif interaction.interaction_type == 'watchlist':
                        score = similarity * 0.6
                    else:
                        score = similarity * 0.5
                    
                    recommendations[interaction.content_id] += score
        
        # Sort and return top recommendations
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:limit]

class DeepContentAnalyzer:
    """Advanced content analysis for better content-based filtering"""
    
    def __init__(self, db, models):
        self.db = db
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.Person = models.get('Person')
        self.ContentPerson = models.get('ContentPerson')
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000, 
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.content_embeddings = {}
        
    def create_content_embedding(self, content_id):
        """Create comprehensive content embedding"""
        
        if content_id in self.content_embeddings:
            return self.content_embeddings[content_id]
        
        content = self.Content.query.get(content_id)
        if not content:
            return None
        
        embedding = {}
        
        # Genre features (multi-hot encoding)
        try:
            genres = json.loads(content.genres or '[]')
            embedding['genres'] = set(genres)
        except:
            embedding['genres'] = set()
        
        # Language features
        try:
            languages = json.loads(content.languages or '[]')
            embedding['languages'] = set(languages)
        except:
            embedding['languages'] = set()
        
        # Cast and crew features
        embedding['cast'] = set()
        embedding['directors'] = set()
        
        if self.ContentPerson:
            cast_crew = self.ContentPerson.query.filter_by(content_id=content_id).all()
            for person_rel in cast_crew:
                if person_rel.role_type == 'cast':
                    embedding['cast'].add(person_rel.person_id)
                elif person_rel.job == 'Director':
                    embedding['directors'].add(person_rel.person_id)
        
        # Temporal features
        embedding['release_year'] = content.release_date.year if content.release_date else 0
        embedding['release_decade'] = (embedding['release_year'] // 10) * 10 if embedding['release_year'] else 0
        
        # Quality features
        embedding['rating'] = content.rating or 0
        embedding['popularity'] = content.popularity or 0
        embedding['vote_count'] = content.vote_count or 0
        
        # Content type
        embedding['content_type'] = content.content_type
        
        # Runtime features
        embedding['runtime'] = content.runtime or 0
        embedding['runtime_category'] = self._categorize_runtime(content.runtime)
        
        # Text features (overview)
        embedding['overview'] = content.overview or ''
        embedding['title'] = content.title or ''
        
        # Special features
        embedding['is_trending'] = content.is_trending
        embedding['is_new_release'] = content.is_new_release
        embedding['is_critics_choice'] = content.is_critics_choice
        
        self.content_embeddings[content_id] = embedding
        return embedding
    
    def _categorize_runtime(self, runtime):
        """Categorize runtime into buckets"""
        if not runtime:
            return 'unknown'
        elif runtime < 30:
            return 'short'
        elif runtime < 90:
            return 'medium'
        elif runtime < 150:
            return 'long'
        else:
            return 'very_long'
    
    def calculate_content_similarity_advanced(self, content_id1, content_id2):
        """Calculate advanced similarity between two content items"""
        
        embed1 = self.create_content_embedding(content_id1)
        embed2 = self.create_content_embedding(content_id2)
        
        if not embed1 or not embed2:
            return 0.0
        
        similarities = []
        
        # Genre similarity (Jaccard)
        if embed1['genres'] or embed2['genres']:
            genre_sim = len(embed1['genres'] & embed2['genres']) / max(
                len(embed1['genres'] | embed2['genres']), 1
            )
            similarities.append(('genre', genre_sim, 0.25))
        
        # Language similarity
        if embed1['languages'] or embed2['languages']:
            lang_sim = len(embed1['languages'] & embed2['languages']) / max(
                len(embed1['languages'] | embed2['languages']), 1
            )
            similarities.append(('language', lang_sim, 0.15))
        
        # Cast similarity
        if embed1['cast'] or embed2['cast']:
            cast_sim = len(embed1['cast'] & embed2['cast']) / max(
                len(embed1['cast'] | embed2['cast']), 1
            )
            similarities.append(('cast', cast_sim, 0.15))
        
        # Director similarity
        if embed1['directors'] or embed2['directors']:
            director_sim = 1.0 if embed1['directors'] & embed2['directors'] else 0.0
            similarities.append(('director', director_sim, 0.1))
        
        # Temporal similarity
        if embed1['release_year'] and embed2['release_year']:
            year_diff = abs(embed1['release_year'] - embed2['release_year'])
            temporal_sim = max(0, 1 - year_diff / 20)
            similarities.append(('temporal', temporal_sim, 0.1))
        
        # Quality similarity
        if embed1['rating'] and embed2['rating']:
            rating_diff = abs(embed1['rating'] - embed2['rating'])
            quality_sim = max(0, 1 - rating_diff / 5)
            similarities.append(('quality', quality_sim, 0.1))
        
        # Content type similarity
        type_sim = 1.0 if embed1['content_type'] == embed2['content_type'] else 0.3
        similarities.append(('type', type_sim, 0.1))
        
        # Text similarity (using TF-IDF on overview)
        if embed1['overview'] and embed2['overview']:
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([
                    embed1['overview'], embed2['overview']
                ])
                text_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                similarities.append(('text', text_sim, 0.05))
            except:
                pass
        
        # Calculate weighted sum
        total_weight = sum(weight for _, _, weight in similarities)
        if total_weight == 0:
            return 0.0
        
        final_similarity = sum(sim * weight for _, sim, weight in similarities) / total_weight
        
        return final_similarity

class PersonalizedRecommendationEngine:
    """Main engine for 100% personalized recommendations"""
    
    def __init__(self, db, models):
        self.db = db
        self.models = models
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.behavior_tracker = UserBehaviorTracker(db, models)
        self.collaborative_filter = AdvancedCollaborativeFiltering(db, models)
        self.content_analyzer = DeepContentAnalyzer(db, models)
        self.recommendation_cache = {}
        
    def get_personalized_recommendations(self, user_id, content_type='all', limit=20):
        """Get highly personalized recommendations based on comprehensive user profile"""
        
        # Get user profile
        user_profile = self.behavior_tracker.get_comprehensive_user_profile(user_id)
        
        # Initialize recommendation scores
        recommendation_scores = defaultdict(float)
        recommendation_reasons = defaultdict(list)
        
        # 1. Search-based recommendations (highest priority)
        search_recommendations = self._get_search_based_recommendations(user_profile)
        for content_id, score in search_recommendations.items():
            recommendation_scores[content_id] += score * 1.5  # Boost search-based
            recommendation_reasons[content_id].append("Matches your searches")
        
        # 2. Content-based recommendations from viewed/liked content
        content_recommendations = self._get_content_based_recommendations(user_profile)
        for content_id, score in content_recommendations.items():
            recommendation_scores[content_id] += score * 1.2
            recommendation_reasons[content_id].append("Similar to what you've watched")
        
        # 3. Genre and language preferences
        preference_recommendations = self._get_preference_based_recommendations(user_profile)
        for content_id, score in preference_recommendations.items():
            recommendation_scores[content_id] += score * 1.0
            recommendation_reasons[content_id].append("Matches your preferences")
        
        # 4. Collaborative filtering from similar users
        collaborative_recommendations = self.collaborative_filter.get_user_based_recommendations(user_id)
        for content_id, score in collaborative_recommendations[:50]:
            recommendation_scores[content_id] += score * 0.8
            recommendation_reasons[content_id].append("Popular with similar users")
        
        # 5. Trending and new releases in preferred categories
        trending_recommendations = self._get_trending_in_preferences(user_profile)
        for content_id, score in trending_recommendations.items():
            recommendation_scores[content_id] += score * 0.6
            recommendation_reasons[content_id].append("Trending in your interests")
        
        # Filter out already seen content
        seen_content = set(user_profile['viewed_content'].keys())
        
        # Filter out content from avoided categories
        avoided_genres = user_profile['negative_preferences']['avoided_genres']
        
        # Apply content type filter
        if content_type != 'all':
            filtered_scores = {}
            for content_id, score in recommendation_scores.items():
                content = self.Content.query.get(content_id)
                if content and content.content_type == content_type:
                    # Check if content has avoided genres
                    try:
                        content_genres = set(json.loads(content.genres or '[]'))
                        if not (content_genres & avoided_genres):
                            filtered_scores[content_id] = score
                    except:
                        filtered_scores[content_id] = score
            recommendation_scores = filtered_scores
        else:
            # Remove content with avoided genres
            filtered_scores = {}
            for content_id, score in recommendation_scores.items():
                if content_id not in seen_content:
                    content = self.Content.query.get(content_id)
                    if content:
                        try:
                            content_genres = set(json.loads(content.genres or '[]'))
                            if not (content_genres & avoided_genres):
                                filtered_scores[content_id] = score
                        except:
                            filtered_scores[content_id] = score
            recommendation_scores = filtered_scores
        
        # Apply quality filters based on user preferences
        if user_profile['quality_preferences'].get('prefer_highly_rated'):
            filtered_scores = {}
            min_rating = user_profile['quality_preferences'].get('min_acceptable_rating', 6.0)
            for content_id, score in recommendation_scores.items():
                content = self.Content.query.get(content_id)
                if content and content.rating and content.rating >= min_rating:
                    filtered_scores[content_id] = score
            recommendation_scores = filtered_scores
        
        # Sort by score
        sorted_recommendations = sorted(
            recommendation_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:limit]
        
        # Build final recommendation list with metadata
        final_recommendations = []
        for content_id, score in sorted_recommendations:
            content = self.Content.query.get(content_id)
            if content:
                final_recommendations.append({
                    'content': content,
                    'score': score,
                    'reasons': recommendation_reasons[content_id],
                    'confidence': min(score / 10.0, 1.0),
                    'personalization_level': 'high'
                })
        
        return final_recommendations
    
    def _get_search_based_recommendations(self, user_profile):
        """Get recommendations based on user's search history"""
        
        recommendations = defaultdict(float)
        
        if not user_profile['search_keywords']:
            return recommendations
        
        # Analyze recent searches
        recent_searches = sorted(
            user_profile['search_keywords'], 
            key=lambda x: x['timestamp'], 
            reverse=True
        )[:20]
        
        for search in recent_searches:
            keyword = search['keyword'].lower()
            weight = search['weight']
            
            # Search for content matching keywords
            matching_content = self.Content.query.filter(
                self.Content.title.ilike(f'%{keyword}%')
            ).limit(10).all()
            
            for content in matching_content:
                recommendations[content.id] += weight * 2.0
            
            # Also search in overview
            overview_matches = self.Content.query.filter(
                self.Content.overview.ilike(f'%{keyword}%')
            ).limit(5).all()
            
            for content in overview_matches:
                recommendations[content.id] += weight * 1.0
            
            # Search for similar genres if keyword matches genre names
            genre_matches = self.Content.query.filter(
                self.Content.genres.ilike(f'%{keyword}%')
            ).limit(10).all()
            
            for content in genre_matches:
                recommendations[content.id] += weight * 1.5
        
        return recommendations
    
    def _get_content_based_recommendations(self, user_profile):
        """Get recommendations based on content similarity to viewed/liked items"""
        
        recommendations = defaultdict(float)
        
        # Get user's top interacted content
        positive_interactions = self.UserInteraction.query.filter_by(
            user_id=user_profile['user_id']
        ).filter(
            self.UserInteraction.interaction_type.in_(['like', 'favorite', 'rating', 'watchlist'])
        ).order_by(
            self.UserInteraction.timestamp.desc()
        ).limit(30).all()
        
        for interaction in positive_interactions:
            base_content_id = interaction.content_id
            
            # Find similar content
            all_content = self.Content.query.limit(1000).all()
            
            for content in all_content:
                if content.id != base_content_id and content.id not in user_profile['viewed_content']:
                    similarity = self.content_analyzer.calculate_content_similarity_advanced(
                        base_content_id, content.id
                    )
                    
                    if similarity > 0.3:  # Minimum similarity threshold
                        # Weight by interaction type and similarity
                        if interaction.interaction_type == 'favorite':
                            weight = 3.0
                        elif interaction.interaction_type == 'like':
                            weight = 2.0
                        elif interaction.interaction_type == 'rating' and interaction.rating:
                            weight = interaction.rating / 5.0
                        else:
                            weight = 1.0
                        
                        recommendations[content.id] += similarity * weight
        
        return recommendations
    
    def _get_preference_based_recommendations(self, user_profile):
        """Get recommendations based on user's genre, language, and type preferences"""
        
        recommendations = defaultdict(float)
        
        # Genre-based recommendations
        top_genres = sorted(
            user_profile['genre_preferences'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        for genre, preference_score in top_genres:
            genre_content = self.Content.query.filter(
                self.Content.genres.contains(genre)
            ).order_by(
                self.Content.rating.desc(),
                self.Content.popularity.desc()
            ).limit(20).all()
            
            for content in genre_content:
                if content.id not in user_profile['viewed_content']:
                    recommendations[content.id] += preference_score * 2.0
        
        # Language-based recommendations
        top_languages = sorted(
            user_profile['language_preferences'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        for language, preference_score in top_languages:
            lang_content = self.Content.query.filter(
                self.Content.languages.contains(language)
            ).order_by(
                self.Content.rating.desc()
            ).limit(15).all()
            
            for content in lang_content:
                if content.id not in user_profile['viewed_content']:
                    recommendations[content.id] += preference_score * 1.5
        
        # Content type based recommendations
        for content_type, preference_score in user_profile['content_type_preferences'].items():
            type_content = self.Content.query.filter_by(
                content_type=content_type
            ).order_by(
                self.Content.rating.desc()
            ).limit(10).all()
            
            for content in type_content:
                if content.id not in user_profile['viewed_content']:
                    recommendations[content.id] += preference_score * 1.0
        
        return recommendations
    
    def _get_trending_in_preferences(self, user_profile):
        """Get trending content in user's preferred categories"""
        
        recommendations = defaultdict(float)
        
        # Get trending content
        trending_content = self.Content.query.filter_by(
            is_trending=True
        ).all()
        
        for content in trending_content:
            if content.id not in user_profile['viewed_content']:
                score = 0.0
                
                # Check genre match
                try:
                    content_genres = set(json.loads(content.genres or '[]'))
                    for genre in content_genres:
                        if genre in user_profile['genre_preferences']:
                            score += user_profile['genre_preferences'][genre] * 2.0
                except:
                    pass
                
                # Check language match
                try:
                    content_languages = set(json.loads(content.languages or '[]'))
                    for language in content_languages:
                        if language in user_profile['language_preferences']:
                            score += user_profile['language_preferences'][language] * 1.5
                except:
                    pass
                
                # Check content type match
                if content.content_type in user_profile['content_type_preferences']:
                    score += user_profile['content_type_preferences'][content.content_type] * 1.0
                
                if score > 0:
                    recommendations[content.id] = score
        
        # Get new releases in preferred categories
        new_releases = self.Content.query.filter_by(
            is_new_release=True
        ).all()
        
        for content in new_releases:
            if content.id not in user_profile['viewed_content']:
                score = 0.0
                
                # Similar scoring for new releases
                try:
                    content_genres = set(json.loads(content.genres or '[]'))
                    for genre in content_genres:
                        if genre in user_profile['genre_preferences']:
                            score += user_profile['genre_preferences'][genre] * 1.5
                except:
                    pass
                
                if score > 0:
                    recommendations[content.id] += score
        
        return recommendations
    
    def get_category_specific_recommendations(self, user_id, category='movies', limit=20):
        """Get recommendations for specific content categories"""
        
        content_type_map = {
            'movies': 'movie',
            'tv_shows': 'tv',
            'anime': 'anime'
        }
        
        content_type = content_type_map.get(category, 'movie')
        
        return self.get_personalized_recommendations(
            user_id=user_id,
            content_type=content_type,
            limit=limit
        )
    
    def explain_recommendation(self, user_id, content_id):
        """Provide detailed explanation for why content was recommended"""
        
        user_profile = self.behavior_tracker.get_comprehensive_user_profile(user_id)
        content = self.Content.query.get(content_id)
        
        if not content:
            return "Content not found"
        
        explanations = []
        
        # Check search match
        for search in user_profile['search_keywords']:
            if search['keyword'].lower() in content.title.lower():
                explanations.append(f"Matches your search for '{search['keyword']}'")
                break
        
        # Check genre match
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            matching_genres = content_genres & set(user_profile['genre_preferences'].keys())
            if matching_genres:
                explanations.append(f"You enjoy {', '.join(list(matching_genres)[:2])} content")
        except:
            pass
        
        # Check similar content
        similar_watched = []
        for watched_id in user_profile['viewed_content'].keys():
            similarity = self.content_analyzer.calculate_content_similarity_advanced(
                content_id, watched_id
            )
            if similarity > 0.5:
                watched_content = self.Content.query.get(watched_id)
                if watched_content:
                    similar_watched.append(watched_content.title)
        
        if similar_watched:
            explanations.append(f"Similar to {similar_watched[0]}")
        
        # Check quality match
        if content.rating and content.rating >= 8.0:
            if user_profile['quality_preferences'].get('prefer_highly_rated'):
                explanations.append("Highly rated content you might enjoy")
        
        # Check trending
        if content.is_trending:
            explanations.append("Currently trending")
        
        if not explanations:
            explanations.append("Recommended based on your viewing patterns")
        
        return " • ".join(explanations)