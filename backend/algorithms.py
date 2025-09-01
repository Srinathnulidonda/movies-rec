# backend/algorithms.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
import logging
import math
from typing import List, Dict, Any, Tuple, Optional
import hashlib

logger = logging.getLogger(__name__)

# Language Priority Configuration
LANGUAGE_WEIGHTS = {
    'telugu': 1.0,
    'te': 1.0,
    'english': 0.8,
    'en': 0.8,
    'hindi': 0.8,
    'hi': 0.8,
    'malayalam': 0.6,
    'ml': 0.6,
    'kannada': 0.6,
    'kn': 0.6,
    'tamil': 0.6,
    'ta': 0.6
}

class ContentBasedFiltering:
    """Content-Based Filtering Strategy with Feature Extraction and Similarity Computation"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.genre_weights = 0.4
        self.keyword_weights = 0.25
        self.cast_director_weights = 0.2
        self.plot_weights = 0.15
        
    def extract_features(self, content_list: List[Any]) -> Dict[int, Dict]:
        """Extract features from content for vectorization"""
        features = {}
        
        for content in content_list:
            content_features = {
                'genres': json.loads(content.genres or '[]'),
                'languages': json.loads(content.languages or '[]'),
                'overview': content.overview or '',
                'rating': content.rating or 0,
                'year': content.release_date.year if content.release_date else 0,
                'popularity': content.popularity or 0,
                'content_type': content.content_type
            }
            
            # Create text representation for TF-IDF
            text_features = ' '.join([
                ' '.join(content_features['genres']),
                content_features['overview'],
                content_features['content_type']
            ])
            
            features[content.id] = {
                'content': content,
                'features': content_features,
                'text': text_features
            }
        
        return features
    
    def compute_similarity(self, content_features: Dict, user_profile: Dict) -> float:
        """Compute similarity between content and user profile"""
        score = 0.0
        
        # Genre similarity
        if user_profile.get('preferred_genres') and content_features.get('genres'):
            genre_overlap = len(set(user_profile['preferred_genres']) & set(content_features['genres']))
            max_genres = max(len(user_profile['preferred_genres']), len(content_features['genres']))
            if max_genres > 0:
                score += self.genre_weights * (genre_overlap / max_genres)
        
        # Language similarity
        if user_profile.get('preferred_languages') and content_features.get('languages'):
            lang_overlap = len(set(user_profile['preferred_languages']) & set(content_features['languages']))
            if lang_overlap > 0:
                score += 0.2  # Bonus for language match
        
        # Rating preference
        if user_profile.get('avg_rating') and content_features.get('rating'):
            rating_diff = abs(user_profile['avg_rating'] - content_features['rating'])
            score += (1 - rating_diff / 10) * 0.1  # Max 0.1 for rating similarity
        
        return score
    
    def get_recommendations(self, user_profile: Dict, content_list: List[Any], limit: int = 20) -> List[Tuple[Any, float]]:
        """Get content-based recommendations for a user"""
        try:
            # Extract features
            features = self.extract_features(content_list)
            
            # Calculate scores
            recommendations = []
            for content_id, content_data in features.items():
                similarity_score = self.compute_similarity(
                    content_data['features'],
                    user_profile
                )
                recommendations.append((content_data['content'], similarity_score))
            
            # Sort by score
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Content-based filtering error: {e}")
            return []

class CollaborativeFiltering:
    """Collaborative Filtering Strategy with User-Based and Item-Based approaches"""
    
    def __init__(self):
        self.min_common_items = 3
        self.k_neighbors = 10
        
    def compute_user_similarity(self, user1_ratings: Dict, user2_ratings: Dict) -> float:
        """Compute Pearson correlation between two users"""
        common_items = set(user1_ratings.keys()) & set(user2_ratings.keys())
        
        if len(common_items) < self.min_common_items:
            return 0.0
        
        ratings1 = [user1_ratings[item] for item in common_items]
        ratings2 = [user2_ratings[item] for item in common_items]
        
        try:
            correlation, _ = pearsonr(ratings1, ratings2)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def user_based_cf(self, target_user_id: int, user_ratings: Dict[int, Dict], item_list: List[Any]) -> List[Tuple[Any, float]]:
        """User-based collaborative filtering"""
        if target_user_id not in user_ratings:
            return []
        
        target_ratings = user_ratings[target_user_id]
        similarities = []
        
        # Find similar users
        for user_id, ratings in user_ratings.items():
            if user_id != target_user_id:
                sim = self.compute_user_similarity(target_ratings, ratings)
                if sim > 0:
                    similarities.append((user_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_users = similarities[:self.k_neighbors]
        
        # Predict ratings for unrated items
        predictions = defaultdict(float)
        weights = defaultdict(float)
        
        for user_id, similarity in top_k_users:
            for item_id, rating in user_ratings[user_id].items():
                if item_id not in target_ratings:
                    predictions[item_id] += similarity * rating
                    weights[item_id] += abs(similarity)
        
        # Normalize predictions
        recommendations = []
        for item_id, weighted_sum in predictions.items():
            if weights[item_id] > 0:
                predicted_rating = weighted_sum / weights[item_id]
                item = next((i for i in item_list if i.id == item_id), None)
                if item:
                    recommendations.append((item, predicted_rating))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
    
    def item_based_cf(self, target_user_id: int, user_ratings: Dict[int, Dict], item_list: List[Any]) -> List[Tuple[Any, float]]:
        """Item-based collaborative filtering"""
        if target_user_id not in user_ratings:
            return []
        
        target_ratings = user_ratings[target_user_id]
        
        # Build item similarity matrix
        item_similarities = defaultdict(lambda: defaultdict(float))
        
        for user_id, ratings in user_ratings.items():
            rated_items = list(ratings.keys())
            for i in range(len(rated_items)):
                for j in range(i + 1, len(rated_items)):
                    item1, item2 = rated_items[i], rated_items[j]
                    # Simple cosine similarity based on ratings
                    item_similarities[item1][item2] += ratings[item1] * ratings[item2]
                    item_similarities[item2][item1] += ratings[item1] * ratings[item2]
        
        # Predict ratings
        recommendations = []
        for item in item_list:
            if item.id not in target_ratings:
                predicted_rating = 0
                weight_sum = 0
                
                for rated_item_id, rating in target_ratings.items():
                    similarity = item_similarities[item.id].get(rated_item_id, 0)
                    if similarity > 0:
                        predicted_rating += similarity * rating
                        weight_sum += similarity
                
                if weight_sum > 0:
                    recommendations.append((item, predicted_rating / weight_sum))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations

class HybridRecommendationEngine:
    """Hybrid Recommendation Engine combining multiple strategies"""
    
    def __init__(self):
        self.content_based = ContentBasedFiltering()
        self.collaborative = CollaborativeFiltering()
        self.content_weight = 0.4
        self.collaborative_weight = 0.3
        self.popularity_weight = 0.2
        self.language_weight = 0.1
        
    def weighted_hybrid(self, user_profile: Dict, user_ratings: Dict, content_list: List[Any], limit: int = 20) -> List[Tuple[Any, float]]:
        """Weighted combination of different strategies"""
        scores = defaultdict(float)
        
        # Content-based scores
        cb_recommendations = self.content_based.get_recommendations(user_profile, content_list, limit * 2)
        for content, score in cb_recommendations:
            scores[content.id] += score * self.content_weight
        
        # Collaborative filtering scores (if available)
        if user_ratings and user_profile.get('user_id'):
            cf_recommendations = self.collaborative.user_based_cf(
                user_profile['user_id'],
                user_ratings,
                content_list
            )
            for content, score in cf_recommendations[:limit * 2]:
                scores[content.id] += score * self.collaborative_weight
        
        # Popularity scores
        popularity_scores = PopularityRanking.calculate_popularity_scores(content_list)
        for content, score in popularity_scores:
            scores[content.id] += score * self.popularity_weight
        
        # Language preference scores
        language_scores = LanguagePriorityFilter.apply_language_scores(content_list, user_profile.get('preferred_languages', []))
        for content, score in language_scores:
            scores[content.id] += score * self.language_weight
        
        # Combine and sort
        final_recommendations = []
        for content in content_list:
            if content.id in scores:
                final_recommendations.append((content, scores[content.id]))
        
        final_recommendations.sort(key=lambda x: x[1], reverse=True)
        return final_recommendations[:limit]
    
    def switching_hybrid(self, context: Dict, user_profile: Dict, content_list: List[Any], limit: int = 20) -> List[Tuple[Any, float]]:
        """Switch between strategies based on context"""
        # New user - use popularity-based
        if not user_profile.get('interaction_count') or user_profile['interaction_count'] < 5:
            return PopularityRanking.calculate_popularity_scores(content_list)[:limit]
        
        # Experienced user with enough data - use collaborative
        if user_profile.get('interaction_count', 0) > 50:
            return self.collaborative.user_based_cf(
                user_profile['user_id'],
                context.get('user_ratings', {}),
                content_list
            )[:limit]
        
        # Default to content-based
        return self.content_based.get_recommendations(user_profile, content_list, limit)

class PopularityRanking:
    """Trending & Popularity Ranking algorithms"""
    
    @staticmethod
    def calculate_popularity_score(content: Any, time_decay: bool = True) -> float:
        """Calculate popularity score with time decay"""
        # Base popularity formula
        tmdb_popularity = content.popularity or 0
        vote_average = content.rating or 0
        vote_count = content.vote_count or 0
        
        # Normalize vote count (assuming max vote count is 10000)
        normalized_vote_count = min(vote_count / 10000, 1.0)
        
        # Weighted popularity score
        score = (tmdb_popularity * 0.3) + (vote_average / 10 * 0.4) + (normalized_vote_count * 0.3)
        
        # Apply time decay if enabled
        if time_decay and content.release_date:
            days_old = (datetime.now().date() - content.release_date).days
            decay_factor = math.exp(-days_old / 365)  # Exponential decay over a year
            score *= decay_factor
        
        return score
    
    @staticmethod
    def calculate_popularity_scores(content_list: List[Any]) -> List[Tuple[Any, float]]:
        """Calculate popularity scores for a list of content"""
        scores = []
        for content in content_list:
            score = PopularityRanking.calculate_popularity_score(content)
            scores.append((content, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    @staticmethod
    def wilson_score_confidence(positive: int, total: int, confidence: float = 0.95) -> float:
        """Wilson score confidence interval for ranking"""
        if total == 0:
            return 0
        
        z = 1.96  # 95% confidence
        phat = positive / total
        
        return (phat + z*z/(2*total) - z * math.sqrt((phat*(1-phat)+z*z/(4*total))/total))/(1+z*z/total)
    
    @staticmethod
    def bayesian_average(rating: float, count: int, global_mean: float = 7.0, min_votes: int = 10) -> float:
        """Bayesian average for robust rating calculation"""
        return (count * rating + min_votes * global_mean) / (count + min_votes)
    
    @staticmethod
    def calculate_trending_score(content: Any, velocity_window: int = 7) -> float:
        """Calculate trending score based on velocity and momentum"""
        # Base popularity
        base_score = PopularityRanking.calculate_popularity_score(content, time_decay=False)
        
        # Velocity boost for new content
        if content.release_date:
            days_old = (datetime.now().date() - content.release_date).days
            if days_old <= velocity_window:
                velocity_boost = 2.0 - (days_old / velocity_window)  # Higher boost for newer content
                base_score *= velocity_boost
        
        # Momentum tracking (simplified - would need historical data in production)
        if content.is_trending:
            base_score *= 1.5  # Boost for already trending content
        
        return base_score

class LanguagePriorityFilter:
    """Language Priority Filtering System"""
    
    @staticmethod
    def get_language_score(languages: List[str], preferred_languages: List[str] = None) -> float:
        """Calculate language priority score"""
        if not languages:
            return 0.0
        
        max_score = 0.0
        
        # Check for Telugu priority (main priority)
        telugu_variants = ['telugu', 'te']
        for lang in languages:
            lang_lower = lang.lower() if isinstance(lang, str) else ''
            if any(variant in lang_lower for variant in telugu_variants):
                max_score = max(max_score, LANGUAGE_WEIGHTS.get('telugu', 1.0))
        
        # Check for English/Hindi (secondary priority)
        secondary_langs = ['english', 'en', 'hindi', 'hi']
        for lang in languages:
            lang_lower = lang.lower() if isinstance(lang, str) else ''
            if any(sec_lang in lang_lower for sec_lang in secondary_langs):
                max_score = max(max_score, LANGUAGE_WEIGHTS.get(lang_lower, 0.8))
        
        # Check for tertiary languages
        tertiary_langs = ['malayalam', 'ml', 'kannada', 'kn', 'tamil', 'ta']
        for lang in languages:
            lang_lower = lang.lower() if isinstance(lang, str) else ''
            if any(ter_lang in lang_lower for ter_lang in tertiary_langs):
                max_score = max(max_score, LANGUAGE_WEIGHTS.get(lang_lower, 0.6))
        
        # Check for dual-language content with Telugu
        has_telugu = any(variant in ' '.join(languages).lower() for variant in telugu_variants)
        if has_telugu and len(languages) > 1:
            max_score *= 1.2  # 20% boost for dual-language content with Telugu
        
        # User preference bonus
        if preferred_languages:
            for pref_lang in preferred_languages:
                if any(pref_lang.lower() in lang.lower() for lang in languages):
                    max_score *= 1.1  # 10% boost for user preference match
        
        return min(max_score, 1.0)  # Cap at 1.0
    
    @staticmethod
    def apply_language_scores(content_list: List[Any], preferred_languages: List[str] = None) -> List[Tuple[Any, float]]:
        """Apply language scores to content list"""
        scores = []
        for content in content_list:
            languages = json.loads(content.languages or '[]')
            score = LanguagePriorityFilter.get_language_score(languages, preferred_languages)
            scores.append((content, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    @staticmethod
    def filter_by_language_priority(content_list: List[Any], main_language: str = 'telugu') -> List[Any]:
        """Filter and sort content by language priority"""
        priority_groups = {
            'telugu': [],
            'english_hindi': [],
            'regional': [],
            'others': []
        }
        
        for content in content_list:
            languages = json.loads(content.languages or '[]')
            languages_lower = [lang.lower() for lang in languages if isinstance(lang, str)]
            
            # Categorize content
            if any('telugu' in lang or 'te' == lang for lang in languages_lower):
                priority_groups['telugu'].append(content)
            elif any(lang in ['english', 'en', 'hindi', 'hi'] for lang in languages_lower):
                priority_groups['english_hindi'].append(content)
            elif any(lang in ['malayalam', 'ml', 'kannada', 'kn', 'tamil', 'ta'] for lang in languages_lower):
                priority_groups['regional'].append(content)
            else:
                priority_groups['others'].append(content)
        
        # Combine in priority order
        sorted_content = []
        sorted_content.extend(priority_groups['telugu'])
        sorted_content.extend(priority_groups['english_hindi'])
        sorted_content.extend(priority_groups['regional'])
        sorted_content.extend(priority_groups['others'])
        
        return sorted_content

class AdvancedAlgorithms:
    """Advanced algorithmic strategies including diversity and contextual recommendations"""
    
    @staticmethod
    def inject_diversity(recommendations: List[Tuple[Any, float]], diversity_weight: float = 0.3) -> List[Tuple[Any, float]]:
        """Inject diversity into recommendations to avoid filter bubbles"""
        if len(recommendations) < 10:
            return recommendations
        
        diverse_recs = []
        seen_genres = set()
        seen_languages = set()
        
        for content, score in recommendations:
            genres = set(json.loads(content.genres or '[]'))
            languages = set(json.loads(content.languages or '[]'))
            
            # Calculate diversity bonus
            genre_novelty = len(genres - seen_genres) / max(len(genres), 1)
            language_novelty = len(languages - seen_languages) / max(len(languages), 1)
            
            diversity_bonus = (genre_novelty + language_novelty) / 2 * diversity_weight
            adjusted_score = score * (1 + diversity_bonus)
            
            diverse_recs.append((content, adjusted_score))
            seen_genres.update(genres)
            seen_languages.update(languages)
        
        diverse_recs.sort(key=lambda x: x[1], reverse=True)
        return diverse_recs
    
    @staticmethod
    def calculate_serendipity_score(content: Any, user_profile: Dict) -> float:
        """Calculate serendipity score for unexpected but relevant recommendations"""
        score = 0.0
        
        # Check if content is outside user's usual preferences
        user_genres = set(user_profile.get('preferred_genres', []))
        content_genres = set(json.loads(content.genres or '[]'))
        
        # Higher score for content outside usual genres but with good rating
        if content_genres and not content_genres.intersection(user_genres):
            if content.rating and content.rating > 7.5:
                score += 0.5
        
        # Bonus for different language than usual
        user_languages = set(user_profile.get('preferred_languages', []))
        content_languages = set(json.loads(content.languages or '[]'))
        
        if content_languages and not content_languages.intersection(user_languages):
            score += 0.3
        
        # Adjust by content quality
        if content.rating:
            score *= (content.rating / 10)
        
        return score
    
    @staticmethod
    def contextual_filtering(content_list: List[Any], context: Dict) -> List[Tuple[Any, float]]:
        """Apply contextual filtering based on time, device, mood, etc."""
        scores = []
        
        for content in content_list:
            score = 1.0
            
            # Time-based filtering
            current_hour = datetime.now().hour
            if context.get('time_aware'):
                if 6 <= current_hour < 12:  # Morning
                    if content.runtime and content.runtime < 90:  # Prefer shorter content
                        score *= 1.2
                elif 18 <= current_hour < 22:  # Evening/Prime time
                    if content.content_type == 'movie':  # Prefer movies
                        score *= 1.3
                elif 22 <= current_hour or current_hour < 2:  # Late night
                    if content.content_type == 'tv':  # Prefer series
                        score *= 1.2
            
            # Device-based filtering
            if context.get('device') == 'mobile':
                if content.runtime and content.runtime < 60:  # Prefer shorter content on mobile
                    score *= 1.1
            elif context.get('device') == 'tv':
                if content.content_type == 'movie':  # Prefer movies on TV
                    score *= 1.2
            
            # Mood-based filtering
            mood = context.get('mood')
            if mood:
                genres = json.loads(content.genres or '[]')
                if mood == 'happy' and 'Comedy' in genres:
                    score *= 1.5
                elif mood == 'romantic' and 'Romance' in genres:
                    score *= 1.5
                elif mood == 'thrilling' and any(g in genres for g in ['Thriller', 'Action']):
                    score *= 1.5
                elif mood == 'relaxed' and any(g in genres for g in ['Drama', 'Documentary']):
                    score *= 1.3
            
            scores.append((content, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

class RecommendationOrchestrator:
    """Main orchestrator that combines all algorithms and strategies"""
    
    def __init__(self):
        self.content_based = ContentBasedFiltering()
        self.collaborative = CollaborativeFiltering()
        self.hybrid = HybridRecommendationEngine()
        
    def get_trending_with_algorithms(self, content_list: List[Any], limit: int = 20, 
                                    region: str = None, apply_language_priority: bool = True) -> Dict[str, List[Dict]]:
        """Get trending content with multi-level ranking"""
        categories = {
            'trending_movies': [],
            'trending_tv_shows': [],
            'trending_anime': [],
            'popular_nearby': [],
            'top_10_today': []
        }
        
        # Separate content by type
        movies = [c for c in content_list if c.content_type == 'movie']
        tv_shows = [c for c in content_list if c.content_type == 'tv']
        anime = [c for c in content_list if c.content_type == 'anime']
        
        # Calculate trending scores
        for content_type, content_subset in [('movie', movies), ('tv', tv_shows), ('anime', anime)]:
            trending_scores = []
            for content in content_subset:
                # Multi-level scoring
                popularity_score = PopularityRanking.calculate_trending_score(content)
                language_score = LanguagePriorityFilter.get_language_score(
                    json.loads(content.languages or '[]')
                )
                
                # Weighted final score
                final_score = (popularity_score * 0.7) + (language_score * 0.3)
                trending_scores.append((content, final_score))
            
            # Sort and apply language priority if enabled
            trending_scores.sort(key=lambda x: x[1], reverse=True)
            
            if apply_language_priority:
                # Reorder by language groups while maintaining score order within groups
                trending_content = [item[0] for item in trending_scores]
                trending_content = LanguagePriorityFilter.filter_by_language_priority(trending_content)
                trending_scores = [(c, s) for c in trending_content for _, s in trending_scores if _[0].id == c.id]
            
            # Assign to categories
            if content_type == 'movie':
                categories['trending_movies'] = self._format_recommendations(trending_scores[:limit])
            elif content_type == 'tv':
                categories['trending_tv_shows'] = self._format_recommendations(trending_scores[:limit])
            elif content_type == 'anime':
                categories['trending_anime'] = self._format_recommendations(trending_scores[:limit])
        
        # Top 10 Today - Combined all types with highest scores
        all_trending = []
        for content in content_list:
            score = PopularityRanking.calculate_trending_score(content)
            language_score = LanguagePriorityFilter.get_language_score(
                json.loads(content.languages or '[]')
            )
            final_score = (score * 0.7) + (language_score * 0.3)
            all_trending.append((content, final_score))
        
        all_trending.sort(key=lambda x: x[1], reverse=True)
        categories['top_10_today'] = self._format_recommendations(all_trending[:10])
        
        # Popular Nearby (region-specific with language priority)
        if region:
            regional_content = [c for c in content_list if self._is_regional_content(c, region)]
            regional_content = LanguagePriorityFilter.filter_by_language_priority(regional_content)
            regional_scores = [(c, PopularityRanking.calculate_popularity_score(c)) for c in regional_content]
            categories['popular_nearby'] = self._format_recommendations(regional_scores[:limit])
        
        return categories
    
    def get_new_releases_with_algorithms(self, content_list: List[Any], limit: int = 20) -> List[Dict]:
        """Get new releases with Telugu priority and algorithmic ranking"""
        # Filter for new releases (last 60 days)
        new_releases = []
        for content in content_list:
            if content.release_date:
                days_since_release = (datetime.now().date() - content.release_date).days
                if days_since_release <= 60:
                    new_releases.append(content)
        
        # Apply multi-level ranking
        scored_releases = []
        for content in new_releases:
            # Calculate multiple scores
            freshness_score = 1.0 - ((datetime.now().date() - content.release_date).days / 60)
            popularity_score = PopularityRanking.calculate_popularity_score(content, time_decay=False)
            language_score = LanguagePriorityFilter.get_language_score(
                json.loads(content.languages or '[]')
            )
            quality_score = PopularityRanking.bayesian_average(
                content.rating or 0,
                content.vote_count or 0
            ) / 10
            
            # Weighted combination with Telugu priority
            languages = json.loads(content.languages or '[]')
            is_telugu = any('telugu' in lang.lower() or lang.lower() == 'te' for lang in languages)
            
            if is_telugu:
                # Telugu content gets highest weight
                final_score = (freshness_score * 0.2) + (popularity_score * 0.2) + \
                             (language_score * 0.4) + (quality_score * 0.2)
            else:
                # Other content with standard weights
                final_score = (freshness_score * 0.3) + (popularity_score * 0.3) + \
                             (language_score * 0.2) + (quality_score * 0.2)
            
            scored_releases.append((content, final_score))
        
        # Sort by score
        scored_releases.sort(key=lambda x: x[1], reverse=True)
        
        # Apply language priority ordering
        releases_content = [item[0] for item in scored_releases]
        prioritized_releases = LanguagePriorityFilter.filter_by_language_priority(
            releases_content,
            main_language='telugu'
        )
        
        # Maintain scores after reordering
        final_releases = []
        for content in prioritized_releases[:limit]:
            score = next((s for c, s in scored_releases if c.id == content.id), 0)
            final_releases.append((content, score))
        
        return self._format_recommendations(final_releases)
    
    def get_personalized_recommendations(self, user_profile: Dict, content_list: List[Any], 
                                        context: Dict = None, limit: int = 20) -> List[Dict]:
        """Get personalized recommendations using hybrid approach"""
        # Use hybrid engine
        recommendations = self.hybrid.weighted_hybrid(
            user_profile,
            context.get('user_ratings', {}) if context else {},
            content_list,
            limit * 2  # Get more for diversity injection
        )
        
        # Apply diversity
        recommendations = AdvancedAlgorithms.inject_diversity(recommendations)
        
        # Apply contextual filtering if context provided
        if context:
            context_scores = AdvancedAlgorithms.contextual_filtering(
                [r[0] for r in recommendations],
                context
            )
            # Merge scores
            final_scores = {}
            for content, base_score in recommendations:
                context_score = next((s for c, s in context_scores if c.id == content.id), 1.0)
                final_scores[content.id] = base_score * context_score
            
            recommendations = [(c, final_scores[c.id]) for c, _ in recommendations if c.id in final_scores]
            recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return self._format_recommendations(recommendations[:limit])
    
    def _format_recommendations(self, recommendations: List[Tuple[Any, float]]) -> List[Dict]:
        """Format recommendations for API response"""
        formatted = []
        for content, score in recommendations:
            formatted.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'languages': json.loads(content.languages or '[]'),
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'algorithm_score': round(score, 3),
                'youtube_trailer_id': content.youtube_trailer_id
            })
        return formatted
    
    def _is_regional_content(self, content: Any, region: str) -> bool:
        """Check if content is relevant to a region"""
        # Simplified regional check - can be enhanced with more logic
        languages = json.loads(content.languages or '[]')
        
        # Map regions to languages
        region_languages = {
            'IN': ['hindi', 'telugu', 'tamil', 'malayalam', 'kannada'],
            'US': ['english'],
            'JP': ['japanese'],
            'KR': ['korean']
        }
        
        if region in region_languages:
            expected_langs = region_languages[region]
            for lang in languages:
                if any(expected in lang.lower() for expected in expected_langs):
                    return True
        
        return False

# Evaluation Metrics
class EvaluationMetrics:
    """Metrics for evaluating recommendation quality"""
    
    @staticmethod
    def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
        """Calculate Precision@K"""
        if k == 0:
            return 0.0
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        return len([item for item in recommended_k if item in relevant_set]) / k
    
    @staticmethod
    def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
        """Calculate Recall@K"""
        if len(relevant) == 0:
            return 0.0
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        return len([item for item in recommended_k if item in relevant_set]) / len(relevant)
    
    @staticmethod
    def ndcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
        """Calculate NDCG@K (Normalized Discounted Cumulative Gain)"""
        def dcg_at_k(r, k):
            r = np.asfarray(r)[:k]
            if r.size:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            return 0.
        
        relevance = [1 if item in relevant else 0 for item in recommended[:k]]
        ideal_relevance = sorted(relevance, reverse=True)
        
        if not ideal_relevance:
            return 0.
        
        dcg = dcg_at_k(relevance, k)
        idcg = dcg_at_k(ideal_relevance, k)
        
        return dcg / idcg if idcg > 0 else 0.
    
    @staticmethod
    def diversity_score(recommendations: List[Any]) -> float:
        """Calculate diversity score based on genre variety"""
        if not recommendations:
            return 0.0
        
        all_genres = []
        for content in recommendations:
            genres = json.loads(content.genres or '[]')
            all_genres.extend(genres)
        
        if not all_genres:
            return 0.0
        
        unique_genres = len(set(all_genres))
        total_genres = len(all_genres)
        
        return unique_genres / total_genres
    
    @staticmethod
    def coverage_score(recommended_items: List[int], catalog_size: int) -> float:
        """Calculate catalog coverage"""
        if catalog_size == 0:
            return 0.0
        return len(set(recommended_items)) / catalog_size