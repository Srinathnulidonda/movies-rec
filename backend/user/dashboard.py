# backend/user/dashboard.py

from flask import request, jsonify
from datetime import datetime, timedelta
import json
import logging
import random
from collections import Counter, defaultdict
import numpy as np
from .utils import (
    require_auth, db, User, UserInteraction, Content, get_enhanced_user_stats, 
    recommendation_engine, profile_analyzer, cache_get, cache_set, get_cache_key, 
    get_content_by_ids, calculate_watch_time, get_personalized_recommendations,
    get_user_profile_insights
)

logger = logging.getLogger(__name__)

@require_auth
def get_user_analytics(current_user):
    try:
        analytics = get_enhanced_user_stats(current_user.id)
        
        insights = {
            'recommendations': {
                'total_generated': analytics.get('total_interactions', 0),
                'accuracy_score': analytics.get('engagement_metrics', {}).get('engagement_score', 0),
                'improvement_tips': []
            },
            'content_preferences': {
                'diversity_level': 'high' if analytics.get('content_diversity', {}).get('diversity_score', 0) > 0.7 else 'medium',
                'exploration_tendency': analytics.get('discovery_score', 0),
                'quality_preference': analytics.get('quality_preferences', {}).get('quality_preference', 'balanced')
            },
            'engagement_level': 'high' if analytics.get('engagement_metrics', {}).get('engagement_score', 0) > 0.7 else 'moderate'
        }
        
        if analytics.get('total_interactions', 0) < 10:
            insights['recommendations']['improvement_tips'].append(
                "Interact with more CineBrain content (like, favorite, rate) to improve recommendations"
            )
        
        if analytics.get('ratings_given', 0) < 5:
            insights['recommendations']['improvement_tips'].append(
                "Rate CineBrain content to help our AI understand your preferences better"
            )
        
        if analytics.get('content_diversity', {}).get('genre_diversity_count', 0) < 5:
            insights['recommendations']['improvement_tips'].append(
                "Explore different genres on CineBrain to discover new content you might love"
            )
        
        return jsonify({
            'success': True,
            'analytics': analytics,
            'insights': insights,
            'generated_at': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain analytics error: {e}")
        return jsonify({'error': 'Failed to get CineBrain analytics'}), 500

@require_auth
def get_profile_insights(current_user):
    try:
        if not recommendation_engine:
            return jsonify({'error': 'CineBrain recommendation engine not available'}), 503
        
        user_profile = get_user_profile_insights(current_user.id)
        
        if not user_profile:
            return jsonify({
                'success': False,
                'message': 'Could not build CineBrain user profile - insufficient interaction data',
                'suggestion': 'Interact with more CineBrain content to build your profile'
            }), 404
        
        insights = {
            'profile_strength': {
                'completeness': user_profile.get('profile_completeness', 0),
                'confidence': user_profile.get('confidence_score', 0),
                'status': 'strong' if user_profile.get('confidence_score', 0) > 0.7 else 'developing',
                'interactions_needed': max(0, 20 - user_profile.get('implicit_preferences', {}).get('total_interactions', 0))
            },
            'preferences': {
                'top_genres': user_profile.get('implicit_preferences', {}).get('genre_preferences', {}).get('top_genres', [])[:5],
                'preferred_languages': user_profile.get('explicit_preferences', {}).get('preferred_languages', [])[:3],
                'content_types': user_profile.get('implicit_preferences', {}).get('content_type_preferences', {}).get('content_type_scores', {}),
                'quality_threshold': user_profile.get('recommendation_context', {}).get('quality_preference', 6.0)
            },
            'behavior': {
                'engagement_score': user_profile.get('engagement_metrics', {}).get('engagement_score', 0),
                'viewing_style': user_profile.get('behavioral_patterns', {}).get('most_common_interaction', 'explorer'),
                'exploration_tendency': user_profile.get('engagement_metrics', {}).get('discovery_rate', 0),
                'total_interactions': user_profile.get('implicit_preferences', {}).get('total_interactions', 0),
                'consistency': user_profile.get('behavioral_patterns', {}).get('activity_consistency', 0)
            },
            'recent_activity': user_profile.get('behavioral_patterns', {}),
            'recommendations_quality': {
                'accuracy_estimate': min(user_profile.get('confidence_score', 0) * 100, 95),
                'personalization_level': 'high' if user_profile.get('confidence_score', 0) > 0.8 else 'moderate',
                'next_improvement': _get_improvement_suggestion(user_profile)
            }
        }
        
        return jsonify({
            'success': True,
            'insights': insights,
            'last_updated': datetime.utcnow().isoformat(),
            'profile_version': '3.0'
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain Profile insights error: {e}")
        return jsonify({'error': 'Failed to get CineBrain profile insights'}), 500

def _get_improvement_suggestion(user_profile):
    completeness = user_profile.get('profile_completeness', 0)
    total_interactions = user_profile.get('implicit_preferences', {}).get('total_interactions', 0)
    ratings_count = user_profile.get('implicit_preferences', {}).get('rating_patterns', {}).get('total_ratings', 0)
    
    if completeness < 0.3:
        return 'Interact with more CineBrain content (like, favorite, add to watchlist) to improve accuracy'
    elif ratings_count < 5:
        return 'Rate more CineBrain content to help our AI understand your taste better'
    elif completeness < 0.8:
        return 'Explore different genres on CineBrain to get more diverse recommendations'
    else:
        return 'Your CineBrain recommendations are highly accurate! Keep discovering new content'

@require_auth
def get_dashboard(current_user):
    try:
        cache_key = get_cache_key('user_dashboard', current_user.id)
        cached_dashboard = cache_get(cache_key)
        
        if cached_dashboard:
            return jsonify(cached_dashboard), 200
        
        dashboard_data = {
            'user': current_user.username,
            'dashboard': {}
        }
        
        dashboard_data['dashboard']['top_picks'] = get_top_picks_for_you(current_user.id)
        dashboard_data['dashboard']['watchlist'] = get_your_watchlist(current_user.id)
        dashboard_data['dashboard']['favorites'] = get_favorites_collection(current_user.id)
        dashboard_data['dashboard']['personalized_insights'] = get_personalized_insights_section(current_user.id)
        dashboard_data['dashboard']['discover_new'] = get_discover_something_new(current_user.id)
        dashboard_data['dashboard']['daily_mix'] = get_daily_personalized_mix(current_user.id)
        dashboard_data['dashboard']['taste_graph'] = get_evolving_taste_graph(current_user.id)
        dashboard_data['dashboard']['trending_now'] = get_trending_now(current_user.id)
        dashboard_data['dashboard']['mood_based'] = get_mood_based_recommendations(current_user.id)
        dashboard_data['dashboard']['weekly_recap'] = get_weekly_recap(current_user.id)
        dashboard_data['dashboard']['cinestats'] = get_your_cinestats(current_user.id)
        dashboard_data['dashboard']['achievements'] = get_achievements_badges(current_user.id)
        dashboard_data['dashboard']['ai_summary'] = get_ai_summary_card(current_user.id)
        
        dashboard_data['metadata'] = {
            'generated_at': datetime.utcnow().isoformat(),
            'cache_duration': 300,
            'personalization_level': 'high' if recommendation_engine else 'basic'
        }
        
        cache_set(cache_key, dashboard_data, timeout=300)
        
        return jsonify(dashboard_data), 200
        
    except Exception as e:
        logger.error(f"Dashboard error for user {current_user.id}: {e}")
        return jsonify({'error': 'Failed to load dashboard'}), 500

def get_top_picks_for_you(user_id):
    try:
        if recommendation_engine:
            recommendations = get_personalized_recommendations(user_id, limit=8, categories=['for_you'])
            return recommendations.get('recommendations', [])[:8]
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting top picks: {e}")
        return []

def get_your_watchlist(user_id):
    try:
        watchlist_interactions = UserInteraction.query.filter_by(
            user_id=user_id,
            interaction_type='watchlist'
        ).order_by(UserInteraction.timestamp.desc()).limit(8).all()
        
        content_ids = [interaction.content_id for interaction in watchlist_interactions]
        contents = get_content_by_ids(content_ids)
        
        result = []
        for content in contents:
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'rating': content.rating
            })
        
        return result
    except Exception as e:
        logger.error(f"Error getting watchlist: {e}")
        return []

def get_favorites_collection(user_id):
    try:
        favorite_interactions = UserInteraction.query.filter_by(
            user_id=user_id,
            interaction_type='favorite'
        ).order_by(UserInteraction.timestamp.desc()).limit(8).all()
        
        content_ids = [interaction.content_id for interaction in favorite_interactions]
        contents = get_content_by_ids(content_ids)
        
        result = []
        for content in contents:
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'rating': content.rating
            })
        
        return result
    except Exception as e:
        logger.error(f"Error getting favorites: {e}")
        return []

def get_personalized_insights_section(user_id):
    try:
        if not profile_analyzer:
            return {'message': 'Profile analyzer not available'}
        
        profile = profile_analyzer.build_user_profile(user_id)
        
        return {
            'user_segment': profile.get('user_segment', 'new_user'),
            'profile_confidence': round(profile.get('confidence_score', 0) * 100),
            'top_genres': profile.get('implicit_preferences', {}).get('genre_preferences', {}).get('top_genres', [])[:3],
            'engagement_level': profile.get('engagement_metrics', {}).get('engagement_score', 0),
            'recommendation_accuracy': round(profile.get('confidence_score', 0) * 100),
            'cinematic_dna': profile.get('cinematic_dna', {}).get('dominant_theme', 'explorer')
        }
    except Exception as e:
        logger.error(f"Error getting personalized insights: {e}")
        return {'message': 'Insights unavailable'}

def get_discover_something_new(user_id):
    try:
        if recommendation_engine:
            recommendations = get_personalized_recommendations(user_id, limit=6, categories=['discover'])
            return recommendations.get('recommendations', [])[:6]
        else:
            trending_content = Content.query.filter_by(is_trending=True).limit(6).all()
            return [{
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'rating': content.rating,
                'reason': 'Trending content'
            } for content in trending_content]
    except Exception as e:
        logger.error(f"Error getting discover new: {e}")
        return []

def get_daily_personalized_mix(user_id):
    try:
        if recommendation_engine:
            recommendations = get_personalized_recommendations(user_id, limit=10, categories=['daily_mix'])
            return recommendations.get('recommendations', [])[:10]
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting daily mix: {e}")
        return []

def get_evolving_taste_graph(user_id):
    try:
        if not profile_analyzer:
            return {'message': 'Taste graph unavailable'}
        
        profile = profile_analyzer.build_user_profile(user_id)
        
        temporal_patterns = profile.get('temporal_patterns', {})
        genre_preferences = profile.get('implicit_preferences', {}).get('genre_preferences', {}).get('counts', {})
        
        return {
            'taste_evolution': temporal_patterns.get('monthly_trend', 'stable'),
            'genre_distribution': dict(list(genre_preferences.items())[:5]),
            'discovery_rate': profile.get('engagement_metrics', {}).get('discovery_rate', 0),
            'consistency_score': temporal_patterns.get('activity_stability', 0),
            'latest_preferences': profile.get('implicit_preferences', {}).get('genre_preferences', {}).get('top_genres', [])[:3]
        }
    except Exception as e:
        logger.error(f"Error getting taste graph: {e}")
        return {'message': 'Taste graph unavailable'}

def get_trending_now(user_id):
    try:
        global_trending = Content.query.filter_by(is_trending=True).order_by(Content.popularity.desc()).limit(8).all()
        
        user_languages = []
        if profile_analyzer:
            try:
                profile = profile_analyzer.build_user_profile(user_id)
                user_languages = profile.get('explicit_preferences', {}).get('preferred_languages', [])
            except:
                pass
        
        trending_data = []
        for content in global_trending:
            is_local = False
            if content.languages and user_languages:
                content_languages = json.loads(content.languages or '[]')
                is_local = any(lang.lower() in [ul.lower() for ul in user_languages] for lang in content_languages)
            
            trending_data.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'rating': content.rating,
                'popularity': content.popularity,
                'is_local_trending': is_local
            })
        
        return {
            'global': trending_data,
            'local': [item for item in trending_data if item['is_local_trending']][:4]
        }
    except Exception as e:
        logger.error(f"Error getting trending now: {e}")
        return {'global': [], 'local': []}

def get_mood_based_recommendations(user_id):
    try:
        current_hour = datetime.utcnow().hour
        current_day = datetime.utcnow().strftime('%A')
        
        mood_context = determine_mood_context(current_hour, current_day)
        
        if recommendation_engine:
            recommendations = recommendation_engine.generate_recommendations(
                user_id=user_id,
                limit=6,
                context={'mood': mood_context, 'time_of_day': current_hour}
            )
            return {
                'recommendations': recommendations.get('recommendations', [])[:6],
                'detected_mood': mood_context,
                'time_context': f"{current_hour}:00 on {current_day}"
            }
        else:
            return {'detected_mood': mood_context, 'recommendations': []}
    except Exception as e:
        logger.error(f"Error getting mood-based recommendations: {e}")
        return {'detected_mood': 'neutral', 'recommendations': []}

def determine_mood_context(hour, day):
    if hour < 6:
        return 'late_night'
    elif 6 <= hour < 9:
        return 'morning_energy'
    elif 9 <= hour < 12:
        return 'productive'
    elif 12 <= hour < 17:
        return 'afternoon_leisure'
    elif 17 <= hour < 20:
        return 'evening_wind_down'
    elif 20 <= hour < 23:
        return 'prime_time'
    else:
        return 'night_relaxation'

def get_weekly_recap(user_id):
    try:
        week_start = datetime.utcnow() - timedelta(days=7)
        
        interactions = UserInteraction.query.filter(
            UserInteraction.user_id == user_id,
            UserInteraction.timestamp >= week_start
        ).all()
        
        if not interactions:
            return {
                'message': 'No activity this week',
                'suggestion': 'Start exploring CineBrain content!'
            }
        
        interaction_counts = Counter(i.interaction_type for i in interactions)
        content_ids = [i.content_id for i in interactions]
        unique_content = len(set(content_ids))
        
        contents = get_content_by_ids(content_ids)
        genres_explored = []
        total_runtime = 0
        
        for content in contents:
            if content.genres:
                genres_explored.extend(json.loads(content.genres or '[]'))
            if content.runtime and any(i.interaction_type in ['view', 'favorite'] for i in interactions if i.content_id == content.id):
                total_runtime += content.runtime
        
        top_genres = [genre for genre, _ in Counter(genres_explored).most_common(3)]
        
        return {
            'period': f"{week_start.strftime('%B %d')} - {datetime.utcnow().strftime('%B %d, %Y')}",
            'total_interactions': len(interactions),
            'content_discovered': unique_content,
            'breakdown': dict(interaction_counts),
            'estimated_watch_time_hours': round(total_runtime / 60, 1),
            'top_genres_explored': top_genres,
            'most_active_day': max(Counter(i.timestamp.strftime('%A') for i in interactions).items(), key=lambda x: x[1])[0] if interactions else None
        }
    except Exception as e:
        logger.error(f"Error getting weekly recap: {e}")
        return {'message': 'Weekly recap unavailable'}

def get_your_cinestats(user_id):
    try:
        stats = get_enhanced_user_stats(user_id)
        watch_time = calculate_watch_time(user_id)
        
        all_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
        content_ids = [i.content_id for i in all_interactions]
        contents = get_content_by_ids(content_ids)
        
        genres_count = defaultdict(int)
        languages_count = defaultdict(int)
        content_types_count = defaultdict(int)
        
        for content in contents:
            if content.genres:
                for genre in json.loads(content.genres or '[]'):
                    genres_count[genre] += 1
            
            if content.languages:
                for language in json.loads(content.languages or '[]'):
                    languages_count[language] += 1
            
            content_types_count[content.content_type] += 1
        
        return {
            'total_watch_time': watch_time,
            'content_stats': {
                'total_content': len(contents),
                'movies': content_types_count.get('movie', 0),
                'tv_shows': content_types_count.get('tv', 0),
                'anime': content_types_count.get('anime', 0)
            },
            'diversity_metrics': {
                'genres_explored': len(genres_count),
                'languages_watched': len(languages_count),
                'diversity_score': stats.get('content_diversity', 0)
            },
            'engagement_stats': {
                'total_interactions': stats.get('total_interactions', 0),
                'favorites': stats.get('favorites', 0),
                'ratings_given': stats.get('ratings_given', 0),
                'watchlist_items': stats.get('watchlist_items', 0)
            },
            'top_preferences': {
                'favorite_genres': dict(sorted(genres_count.items(), key=lambda x: x[1], reverse=True)[:5]),
                'preferred_languages': dict(sorted(languages_count.items(), key=lambda x: x[1], reverse=True)[:3])
            }
        }
    except Exception as e:
        logger.error(f"Error getting CineStats: {e}")
        return {'message': 'CineStats unavailable'}

def get_achievements_badges(user_id):
    try:
        stats = get_enhanced_user_stats(user_id)
        achievements = []
        
        total_interactions = stats.get('total_interactions', 0)
        favorites_count = stats.get('favorites', 0)
        ratings_count = stats.get('ratings_given', 0)
        
        if total_interactions >= 10:
            achievements.append({
                'id': 'explorer',
                'name': 'Content Explorer',
                'description': 'Discovered 10+ pieces of content',
                'icon': '🔍',
                'earned_at': 'recent',
                'rarity': 'common'
            })
        
        if total_interactions >= 50:
            achievements.append({
                'id': 'enthusiast',
                'name': 'CineBrain Enthusiast',
                'description': 'Made 50+ interactions',
                'icon': '🎬',
                'earned_at': 'recent',
                'rarity': 'uncommon'
            })
        
        if favorites_count >= 20:
            achievements.append({
                'id': 'curator',
                'name': 'Content Curator',
                'description': 'Favorited 20+ items',
                'icon': '❤️',
                'earned_at': 'recent',
                'rarity': 'rare'
            })
        
        if ratings_count >= 25:
            achievements.append({
                'id': 'critic',
                'name': 'Movie Critic',
                'description': 'Rated 25+ content items',
                'icon': '⭐',
                'earned_at': 'recent',
                'rarity': 'rare'
            })
        
        user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
        content_ids = [i.content_id for i in user_interactions]
        unique_content = len(set(content_ids))
        
        if unique_content >= 100:
            achievements.append({
                'id': 'completionist',
                'name': 'Completionist',
                'description': 'Explored 100+ unique content items',
                'icon': '🏆',
                'earned_at': 'recent',
                'rarity': 'legendary'
            })
        
        return achievements
    except Exception as e:
        logger.error(f"Error getting achievements: {e}")
        return []

def get_ai_summary_card(user_id):
    try:
        week_start = datetime.utcnow() - timedelta(days=7)
        
        recent_interactions = UserInteraction.query.filter(
            UserInteraction.user_id == user_id,
            UserInteraction.timestamp >= week_start
        ).all()
        
        if not recent_interactions:
            return "Welcome to CineBrain! Start exploring content to see your personalized summary."
        
        content_ids = [i.content_id for i in recent_interactions]
        contents = get_content_by_ids(content_ids)
        
        genres_count = Counter()
        content_types_count = Counter()
        
        for content in contents:
            if content.genres:
                genres_count.update(json.loads(content.genres or '[]'))
            content_types_count[content.content_type] += 1
        
        top_genre = genres_count.most_common(1)[0][0] if genres_count else 'various genres'
        top_content_type = content_types_count.most_common(1)[0][0] if content_types_count else 'content'
        
        interaction_counts = Counter(i.interaction_type for i in recent_interactions)
        favorites_this_week = interaction_counts.get('favorite', 0)
        ratings_this_week = interaction_counts.get('rating', 0)
        
        summary_templates = [
            f"This week you explored {len(contents)} pieces of content, with a focus on {top_genre}!",
            f"You discovered {len(contents)} new {top_content_type} titles and favorited {favorites_this_week} of them.",
            f"Your week included {len(contents)} content discoveries, primarily in {top_genre}.",
            f"You've been active with {len(recent_interactions)} interactions across {len(contents)} different titles!"
        ]
        
        if favorites_this_week > 0:
            summary_templates.append(f"You found {favorites_this_week} new favorites this week!")
        
        if ratings_this_week > 0:
            summary_templates.append(f"You rated {ratings_this_week} titles this week, helping improve recommendations!")
        
        return random.choice(summary_templates)
    except Exception as e:
        logger.error(f"Error generating AI summary: {e}")
        return "Your CineBrain journey continues! Keep discovering amazing content."