# user/dashboard.py
from flask import request, jsonify
from datetime import datetime
import json
import logging
from .utils import require_auth, get_enhanced_user_stats, recommendation_engine

logger = logging.getLogger(__name__)

@require_auth
def get_user_analytics(current_user):
    """Get comprehensive user analytics"""
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
    """Get profile insights for user dashboard"""
    try:
        if not recommendation_engine:
            return jsonify({'error': 'CineBrain recommendation engine not available'}), 503
        
        user_profile = recommendation_engine.user_profiler.build_comprehensive_user_profile(current_user.id)
        
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
                'top_genres': user_profile.get('genre_preferences', {}).get('top_genres', [])[:5],
                'preferred_languages': user_profile.get('language_preferences', {}).get('preferred_languages', [])[:3],
                'content_types': user_profile.get('content_type_preferences', {}).get('content_type_scores', {}),
                'quality_threshold': user_profile.get('quality_preferences', {}).get('min_rating', 6.0)
            },
            'behavior': {
                'engagement_score': user_profile.get('engagement_score', 0),
                'viewing_style': user_profile.get('implicit_preferences', {}).get('most_common_interaction', 'explorer'),
                'exploration_tendency': user_profile.get('exploration_tendency', 0),
                'total_interactions': user_profile.get('implicit_preferences', {}).get('total_interactions', 0),
                'consistency': user_profile.get('temporal_patterns', {}).get('activity_consistency', 0)
            },
            'recent_activity': user_profile.get('recent_activity', {}),
            'recommendations_quality': {
                'accuracy_estimate': min(user_profile.get('confidence_score', 0) * 100, 95),
                'personalization_level': 'high' if user_profile.get('confidence_score', 0) > 0.8 else 'moderate',
                'next_improvement': _get_improvement_suggestion(user_profile)
            }
        }
        
        return jsonify({
            'success': True,
            'insights': insights,
            'last_updated': user_profile.get('last_updated', datetime.utcnow()).isoformat(),
            'profile_version': '3.0'
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain Profile insights error: {e}")
        return jsonify({'error': 'Failed to get CineBrain profile insights'}), 500

def _get_improvement_suggestion(user_profile):
    """Get personalized improvement suggestion"""
    completeness = user_profile.get('profile_completeness', 0)
    total_interactions = user_profile.get('implicit_preferences', {}).get('total_interactions', 0)
    ratings_count = user_profile.get('explicit_preferences', {}).get('ratings_count', 0)
    
    if completeness < 0.3:
        return 'Interact with more CineBrain content (like, favorite, add to watchlist) to improve accuracy'
    elif ratings_count < 5:
        return 'Rate more CineBrain content to help our AI understand your taste better'
    elif completeness < 0.8:
        return 'Explore different genres on CineBrain to get more diverse recommendations'
    else:
        return 'Your CineBrain recommendations are highly accurate! Keep discovering new content'