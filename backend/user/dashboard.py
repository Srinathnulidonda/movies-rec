# user/dashboard.py
from flask import request, jsonify
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import logging
from .utils import require_auth, get_enhanced_user_stats, recommendation_engine, get_cinematic_dna_summary, profile_analyzer

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
        # Try to use new profile analyzer first
        if profile_analyzer:
            try:
                user_profile = profile_analyzer.build_comprehensive_profile(current_user.id)
                
                if user_profile:
                    # Enhanced insights from new system
                    cinematic_dna = user_profile.get('cinematic_dna', {})
                    behavior_profile = user_profile.get('behavior_profile', {})
                    
                    insights = {
                        'profile_strength': {
                            'completeness': user_profile.get('profile_confidence', 0),
                            'confidence': user_profile.get('profile_confidence', 0),
                            'status': user_profile.get('personalization_readiness', 'developing'),
                            'recommendation_strategy': user_profile.get('recommendations_strategy', 'content_based'),
                            'interactions_needed': max(0, 20 - user_profile.get('content_history_size', 0))
                        },
                        'cinematic_dna': {
                            'sophistication_score': cinematic_dna.get('cinematic_sophistication_score', 0),
                            'narrative_preferences': cinematic_dna.get('narrative_preferences', {}),
                            'style_affinities': cinematic_dna.get('style_affinities', {}),
                            'cultural_alignment': cinematic_dna.get('cultural_alignment', {}),
                            'telugu_affinity': cinematic_dna.get('telugu_cultural_affinity', 0),
                            'indian_affinity': cinematic_dna.get('indian_cultural_affinity', 0),
                            'global_exposure': cinematic_dna.get('global_cinema_exposure', 0),
                            'production_scale_preference': cinematic_dna.get('production_scale_preference', 'medium')
                        },
                        'preferences': {
                            'top_genres': list(cinematic_dna.get('genre_sophistication', {}).keys())[:5],
                            'preferred_languages': _extract_preferred_languages(user_profile),
                            'narrative_themes': list(cinematic_dna.get('narrative_preferences', {}).keys())[:3],
                            'quality_threshold': behavior_profile.get('rating_behavior', {}).get('harsh_rating_threshold', 6.0)
                        },
                        'behavior': {
                            'engagement_score': behavior_profile.get('engagement_patterns', {}).get('total_weighted_engagement', 0),
                            'viewing_style': _determine_viewing_style(behavior_profile),
                            'exploration_tendency': behavior_profile.get('content_exploration', {}).get('exploration_tendency', 'medium'),
                            'total_interactions': user_profile.get('content_history_size', 0),
                            'consistency': behavior_profile.get('temporal_behavior', {}).get('activity_consistency', 0),
                            'binge_tendency': behavior_profile.get('temporal_behavior', {}).get('binge_tendency', 0)
                        },
                        'recent_activity': behavior_profile.get('recent_activity', {}),
                        'recommendations_quality': {
                            'accuracy_estimate': min(user_profile.get('profile_confidence', 0) * 100, 95),
                            'personalization_level': user_profile.get('personalization_readiness', 'developing'),
                            'next_improvement': _get_advanced_improvement_suggestion(user_profile),
                            'strategies_available': _get_available_strategies(user_profile)
                        }
                    }
                    
                    return jsonify({
                        'success': True,
                        'insights': insights,
                        'profile_version': user_profile.get('profile_version', '3.0'),
                        'last_updated': user_profile.get('profile_created_at', datetime.utcnow().isoformat()),
                        'next_update': user_profile.get('next_update_due', (datetime.utcnow() + timedelta(hours=24)).isoformat()),
                        'advanced_features': {
                            'cinematic_dna': True,
                            'behavioral_analysis': True,
                            'preference_embeddings': True,
                            'real_time_learning': True
                        }
                    }), 200
            except Exception as e:
                logger.warning(f"Error using advanced profile analyzer: {e}")
                # Fall through to legacy system
        
        # Fallback to legacy recommendation engine
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

def _extract_preferred_languages(user_profile: Dict[str, Any]) -> List[str]:
    """Extract preferred languages from advanced profile"""
    cinematic_dna = user_profile.get('cinematic_dna', {})
    cultural_alignment = cinematic_dna.get('cultural_alignment', {})
    
    languages = []
    if cultural_alignment.get('telugu_traditional', 0) > 0.5:
        languages.append('Telugu')
    if cultural_alignment.get('indian_mainstream', 0) > 0.5:
        languages.extend(['Hindi', 'Tamil'])
    if cultural_alignment.get('global_blockbuster', 0) > 0.5:
        languages.append('English')
    
    return languages[:3] if languages else ['Telugu', 'English', 'Hindi']

def _determine_viewing_style(behavior_profile: Dict[str, Any]) -> str:
    """Determine user's viewing style from behavior"""
    binge = behavior_profile.get('temporal_behavior', {}).get('binge_tendency', 0)
    exploration = behavior_profile.get('content_exploration', {}).get('overall_exploration_score', 0)
    
    if binge > 0.7:
        return 'binge_watcher'
    elif exploration > 0.7:
        return 'explorer'
    elif binge < 0.3 and exploration < 0.3:
        return 'casual_viewer'
    else:
        return 'balanced_viewer'

def _get_advanced_improvement_suggestion(user_profile: Dict[str, Any]) -> str:
    """Get advanced improvement suggestion based on profile analysis"""
    readiness = user_profile.get('personalization_readiness', 'cold_start')
    content_count = user_profile.get('content_history_size', 0)
    confidence = user_profile.get('profile_confidence', 0)
    
    if readiness == 'cold_start':
        return 'Start by exploring content in Telugu and your preferred languages'
    elif readiness == 'low':
        return 'Rate more content to help CineBrain understand your taste better'
    elif readiness == 'medium':
        return 'Try exploring different genres to enhance your Cinematic DNA profile'
    elif readiness == 'high' and confidence < 0.9:
        return 'Your profile is well-developed! Keep discovering new content'
    else:
        return 'Your CineBrain profile is optimized for maximum personalization!'

def _get_available_strategies(user_profile: Dict[str, Any]) -> List[str]:
    """Get list of recommendation strategies available for user"""
    readiness = user_profile.get('personalization_readiness', 'cold_start')
    
    strategies = ['popularity_based', 'language_priority']
    
    if readiness in ['low', 'medium', 'high']:
        strategies.append('content_based')
    
    if readiness in ['medium', 'high']:
        strategies.append('collaborative_filtering')
        strategies.append('cinematic_dna_matching')
    
    if readiness == 'high':
        strategies.append('advanced_hybrid')
        strategies.append('preference_embedding_similarity')
    
    return strategies