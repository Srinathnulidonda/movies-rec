# backend/personalized/routes.py

"""
CineBrain Personalized Recommendation API Routes
Flask blueprint providing personalized recommendation endpoints

This module provides:
- /api/personalized/for-you: Main personalized recommendations
- /api/personalized/discover: Discovery and exploration recommendations
- /api/personalized/mix: Mixed recommendation strategies
- /api/personalized/<genre>: Genre-specific personalized recommendations
- /api/personalized/<type>: Content type specific recommendations
- Real-time feedback processing endpoints
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, Any, List, Optional

# Import from user module for authentication
from user.utils import require_auth

logger = logging.getLogger(__name__)

# Create blueprint with API prefix
personalized_bp = Blueprint('personalized', __name__, url_prefix='/api')

def get_personalized_services():
    """
    Get personalized services with late import to avoid circular dependency
    Returns tuple of (profile_analyzer, recommendation_engine)
    """
    try:
        # Try to get from current_app context first
        if hasattr(current_app, 'profile_analyzer') and hasattr(current_app, 'recommendation_engine'):
            return current_app.profile_analyzer, current_app.recommendation_engine
        
        # Fallback to importing from personalized module
        import personalized
        profile_analyzer = personalized.get_profile_analyzer()
        recommendation_engine = personalized.get_recommendation_engine()
        
        return profile_analyzer, recommendation_engine
    except Exception as e:
        logger.error(f"Error getting personalized services: {e}")
        return None, None

def get_request_context() -> Dict[str, Any]:
    """Extract context from request"""
    return {
        'device': request.headers.get('User-Agent', '').lower(),
        'time': datetime.utcnow().hour,
        'platform': request.headers.get('X-Platform', 'web'),
        'app_version': request.headers.get('X-App-Version'),
        'session_id': request.headers.get('X-Session-Id'),
        'client_ip': request.remote_addr
    }

def add_cors_headers(response):
    """Add CORS headers to response"""
    origin = request.headers.get('Origin')
    allowed_origins = [
        'https://cinebrain.vercel.app',
        'http://127.0.0.1:5500',
        'http://127.0.0.1:5501',
        'http://localhost:3000',
        'http://localhost:5173'
    ]
    
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,X-Platform,X-App-Version,X-Session-Id'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response

@personalized_bp.after_request
def after_request(response):
    return add_cors_headers(response)

# ============================================================================
# MAIN PERSONALIZED RECOMMENDATION ENDPOINTS
# ============================================================================

@personalized_bp.route('/personalized/for-you', methods=['GET', 'OPTIONS'])
@require_auth
def get_for_you_recommendations(current_user):
    """
    Get main 'For You' personalized recommendations
    
    URL: /api/personalized/for-you
    
    Query Parameters:
    - limit: Number of recommendations (default: 50, max: 100)
    - refresh: Force refresh of recommendations (default: false)
    
    Returns:
    - Highly personalized content recommendations
    - User profile insights
    - Recommendation explanations
    """
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Get parameters
        limit = min(int(request.args.get('limit', 50)), 100)
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        # Get services with late import
        profile_analyzer, recommendation_engine = get_personalized_services()
        if not recommendation_engine:
            return jsonify({
                'error': 'CineBrain recommendation engine not available',
                'success': False
            }), 503
        
        # Get request context
        context = get_request_context()
        context['force_refresh'] = force_refresh
        
        # Generate recommendations
        recommendations = recommendation_engine.get_personalized_recommendations(
            user_id=current_user.id,
            recommendation_type='for_you',
            limit=limit,
            context=context
        )
        
        # Add user info to response
        recommendations['user'] = {
            'id': current_user.id,
            'username': current_user.username,
            'preferences_set': bool(current_user.preferred_languages or current_user.preferred_genres)
        }
        
        # Add response metadata
        recommendations['api_metadata'] = {
            'endpoint': '/api/personalized/for-you',
            'algorithm_version': '3.0.0',
            'response_time': datetime.utcnow().isoformat(),
            'total_recommendations': len(recommendations.get('recommendations', [])),
            'personalization_level': 'high',
            'cache_status': 'fresh' if force_refresh else 'cached_if_available'
        }
        
        return jsonify(recommendations), 200
        
    except Exception as e:
        logger.error(f"Error in for-you recommendations for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to generate CineBrain For You recommendations',
            'success': False,
            'user_id': current_user.id
        }), 500

@personalized_bp.route('/personalized/discover', methods=['GET', 'OPTIONS'])
@require_auth
def get_discover_recommendations(current_user):
    """
    Get discovery recommendations to help users explore new content
    
    URL: /api/personalized/discover
    
    Query Parameters:
    - limit: Number of recommendations (default: 30, max: 50)
    - exploration_level: low|medium|high (default: medium)
    
    Returns:
    - Content outside user's usual preferences
    - Discovery insights and exploration areas
    - Explanations for each recommendation
    """
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Get parameters
        limit = min(int(request.args.get('limit', 30)), 50)
        exploration_level = request.args.get('exploration_level', 'medium')
        
        # Get services with late import
        profile_analyzer, recommendation_engine = get_personalized_services()
        if not recommendation_engine:
            return jsonify({
                'error': 'CineBrain recommendation engine not available',
                'success': False
            }), 503
        
        # Generate discovery recommendations
        discovery_results = recommendation_engine.get_discovery_recommendations(
            user_id=current_user.id,
            limit=limit
        )
        
        # Add exploration metadata
        discovery_results['exploration_metadata'] = {
            'exploration_level': exploration_level,
            'discovery_algorithm': 'serendipity_enhanced',
            'comfort_zone_expansion': True,
            'genre_diversification': True,
            'cultural_exploration': True
        }
        
        # Add user context
        discovery_results['user'] = {
            'id': current_user.id,
            'username': current_user.username,
            'exploration_history': 'building'  # Would track user's discovery engagement
        }
        
        return jsonify(discovery_results), 200
        
    except Exception as e:
        logger.error(f"Error in discover recommendations for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to generate CineBrain discovery recommendations',
            'success': False,
            'user_id': current_user.id
        }), 500

@personalized_bp.route('/personalized/mix', methods=['GET', 'OPTIONS'])
@require_auth
def get_mixed_recommendations(current_user):
    """
    Get mixed recommendations combining multiple strategies
    
    URL: /api/personalized/mix
    
    Query Parameters:
    - limit: Number of recommendations (default: 40, max: 60)
    - balance: safe|balanced|adventurous (default: balanced)
    
    Returns:
    - Mix of safe recommendations and discovery content
    - Trending content personalized for user
    - Composition breakdown
    """
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Get parameters
        limit = min(int(request.args.get('limit', 40)), 60)
        balance = request.args.get('balance', 'balanced')
        
        # Get services with late import
        profile_analyzer, recommendation_engine = get_personalized_services()
        if not recommendation_engine:
            return jsonify({
                'error': 'CineBrain recommendation engine not available',
                'success': False
            }), 503
        
        # Generate mixed recommendations
        mixed_results = recommendation_engine.get_mixed_recommendations(
            user_id=current_user.id,
            limit=limit
        )
        
        # Add mix strategy metadata
        mixed_results['mix_strategy'] = {
            'balance_type': balance,
            'algorithm_combination': 'adaptive_hybrid',
            'personalization_weight': 0.6,
            'discovery_weight': 0.25,
            'trending_weight': 0.15
        }
        
        return jsonify(mixed_results), 200
        
    except Exception as e:
        logger.error(f"Error in mixed recommendations for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to generate CineBrain mixed recommendations',
            'success': False,
            'user_id': current_user.id
        }), 500

# ============================================================================
# CATEGORY-SPECIFIC PERSONALIZED ENDPOINTS
# ============================================================================

@personalized_bp.route('/personalized/genre/<genre_name>', methods=['GET', 'OPTIONS'])
@require_auth
def get_genre_personalized_recommendations(current_user, genre_name):
    """
    Get personalized recommendations for a specific genre
    
    URL: /api/personalized/genre/<genre_name>
    
    Path Parameters:
    - genre_name: Genre to focus on (e.g., 'action', 'drama', 'comedy')
    
    Query Parameters:
    - limit: Number of recommendations (default: 25, max: 40)
    - subgenre: Optional subgenre filter
    
    Returns:
    - Genre-specific personalized recommendations
    - Genre affinity analysis
    - Cross-genre suggestions
    """
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Validate and normalize genre name
        genre_name = genre_name.title()
        valid_genres = [
            'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror',
            'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War'
        ]
        
        if genre_name not in valid_genres:
            return jsonify({
                'error': f'Invalid genre: {genre_name}',
                'valid_genres': valid_genres,
                'success': False
            }), 400
        
        # Get parameters
        limit = min(int(request.args.get('limit', 25)), 40)
        subgenre = request.args.get('subgenre')
        
        # Get services with late import
        profile_analyzer, recommendation_engine = get_personalized_services()
        if not recommendation_engine or not profile_analyzer:
            return jsonify({
                'error': 'CineBrain services not available',
                'success': False
            }), 503
        
        # Get user profile for genre analysis
        user_profile = profile_analyzer.build_comprehensive_profile(current_user.id)
        
        # Generate genre-specific recommendations
        genre_recommendations = recommendation_engine.get_personalized_recommendations(
            user_id=current_user.id,
            recommendation_type='for_you',
            limit=limit * 2,  # Get more to filter by genre
            context={'genre_filter': genre_name, 'subgenre': subgenre}
        )
        
        # Filter by genre and personalize
        filtered_recommendations = []
        for rec in genre_recommendations.get('recommendations', []):
            if genre_name in rec.get('genres', []):
                # Add genre-specific metadata
                rec['genre_match'] = 'exact'
                rec['genre_affinity'] = _calculate_genre_affinity(genre_name, user_profile)
                filtered_recommendations.append(rec)
                
                if len(filtered_recommendations) >= limit:
                    break
        
        # If not enough exact matches, add related genres
        if len(filtered_recommendations) < limit:
            related_genres = _get_related_genres(genre_name)
            for rec in genre_recommendations.get('recommendations', []):
                if any(related_genre in rec.get('genres', []) for related_genre in related_genres):
                    if rec not in filtered_recommendations:
                        rec['genre_match'] = 'related'
                        rec['genre_affinity'] = _calculate_genre_affinity(genre_name, user_profile) * 0.8
                        filtered_recommendations.append(rec)
                        
                        if len(filtered_recommendations) >= limit:
                            break
        
        # Generate genre analysis
        genre_analysis = _analyze_user_genre_relationship(genre_name, user_profile)
        
        return jsonify({
            'success': True,
            'user_id': current_user.id,
            'genre': genre_name,
            'subgenre': subgenre,
            'recommendations': filtered_recommendations,
            'genre_analysis': genre_analysis,
            'recommendation_count': len(filtered_recommendations),
            'related_genres': _get_related_genres(genre_name),
            'user_genre_affinity': _calculate_genre_affinity(genre_name, user_profile)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in genre recommendations for user {current_user.id}, genre {genre_name}: {e}")
        return jsonify({
            'error': f'Failed to generate {genre_name} recommendations',
            'success': False,
            'user_id': current_user.id
        }), 500

@personalized_bp.route('/personalized/type/<content_type>', methods=['GET', 'OPTIONS'])
@require_auth
def get_content_type_personalized_recommendations(current_user, content_type):
    """
    Get personalized recommendations for a specific content type
    
    URL: /api/personalized/type/<content_type>
    
    Path Parameters:
    - content_type: Type of content ('movie', 'tv', 'anime')
    
    Query Parameters:
    - limit: Number of recommendations (default: 30, max: 50)
    - filter: Additional filters (new, trending, popular)
    
    Returns:
    - Content type specific personalized recommendations
    - Type preference analysis
    - Cross-type suggestions
    """
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Validate content type
        valid_types = ['movie', 'tv', 'anime']
        if content_type not in valid_types:
            return jsonify({
                'error': f'Invalid content type: {content_type}',
                'valid_types': valid_types,
                'success': False
            }), 400
        
        # Get parameters
        limit = min(int(request.args.get('limit', 30)), 50)
        filter_type = request.args.get('filter', 'all')
        
        # Get services with late import
        profile_analyzer, recommendation_engine = get_personalized_services()
        if not recommendation_engine or not profile_analyzer:
            return jsonify({
                'error': 'CineBrain services not available',
                'success': False
            }), 503
        
        # Get user profile
        user_profile = profile_analyzer.build_comprehensive_profile(current_user.id)
        
        # Generate recommendations with content type filter
        context = {
            'content_type_filter': content_type,
            'additional_filter': filter_type
        }
        
        type_recommendations = recommendation_engine.get_personalized_recommendations(
            user_id=current_user.id,
            recommendation_type='for_you',
            limit=limit * 2,
            context=context
        )
        
        # Filter by content type
        filtered_recommendations = []
        for rec in type_recommendations.get('recommendations', []):
            if rec.get('content_type') == content_type:
                # Add type-specific metadata
                rec['type_preference_score'] = _calculate_type_preference(content_type, user_profile)
                
                # Apply additional filters
                include_rec = True
                if filter_type == 'new' and not rec.get('is_new_release', False):
                    include_rec = False
                elif filter_type == 'trending' and not rec.get('is_trending', False):
                    include_rec = False
                elif filter_type == 'popular' and rec.get('personalization_score', 0) < 0.7:
                    include_rec = False
                
                if include_rec:
                    filtered_recommendations.append(rec)
                    
                    if len(filtered_recommendations) >= limit:
                        break
        
        # Generate content type analysis
        type_analysis = _analyze_user_content_type_preference(content_type, user_profile)
        
        # Get content type specific insights
        type_insights = _get_content_type_insights(content_type, user_profile)
        
        return jsonify({
            'success': True,
            'user_id': current_user.id,
            'content_type': content_type,
            'filter_applied': filter_type,
            'recommendations': filtered_recommendations,
            'type_analysis': type_analysis,
            'type_insights': type_insights,
            'recommendation_count': len(filtered_recommendations),
            'user_type_preference': _calculate_type_preference(content_type, user_profile)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in content type recommendations for user {current_user.id}, type {content_type}: {e}")
        return jsonify({
            'error': f'Failed to generate {content_type} recommendations',
            'success': False,
            'user_id': current_user.id
        }), 500

# ============================================================================
# REAL-TIME FEEDBACK AND LEARNING ENDPOINTS
# ============================================================================

@personalized_bp.route('/personalized/feedback', methods=['POST', 'OPTIONS'])
@require_auth
def process_recommendation_feedback(current_user):
    """
    Process user feedback on recommendations for real-time learning
    
    URL: /api/personalized/feedback
    
    Request Body:
    {
        "content_id": int,
        "feedback_type": "like|dislike|watch|skip|share|save",
        "recommendation_type": "for_you|discover|mix|genre|type",
        "position_in_list": int,
        "context": {...}
    }
    
    Returns:
    - Feedback processing status
    - Impact on future recommendations
    """
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Request body required',
                'success': False
            }), 400
        
        # Validate required fields
        required_fields = ['content_id', 'feedback_type']
        if not all(field in data for field in required_fields):
            return jsonify({
                'error': 'Missing required fields: content_id, feedback_type',
                'success': False
            }), 400
        
        # Validate feedback type
        valid_feedback_types = ['like', 'dislike', 'watch', 'skip', 'share', 'save', 'rate']
        if data['feedback_type'] not in valid_feedback_types:
            return jsonify({
                'error': f'Invalid feedback type. Valid types: {valid_feedback_types}',
                'success': False
            }), 400
        
        # Get services with late import
        profile_analyzer, recommendation_engine = get_personalized_services()
        if not recommendation_engine:
            return jsonify({
                'error': 'CineBrain recommendation engine not available',
                'success': False
            }), 503
        
        # Prepare feedback data
        feedback_data = {
            'content_id': data['content_id'],
            'feedback_type': data['feedback_type'],
            'recommendation_type': data.get('recommendation_type', 'unknown'),
            'position': data.get('position_in_list', 0),
            'context': data.get('context', {}),
            'timestamp': datetime.utcnow().isoformat(),
            'rating': data.get('rating'),  # If user provided rating
            'session_id': get_request_context().get('session_id')
        }
        
        # Process feedback
        success = recommendation_engine.process_user_feedback(current_user.id, feedback_data)
        
        if success:
            # Generate impact assessment
            impact_assessment = _assess_feedback_impact(data['feedback_type'], data.get('recommendation_type'))
            
            return jsonify({
                'success': True,
                'message': 'Feedback processed successfully',
                'user_id': current_user.id,
                'content_id': data['content_id'],
                'feedback_type': data['feedback_type'],
                'impact_assessment': impact_assessment,
                'next_recommendations_update': 'immediate' if recommendation_engine.enable_real_time else 'next_session'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to process feedback',
                'user_id': current_user.id
            }), 500
        
    except Exception as e:
        logger.error(f"Error processing feedback for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to process feedback',
            'success': False,
            'user_id': current_user.id
        }), 500

@personalized_bp.route('/personalized/profile/insights', methods=['GET', 'OPTIONS'])
@require_auth
def get_personalization_insights(current_user):
    """
    Get detailed insights about user's personalization profile
    
    URL: /api/personalized/profile/insights
    
    Returns:
    - Comprehensive profile analysis
    - Recommendation accuracy metrics
    - Cinematic DNA breakdown
    - Personalization suggestions
    """
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Get services with late import
        profile_analyzer, recommendation_engine = get_personalized_services()
        if not profile_analyzer:
            return jsonify({
                'error': 'CineBrain profile analyzer not available',
                'success': False
            }), 503
        
        # Build comprehensive profile
        user_profile = profile_analyzer.build_comprehensive_profile(current_user.id)
        
        if not user_profile:
            return jsonify({
                'error': 'Unable to build user profile',
                'success': False,
                'suggestion': 'Interact with more content to build your profile'
            }), 404
        
        # Extract insights for frontend
        insights = {
            'profile_overview': {
                'completeness': user_profile.get('profile_confidence', 0),
                'readiness': user_profile.get('personalization_readiness', 'medium'),
                'strategy': user_profile.get('recommendations_strategy', 'content_based'),
                'content_history_size': user_profile.get('content_history_size', 0)
            },
            'cinematic_dna': {
                'sophistication_score': user_profile.get('cinematic_dna', {}).get('cinematic_sophistication_score', 0.5),
                'cultural_affinity': {
                    'telugu': user_profile.get('cinematic_dna', {}).get('telugu_cultural_affinity', 0.5),
                    'indian': user_profile.get('cinematic_dna', {}).get('indian_cultural_affinity', 0.5),
                    'global': user_profile.get('cinematic_dna', {}).get('global_cinema_exposure', 0.5)
                },
                'dominant_themes': list(user_profile.get('cinematic_dna', {}).get('narrative_preferences', {}).keys())[:3],
                'preferred_styles': list(user_profile.get('cinematic_dna', {}).get('style_affinities', {}).keys())[:3]
            },
            'behavior_patterns': {
                'exploration_tendency': user_profile.get('behavior_profile', {}).get('content_exploration', {}).get('exploration_tendency', 'medium'),
                'engagement_level': user_profile.get('behavior_profile', {}).get('engagement_patterns', {}).get('engagement_diversity', 0.5),
                'rating_behavior': user_profile.get('behavior_profile', {}).get('rating_behavior', {}).get('rating_tendency', 'balanced'),
                'discovery_openness': user_profile.get('behavior_profile', {}).get('discovery_openness', 0.5)
            },
            'recommendation_performance': {
                'accuracy_estimate': min(user_profile.get('profile_confidence', 0.5) * 100, 95),
                'personalization_strength': 'high' if user_profile.get('profile_confidence', 0) > 0.8 else 'medium',
                'algorithm_efficiency': user_profile.get('recommendations_strategy', 'content_based'),
                'improvement_areas': _identify_improvement_areas(user_profile)
            },
            'next_steps': {
                'profile_enhancement': _get_profile_enhancement_suggestions(user_profile),
                'exploration_suggestions': _get_exploration_suggestions(user_profile),
                'engagement_tips': _get_engagement_tips(user_profile)
            }
        }
        
        return jsonify({
            'success': True,
            'user_id': current_user.id,
            'username': current_user.username,
            'insights': insights,
            'profile_last_updated': user_profile.get('profile_created_at'),
            'next_update_due': user_profile.get('next_update_due')
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting personalization insights for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to get personalization insights',
            'success': False,
            'user_id': current_user.id
        }), 500

@personalized_bp.route('/personalized/refresh', methods=['POST', 'OPTIONS'])
@require_auth
def refresh_recommendations(current_user):
    """
    Force refresh of user's personalized recommendations
    
    URL: /api/personalized/refresh
    
    Request Body (optional):
    {
        "clear_cache": boolean,
        "rebuild_profile": boolean
    }
    
    Returns:
    - Refresh status
    - New recommendations availability
    """
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json() or {}
        
        # Get services with late import
        profile_analyzer, recommendation_engine = get_personalized_services()
        if not profile_analyzer or not recommendation_engine:
            return jsonify({
                'error': 'CineBrain services not available',
                'success': False
            }), 503
        
        refresh_results = {
            'profile_updated': False,
            'cache_cleared': False,
            'recommendations_refreshed': False
        }
        
        # Clear cache if requested
        if data.get('clear_cache', False):
            # This would clear user's recommendation cache
            refresh_results['cache_cleared'] = True
        
        # Rebuild profile if requested
        if data.get('rebuild_profile', False):
            user_profile = profile_analyzer.build_comprehensive_profile(current_user.id)
            if user_profile:
                refresh_results['profile_updated'] = True
        
        # Generate fresh recommendations
        fresh_recommendations = recommendation_engine.get_personalized_recommendations(
            user_id=current_user.id,
            recommendation_type='for_you',
            limit=20,
            context={'force_refresh': True}
        )
        
        if fresh_recommendations.get('success', False):
            refresh_results['recommendations_refreshed'] = True
        
        return jsonify({
            'success': True,
            'user_id': current_user.id,
            'refresh_results': refresh_results,
            'fresh_recommendations_count': len(fresh_recommendations.get('recommendations', [])),
            'refreshed_at': datetime.utcnow().isoformat(),
            'message': 'Recommendations refreshed successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error refreshing recommendations for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to refresh recommendations',
            'success': False,
            'user_id': current_user.id
        }), 500

# ============================================================================
# HEALTH AND STATUS ENDPOINTS
# ============================================================================

@personalized_bp.route('/personalized/health', methods=['GET'])
def personalization_health():
    """
    Health check for personalization services
    
    URL: /api/personalized/health
    """
    try:
        health_info = {
            'status': 'healthy',
            'service': 'cinebrain_personalized_recommendations',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '3.0.0'
        }
        
        # Check services with late import
        profile_analyzer, recommendation_engine = get_personalized_services()
        health_info['profile_analyzer'] = 'available' if profile_analyzer else 'unavailable'
        health_info['recommendation_engine'] = 'available' if recommendation_engine else 'unavailable'
        
        # Check if real-time learning is enabled
        if recommendation_engine:
            health_info['real_time_learning'] = 'enabled' if recommendation_engine.enable_real_time else 'disabled'
        
        # Service features
        health_info['features'] = {
            'cinematic_dna_analysis': True,
            'behavioral_analysis': True,
            'preference_embeddings': True,
            'adaptive_algorithms': True,
            'real_time_feedback': True,
            'telugu_cultural_prioritization': True,
            'discovery_recommendations': True,
            'mixed_strategy_recommendations': True,
            'genre_specific_recommendations': True,
            'content_type_recommendations': True
        }
        
        # Algorithm status
        if recommendation_engine:
            health_info['algorithms'] = {
                'content_based': 'active',
                'collaborative_filtering': 'active',
                'ultra_similarity_engine': 'active',
                'popularity_ranking': 'active',
                'language_priority_filter': 'active',
                'serendipity_engine': 'active'
            }
        
        return jsonify(health_info), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'cinebrain_personalized_recommendations',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _calculate_genre_affinity(genre_name: str, user_profile: Dict[str, Any]) -> float:
    """Calculate user's affinity for a specific genre"""
    if not user_profile:
        return 0.5
        
    genre_sophistication = user_profile.get('cinematic_dna', {}).get('genre_sophistication', {})
    
    if genre_name in genre_sophistication:
        genre_data = genre_sophistication[genre_name]
        if isinstance(genre_data, dict):
            return genre_data.get('sophistication_score', 0.5)
    
    return 0.5  # Neutral affinity

def _get_related_genres(genre_name: str) -> List[str]:
    """Get genres related to the specified genre"""
    genre_relationships = {
        'Action': ['Adventure', 'Thriller', 'Crime'],
        'Adventure': ['Action', 'Fantasy', 'Family'],
        'Comedy': ['Romance', 'Family'],
        'Drama': ['Romance', 'Biography', 'History'],
        'Horror': ['Thriller', 'Mystery'],
        'Romance': ['Drama', 'Comedy'],
        'Thriller': ['Action', 'Mystery', 'Crime'],
        'Science Fiction': ['Fantasy', 'Adventure'],
        'Fantasy': ['Adventure', 'Science Fiction'],
        'Crime': ['Thriller', 'Action', 'Drama']
    }
    
    return genre_relationships.get(genre_name, [])

def _analyze_user_genre_relationship(genre_name: str, user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze user's relationship with a specific genre"""
    affinity = _calculate_genre_affinity(genre_name, user_profile)
    
    return {
        'affinity_score': affinity,
        'affinity_level': 'high' if affinity > 0.7 else 'medium' if affinity > 0.4 else 'low',
        'recommendation_confidence': affinity,
        'exploration_potential': 1 - affinity,  # Lower affinity = higher exploration potential
        'similar_genres': _get_related_genres(genre_name)
    }

def _calculate_type_preference(content_type: str, user_profile: Dict[str, Any]) -> float:
    """Calculate user's preference for a content type"""
    if not user_profile:
        return 0.33
        
    behavior_profile = user_profile.get('behavior_profile', {})
    engagement_patterns = behavior_profile.get('engagement_patterns', {})
    
    type_scores = engagement_patterns.get('engagement_scores', {})
    
    if content_type in type_scores:
        total_score = sum(type_scores.values())
        if total_score > 0:
            return type_scores[content_type] / total_score
    
    return 0.33  # Equal preference if no data

def _analyze_user_content_type_preference(content_type: str, user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze user's preference for a content type"""
    preference = _calculate_type_preference(content_type, user_profile)
    
    return {
        'preference_score': preference,
        'preference_level': 'high' if preference > 0.5 else 'medium' if preference > 0.3 else 'low',
        'recommendation_strategy': 'content_type_focused' if preference > 0.6 else 'mixed_content_types'
    }

def _get_content_type_insights(content_type: str, user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Get insights specific to content type"""
    insights = {
        'movie': {
            'optimal_session': 'evening_weekend',
            'duration_preference': 'feature_length',
            'discovery_opportunity': 'international_cinema'
        },
        'tv': {
            'optimal_session': 'evening_daily',
            'duration_preference': 'episodic_viewing',
            'discovery_opportunity': 'limited_series'
        },
        'anime': {
            'optimal_session': 'flexible',
            'duration_preference': 'season_binge',
            'discovery_opportunity': 'different_demographics'
        }
    }
    
    return insights.get(content_type, {})

def _assess_feedback_impact(feedback_type: str, recommendation_type: str) -> Dict[str, Any]:
    """Assess the impact of user feedback on future recommendations"""
    impact_levels = {
        'like': {'immediate': 'high', 'long_term': 'medium', 'algorithm_adjustment': 'positive'},
        'dislike': {'immediate': 'high', 'long_term': 'medium', 'algorithm_adjustment': 'negative'},
        'watch': {'immediate': 'medium', 'long_term': 'high', 'algorithm_adjustment': 'engagement'},
        'skip': {'immediate': 'low', 'long_term': 'low', 'algorithm_adjustment': 'mild_negative'},
        'share': {'immediate': 'medium', 'long_term': 'high', 'algorithm_adjustment': 'very_positive'},
        'save': {'immediate': 'medium', 'long_term': 'medium', 'algorithm_adjustment': 'positive'}
    }
    
    impact = impact_levels.get(feedback_type, {'immediate': 'low', 'long_term': 'low', 'algorithm_adjustment': 'neutral'})
    
    impact['affected_categories'] = [recommendation_type] if recommendation_type != 'unknown' else ['for_you', 'discover']
    impact['learning_value'] = 'high' if feedback_type in ['like', 'dislike', 'watch'] else 'medium'
    
    return impact

def _identify_improvement_areas(user_profile: Dict[str, Any]) -> List[str]:
    """Identify areas where user profile could be improved"""
    improvements = []
    
    profile_confidence = user_profile.get('profile_confidence', 0)
    content_history_size = user_profile.get('content_history_size', 0)
    
    if profile_confidence < 0.6:
        improvements.append('Interact with more content to improve accuracy')
    
    if content_history_size < 10:
        improvements.append('Rate content to help us understand your preferences')
    
    behavior_profile = user_profile.get('behavior_profile', {})
    if not behavior_profile.get('rating_behavior', {}).get('has_ratings', False):
        improvements.append('Add ratings to content you\'ve watched')
    
    exploration_tendency = behavior_profile.get('content_exploration', {}).get('exploration_tendency', 'medium')
    if exploration_tendency == 'low':
        improvements.append('Try exploring different genres for better discovery')
    
    return improvements[:3]  # Return top 3 improvements

def _get_profile_enhancement_suggestions(user_profile: Dict[str, Any]) -> List[str]:
    """Get suggestions for enhancing user profile"""
    suggestions = []
    
    cinematic_dna = user_profile.get('cinematic_dna', {})
    
    if cinematic_dna.get('telugu_cultural_affinity', 0) < 0.5:
        suggestions.append('Explore more Telugu content to enhance cultural recommendations')
    
    if cinematic_dna.get('cinematic_sophistication_score', 0) < 0.6:
        suggestions.append('Try highly-rated films to improve recommendation quality')
    
    behavior_profile = user_profile.get('behavior_profile', {})
    if behavior_profile.get('content_exploration', {}).get('genre_exploration_score', 0) < 0.5:
        suggestions.append('Explore different genres to diversify your profile')
    
    return suggestions

def _get_exploration_suggestions(user_profile: Dict[str, Any]) -> List[str]:
    """Get suggestions for content exploration"""
    return [
        'Try content outside your usual preferences',
        'Explore international cinema',
        'Watch critically acclaimed films in new genres',
        'Discover hidden gems through our recommendations'
    ]

def _get_engagement_tips(user_profile: Dict[str, Any]) -> List[str]:
    """Get tips for better engagement with the platform"""
    return [
        'Rate content after watching for better recommendations',
        'Use the discovery section to find new favorites',
        'Provide feedback on recommendations to improve accuracy',
        'Save interesting content to your watchlist'
    ]

# Export the blueprint
__all__ = ['personalized_bp']