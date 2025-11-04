# backend/personalized/routes.py

"""
CineBrain Personalized Recommendation Routes
==========================================

Flask blueprint providing REST API endpoints for personalized recommendations
with modern feed-like behavior and real-time learning capabilities.
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
import jwt
from functools import wraps
import time

from .recommendation_engine import (
    ModernPersonalizationEngine,
    RecommendationOrchestrator,
    RecommendationResponse
)
from .profile_analyzer import (
    UserProfileAnalyzer,
    UserInteractionEvent
)
from .utils import (
    EmbeddingManager,
    SimilarityCalculator,
    CacheManager,
    PerformanceOptimizer,
    create_cache_key
)

# Create blueprint
personalized_bp = Blueprint('personalized', __name__)

logger = logging.getLogger(__name__)

# Global variables for dependency injection
personalization_engine: Optional[ModernPersonalizationEngine] = None
profile_analyzer: Optional[UserProfileAnalyzer] = None
cache_manager: Optional[CacheManager] = None
performance_optimizer = PerformanceOptimizer()

def require_auth(f):
    """Authentication decorator for personalized routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return '', 200
        
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                'success': False,
                'error': 'CineBrain authentication required',
                'code': 'AUTH_REQUIRED'
            }), 401
        
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user_id = payload.get('user_id')
            
            if not request.user_id:
                raise jwt.InvalidTokenError("User ID not found in token")
            
            return f(*args, **kwargs)
            
        except jwt.ExpiredSignatureError:
            return jsonify({
                'success': False,
                'error': 'CineBrain token expired',
                'code': 'TOKEN_EXPIRED'
            }), 401
        except jwt.InvalidTokenError as e:
            return jsonify({
                'success': False,
                'error': 'Invalid CineBrain token',
                'code': 'TOKEN_INVALID',
                'details': str(e)
            }), 401
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return jsonify({
                'success': False,
                'error': 'Authentication failed',
                'code': 'AUTH_FAILED'
            }), 401
    
    return decorated_function

def validate_request_params(required_params: List[str] = None, optional_params: Dict[str, Any] = None):
    """Decorator to validate request parameters"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Get request data
                if request.is_json:
                    data = request.get_json() or {}
                else:
                    data = request.args.to_dict()
                
                # Validate required parameters
                if required_params:
                    missing_params = [param for param in required_params if param not in data]
                    if missing_params:
                        return jsonify({
                            'success': False,
                            'error': f'Missing required parameters: {", ".join(missing_params)}',
                            'code': 'MISSING_PARAMS'
                        }), 400
                
                # Apply default values for optional parameters
                if optional_params:
                    for param, default_value in optional_params.items():
                        if param not in data:
                            data[param] = default_value
                
                # Add validated data to request
                request.validated_data = data
                
                return f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Parameter validation error: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Invalid request parameters',
                    'code': 'INVALID_PARAMS'
                }), 400
        
        return decorated_function
    return decorator

@personalized_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for personalization service"""
    try:
        health_info = {
            'status': 'healthy',
            'service': 'cinebrain_personalization',
            'version': '2.0.0',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'personalization_engine': personalization_engine is not None,
                'profile_analyzer': profile_analyzer is not None,
                'cache_manager': cache_manager is not None,
                'performance_optimizer': True
            }
        }
        
        # Add performance statistics
        if performance_optimizer:
            health_info['performance'] = performance_optimizer.get_performance_stats()
        
        # Add cache statistics
        if cache_manager:
            health_info['cache'] = cache_manager.get_cache_stats()
        
        return jsonify(health_info), 200
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'service': 'cinebrain_personalization',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@personalized_bp.route('/recommendations', methods=['GET', 'OPTIONS'])
@require_auth
@validate_request_params(
    optional_params={
        'limit': 50,
        'category': 'feed',
        'include_metadata': True,
        'force_refresh': False
    }
)
@performance_optimizer.time_function('get_personalized_recommendations')
def get_personalized_recommendations():
    """
    Get personalized recommendations for the authenticated user
    
    Query Parameters:
    - limit (int): Maximum number of recommendations (default: 50, max: 100)
    - category (str): Recommendation category ('feed', 'discover', 'trending')
    - include_metadata (bool): Include algorithm metadata (default: True)
    - force_refresh (bool): Force cache refresh (default: False)
    
    Returns:
    - JSON response with personalized recommendations
    """
    try:
        if not personalization_engine:
            return jsonify({
                'success': False,
                'error': 'CineBrain personalization engine not available',
                'code': 'SERVICE_UNAVAILABLE'
            }), 503
        
        user_id = request.user_id
        data = request.validated_data
        
        # Validate and sanitize parameters
        limit = min(int(data.get('limit', 50)), 100)  # Cap at 100
        category = data.get('category', 'feed')
        include_metadata = data.get('include_metadata', True)
        force_refresh = data.get('force_refresh', False)
        
        # Check cache if not forcing refresh
        if not force_refresh and cache_manager:
            cache_key = create_cache_key('user_recommendations', user_id, category, limit)
            cached_result = cache_manager.cache.get(cache_key) if cache_manager.cache else None
            
            if cached_result:
                try:
                    cached_data = json.loads(cached_result)
                    logger.info(f"Returning cached recommendations for user {user_id}")
                    return jsonify(cached_data), 200
                except Exception as e:
                    logger.warning(f"Error loading cached data: {e}")
        
        # Generate fresh recommendations
        start_time = time.time()
        
        recommendation_response = personalization_engine.generate_personalized_feed(
            user_id=user_id,
            limit=limit
        )
        
        generation_time = time.time() - start_time
        
        # Format response
        response_data = {
            'success': True,
            'user_id': user_id,
            'category': category,
            'recommendations': [
                {
                    'id': rec.content_id,
                    'title': rec.title,
                    'content_type': rec.content_type,
                    'genres': rec.genres,
                    'languages': rec.languages,
                    'rating': rec.rating,
                    'release_date': rec.release_date,
                    'poster_path': rec.poster_path,
                    'overview': rec.overview,
                    'youtube_trailer_id': rec.youtube_trailer_id,
                    'score': rec.recommendation_score,
                    'reasons': rec.recommendation_reasons,
                    'confidence': rec.confidence_level,
                    'source_algorithm': rec.algorithm_source
                } | (
                    {'personalization_factors': rec.personalization_factors} if include_metadata else {}
                )
                for rec in recommendation_response.recommendations
            ],
            'total_count': recommendation_response.total_count,
            'generated_at': recommendation_response.generated_at.isoformat(),
            'generation_time_ms': round(generation_time * 1000, 2)
        }
        
        # Add metadata if requested
        if include_metadata:
            response_data['metadata'] = {
                'algorithm_breakdown': recommendation_response.algorithm_breakdown,
                'personalization_strength': round(recommendation_response.personalization_strength, 3),
                'freshness_score': round(recommendation_response.freshness_score, 3),
                'diversity_score': round(recommendation_response.diversity_score, 3),
                'cache_duration': recommendation_response.cache_duration,
                'next_refresh': recommendation_response.next_refresh.isoformat(),
                'telugu_priority_applied': True,
                'engine_version': '2.0.0'
            }
        
        # Cache the response
        if cache_manager and cache_manager.cache:
            try:
                cache_key = create_cache_key('user_recommendations', user_id, category, limit)
                cache_manager.cache.set(
                    cache_key, 
                    json.dumps(response_data), 
                    timeout=recommendation_response.cache_duration
                )
            except Exception as e:
                logger.warning(f"Error caching recommendations: {e}")
        
        logger.info(f"Generated {len(recommendation_response.recommendations)} recommendations for user {user_id} in {generation_time:.2f}s")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error generating recommendations for user {request.user_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to generate CineBrain recommendations',
            'code': 'GENERATION_FAILED',
            'details': str(e) if current_app.debug else None
        }), 500

@personalized_bp.route('/interaction', methods=['POST', 'OPTIONS'])
@require_auth
@validate_request_params(
    required_params=['content_id', 'interaction_type'],
    optional_params={
        'rating': None,
        'session_id': None,
        'context': {}
    }
)
@performance_optimizer.time_function('record_user_interaction')
def record_user_interaction():
    """
    Record user interaction for real-time learning
    
    Request Body:
    - content_id (int): ID of the content interacted with
    - interaction_type (str): Type of interaction ('view', 'like', 'favorite', 'rating', etc.)
    - rating (float, optional): User rating (1-10)
    - session_id (str, optional): Session identifier
    - context (dict, optional): Additional context information
    
    Returns:
    - JSON response with success status
    """
    try:
        if not profile_analyzer:
            return jsonify({
                'success': False,
                'error': 'CineBrain profile analyzer not available',
                'code': 'SERVICE_UNAVAILABLE'
            }), 503
        
        user_id = request.user_id
        data = request.validated_data
        
        # Validate interaction type
        valid_interactions = ['view', 'like', 'favorite', 'rating', 'share', 'watchlist', 'search']
        interaction_type = data['interaction_type']
        
        if interaction_type not in valid_interactions:
            return jsonify({
                'success': False,
                'error': f'Invalid interaction type. Valid types: {", ".join(valid_interactions)}',
                'code': 'INVALID_INTERACTION_TYPE'
            }), 400
        
        # Validate rating if provided
        rating = data.get('rating')
        if rating is not None:
            try:
                rating = float(rating)
                if not (1 <= rating <= 10):
                    return jsonify({
                        'success': False,
                        'error': 'Rating must be between 1 and 10',
                        'code': 'INVALID_RATING'
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'error': 'Rating must be a valid number',
                    'code': 'INVALID_RATING_FORMAT'
                }), 400
        
        # Create interaction event
        interaction_event = UserInteractionEvent(
            user_id=user_id,
            content_id=int(data['content_id']),
            interaction_type=interaction_type,
            timestamp=datetime.utcnow(),
            rating=rating,
            session_id=data.get('session_id'),
            metadata=data.get('metadata', {}),
            context=data.get('context', {})
        )
        
        # Update user profile in real-time
        success = profile_analyzer.update_profile_realtime(user_id, interaction_event)
        
        if success:
            # Invalidate recommendation cache for this user
            if cache_manager:
                cache_manager.invalidate_user_cache(user_id)
            
            logger.info(f"Recorded {interaction_type} interaction for user {user_id} on content {data['content_id']}")
            
            return jsonify({
                'success': True,
                'message': 'Interaction recorded successfully',
                'user_id': user_id,
                'content_id': data['content_id'],
                'interaction_type': interaction_type,
                'timestamp': interaction_event.timestamp.isoformat(),
                'profile_updated': True,
                'cache_invalidated': True
            }), 200
        else:
            logger.error(f"Failed to record interaction for user {user_id}")
            return jsonify({
                'success': False,
                'error': 'Failed to record interaction',
                'code': 'INTERACTION_FAILED'
            }), 500
            
    except Exception as e:
        logger.error(f"Error recording interaction for user {request.user_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to record CineBrain interaction',
            'code': 'INTERACTION_ERROR',
            'details': str(e) if current_app.debug else None
        }), 500

@personalized_bp.route('/profile', methods=['GET', 'OPTIONS'])
@require_auth
@performance_optimizer.time_function('get_user_profile')
def get_user_profile():
    """
    Get comprehensive user profile and preferences
    
    Returns:
    - JSON response with user preference profile
    """
    try:
        if not profile_analyzer:
            return jsonify({
                'success': False,
                'error': 'CineBrain profile analyzer not available',
                'code': 'SERVICE_UNAVAILABLE'
            }), 503
        
        user_id = request.user_id
        
        # Build comprehensive user profile
        user_profile = profile_analyzer.build_comprehensive_user_profile(user_id)
        
        if not user_profile:
            return jsonify({
                'success': False,
                'error': 'Could not build user profile',
                'code': 'PROFILE_BUILD_FAILED',
                'suggestion': 'Interact with more CineBrain content to build your profile'
            }), 404
        
        # Format profile for API response
        profile_data = {
            'success': True,
            'user_id': user_profile.user_id,
            'profile': {
                'genre_preferences': user_profile.genre_preferences,
                'language_preferences': user_profile.language_preferences,
                'content_type_preferences': user_profile.content_type_preferences,
                'quality_threshold': user_profile.quality_threshold,
                'sophistication_score': round(user_profile.sophistication_score, 3),
                'engagement_level': user_profile.engagement_level,
                'confidence_score': round(user_profile.confidence_score, 3),
                'last_updated': user_profile.last_updated.isoformat()
            },
            'insights': {
                'temporal_patterns': user_profile.temporal_patterns,
                'cinematic_dna': user_profile.cinematic_dna,
                'telugu_affinity': user_profile.cinematic_dna.get('telugu_cinema_affinity', 0),
                'personalization_level': 'high' if user_profile.confidence_score > 0.7 else 'moderate' if user_profile.confidence_score > 0.4 else 'developing'
            },
            'recommendations_info': {
                'accuracy_estimate': min(user_profile.confidence_score * 100, 95),
                'improvement_tip': _get_improvement_tip(user_profile),
                'next_optimization': (datetime.utcnow() + timedelta(days=7)).isoformat()
            },
            'generated_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Retrieved profile for user {user_id} with confidence {user_profile.confidence_score}")
        
        return jsonify(profile_data), 200
        
    except Exception as e:
        logger.error(f"Error getting profile for user {request.user_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get CineBrain user profile',
            'code': 'PROFILE_ERROR',
            'details': str(e) if current_app.debug else None
        }), 500

@personalized_bp.route('/similar-users', methods=['GET', 'OPTIONS'])
@require_auth
@validate_request_params(
    optional_params={'limit': 10}
)
@performance_optimizer.time_function('get_similar_users')
def get_similar_users():
    """
    Get users with similar preferences
    
    Query Parameters:
    - limit (int): Maximum number of similar users (default: 10, max: 20)
    
    Returns:
    - JSON response with similar users
    """
    try:
        if not personalization_engine:
            return jsonify({
                'success': False,
                'error': 'CineBrain personalization engine not available',
                'code': 'SERVICE_UNAVAILABLE'
            }), 503
        
        user_id = request.user_id
        limit = min(int(request.validated_data.get('limit', 10)), 20)
        
        # Get user embedding
        user_embedding = personalization_engine.embedding_manager.get_user_embedding(user_id)
        
        if user_embedding is None:
            return jsonify({
                'success': False,
                'error': 'User profile not found',
                'code': 'PROFILE_NOT_FOUND',
                'suggestion': 'Interact with more CineBrain content to build your profile'
            }), 404
        
        # This would typically involve getting other user embeddings
        # For demo purposes, we'll return a simplified response
        
        similar_users_data = {
            'success': True,
            'user_id': user_id,
            'similar_users': [],  # Would be populated with actual similar users
            'total_count': 0,
            'message': 'Similar user functionality available in advanced mode',
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return jsonify(similar_users_data), 200
        
    except Exception as e:
        logger.error(f"Error getting similar users for user {request.user_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get similar CineBrain users',
            'code': 'SIMILAR_USERS_ERROR'
        }), 500

@personalized_bp.route('/refresh', methods=['POST', 'OPTIONS'])
@require_auth
@performance_optimizer.time_function('refresh_user_profile')
def refresh_user_profile():
    """
    Force refresh user profile and recommendations
    
    Returns:
    - JSON response with refresh status
    """
    try:
        if not profile_analyzer or not cache_manager:
            return jsonify({
                'success': False,
                'error': 'CineBrain services not available',
                'code': 'SERVICE_UNAVAILABLE'
            }), 503
        
        user_id = request.user_id
        
        # Invalidate all user caches
        cache_manager.invalidate_user_cache(user_id)
        
        # Force rebuild user profile
        user_profile = profile_analyzer.build_comprehensive_user_profile(user_id, force_refresh=True)
        
        if user_profile:
            refresh_data = {
                'success': True,
                'message': 'User profile refreshed successfully',
                'user_id': user_id,
                'profile_confidence': round(user_profile.confidence_score, 3),
                'last_updated': user_profile.last_updated.isoformat(),
                'cache_cleared': True,
                'recommendations_will_refresh': True,
                'refresh_timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Refreshed profile for user {user_id}")
            return jsonify(refresh_data), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to refresh user profile',
                'code': 'REFRESH_FAILED'
            }), 500
            
    except Exception as e:
        logger.error(f"Error refreshing profile for user {request.user_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to refresh CineBrain profile',
            'code': 'REFRESH_ERROR'
        }), 500

@personalized_bp.route('/stats', methods=['GET', 'OPTIONS'])
@require_auth
def get_personalization_stats():
    """
    Get personalization service statistics
    
    Returns:
    - JSON response with service statistics
    """
    try:
        stats_data = {
            'success': True,
            'service': 'cinebrain_personalization',
            'version': '2.0.0',
            'user_id': request.user_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add performance statistics
        if performance_optimizer:
            stats_data['performance'] = performance_optimizer.get_performance_stats()
        
        # Add cache statistics
        if cache_manager:
            stats_data['cache'] = cache_manager.get_cache_stats()
        
        # Add profile analyzer statistics
        if profile_analyzer:
            stats_data['profile_analyzer'] = profile_analyzer.get_performance_stats()
        
        return jsonify(stats_data), 200
        
    except Exception as e:
        logger.error(f"Error getting personalization stats: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get personalization stats',
            'code': 'STATS_ERROR'
        }), 500

# Error handlers
@personalized_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'CineBrain personalization endpoint not found',
        'code': 'ENDPOINT_NOT_FOUND'
    }), 404

@personalized_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'success': False,
        'error': 'Method not allowed for this CineBrain endpoint',
        'code': 'METHOD_NOT_ALLOWED'
    }), 405

@personalized_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'CineBrain personalization service error',
        'code': 'INTERNAL_ERROR'
    }), 500

# CORS handling
@personalized_bp.after_request
def after_request(response):
    """Add CORS headers to all responses"""
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
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response

# Helper functions
def _get_improvement_tip(user_profile) -> str:
    """Get personalized improvement tip for user"""
    if user_profile.confidence_score < 0.3:
        return 'Interact with more CineBrain content (like, favorite, rate) to improve recommendations'
    elif user_profile.confidence_score < 0.6:
        return 'Rate more CineBrain content to help our AI understand your preferences better'
    elif user_profile.confidence_score < 0.8:
        return 'Explore different genres on CineBrain to get more diverse recommendations'
    else:
        return 'Your CineBrain recommendations are highly optimized! Keep discovering new content'

# Initialize function to be called from main app
def init_personalized_routes(personalization_system):
    """Initialize routes with personalization system components"""
    global personalization_engine, profile_analyzer, cache_manager
    
    personalization_engine = personalization_system.get('recommendation_engine')
    profile_analyzer = personalization_system.get('profile_analyzer')
    cache_manager = personalization_system.get('cache_manager')
    
    logger.info("CineBrain personalized routes initialized with system components")

# Export blueprint
__all__ = ['personalized_bp', 'init_personalized_routes']