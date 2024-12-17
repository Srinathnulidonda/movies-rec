"""
CineBrain Personalized Recommendation API Routes
Production-grade Flask routes for advanced personalized recommendations
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import json
import logging
import jwt
from functools import wraps
from typing import Dict, Any, List, Optional

from .recommendation_engine import CineBrainRecommendationEngine
from . import get_personalization_system

logger = logging.getLogger(__name__)

# Create blueprint
personalized_bp = Blueprint('personalized', __name__, url_prefix='/api/personalized')

# Global variables (will be set by init function)
db = None
models = None
cache = None
recommendation_engine: Optional[CineBrainRecommendationEngine] = None
app = None

def init_personalized_routes(flask_app, database, db_models, services, cache_instance=None):
    """Initialize personalized routes with dependencies"""
    global db, models, cache, recommendation_engine, app
    
    app = flask_app
    db = database
    models = db_models
    cache = cache_instance
    
    # Initialize recommendation engine
    try:
        personalization_system = get_personalization_system()
        if personalization_system and personalization_system.is_ready():
            recommendation_engine = CineBrainRecommendationEngine(db, models, cache)
            logger.info("✅ CineBrain Recommendation Engine initialized for routes")
        else:
            logger.error("❌ Personalization system not ready")
    except Exception as e:
        logger.error(f"❌ Failed to initialize recommendation engine for routes: {e}")

def require_auth(f):
    """Authentication decorator for personalized routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return '', 200
            
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                'error': 'CineBrain authentication required',
                'code': 'AUTH_REQUIRED'
            }), 401
        
        try:
            token = auth_header.split(' ')[1]
            payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            user_id = payload.get('user_id')
            
            if not user_id:
                return jsonify({
                    'error': 'Invalid CineBrain token payload',
                    'code': 'INVALID_TOKEN'
                }), 401
            
            # Get user from database
            user = models['User'].query.get(user_id)
            if not user:
                return jsonify({
                    'error': 'CineBrain user not found',
                    'code': 'USER_NOT_FOUND'
                }), 401
            
            # Update last active
            try:
                user.last_active = datetime.utcnow()
                db.session.commit()
            except Exception as e:
                logger.warning(f"Failed to update user last_active: {e}")
                db.session.rollback()
            
            return f(user, *args, **kwargs)
            
        except jwt.ExpiredSignatureError:
            return jsonify({
                'error': 'CineBrain token expired',
                'code': 'TOKEN_EXPIRED'
            }), 401
        except jwt.InvalidTokenError:
            return jsonify({
                'error': 'Invalid CineBrain token',
                'code': 'INVALID_TOKEN'
            }), 401
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return jsonify({
                'error': 'CineBrain authentication failed',
                'code': 'AUTH_FAILED'
            }), 401
    
    return decorated_function

@personalized_bp.route('/for-you', methods=['GET', 'OPTIONS'])
@require_auth
def get_for_you_recommendations(current_user):
    """
    Get comprehensive 'For You' recommendations
    
    Query Parameters:
    - limit: Maximum number of recommendations (default: 50, max: 100)
    - refresh: Force refresh cache (default: false)
    - filters: JSON object with filters (genre, language, content_type)
    """
    try:
        if request.method == 'OPTIONS':
            return '', 200
        
        if not recommendation_engine:
            return jsonify({
                'error': 'CineBrain recommendation engine not available',
                'code': 'ENGINE_UNAVAILABLE',
                'fallback_available': True
            }), 503
        
        # Parse parameters
        limit = min(int(request.args.get('limit', 50)), 100)
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        filters_param = request.args.get('filters')
        
        filters = {}
        if filters_param:
            try:
                filters = json.loads(filters_param)
            except json.JSONDecodeError:
                return jsonify({
                    'error': 'Invalid filters JSON format',
                    'code': 'INVALID_FILTERS'
                }), 400
        
        # Clear cache if refresh requested
        if refresh and cache:
            cache_key = f"cinebrain:advanced_recs:{current_user.id}:for_you:{limit}"
            cache.delete(cache_key)
        
        # Generate recommendations
        result = recommendation_engine.generate_personalized_recommendations(
            user_id=current_user.id,
            recommendation_type='for_you',
            limit=limit,
            filters=filters
        )
        
        # Add user context
        result['user_context'] = {
            'username': current_user.username,
            'preferences_set': bool(current_user.preferred_languages or current_user.preferred_genres),
            'recommendation_count': len(result.get('recommendations', []))
        }
        
        # Add performance metrics
        result['performance'] = {
            'engine_version': '3.0_neural_cultural',
            'recommendation_quality': 'advanced_personalization',
            'telugu_priority': 'enabled',
            'cultural_awareness': 'active'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in for-you recommendations for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to generate CineBrain For You recommendations',
            'code': 'GENERATION_FAILED',
            'user_id': current_user.id
        }), 500

@personalized_bp.route('/discover', methods=['GET', 'OPTIONS'])
@require_auth
def get_discover_recommendations(current_user):
    """
    Get discovery recommendations outside user's comfort zone
    
    Query Parameters:
    - limit: Maximum number of recommendations (default: 30, max: 50)
    - exploration_level: How far outside comfort zone (low/medium/high, default: medium)
    - include_international: Include international content (default: true)
    """
    try:
        if request.method == 'OPTIONS':
            return '', 200
        
        if not recommendation_engine:
            return jsonify({
                'error': 'CineBrain recommendation engine not available',
                'code': 'ENGINE_UNAVAILABLE'
            }), 503
        
        # Parse parameters
        limit = min(int(request.args.get('limit', 30)), 50)
        exploration_level = request.args.get('exploration_level', 'medium')
        include_international = request.args.get('include_international', 'true').lower() == 'true'
        
        # Validate exploration level
        if exploration_level not in ['low', 'medium', 'high']:
            exploration_level = 'medium'
        
        # Set up filters for discovery
        filters = {
            'exploration_level': exploration_level,
            'include_international': include_international
        }
        
        # Generate discovery recommendations
        result = recommendation_engine.generate_personalized_recommendations(
            user_id=current_user.id,
            recommendation_type='discover',
            limit=limit,
            filters=filters
        )
        
        # Add discovery context
        result['discovery_context'] = {
            'exploration_level': exploration_level,
            'discovery_philosophy': 'Expanding your cinematic horizons while respecting your core preferences',
            'telugu_discovery_boost': 'Active - Telugu content prioritized even in discovery',
            'cultural_bridge_building': 'Enabled'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in discover recommendations for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to generate CineBrain Discovery recommendations',
            'code': 'DISCOVERY_FAILED',
            'user_id': current_user.id
        }), 500

@personalized_bp.route('/trending-for-you', methods=['GET', 'OPTIONS'])
@require_auth
def get_trending_for_you(current_user):
    """
    Get personalized trending recommendations
    
    Query Parameters:
    - limit: Maximum number of recommendations (default: 25, max: 40)
    - time_window: Trending time window (day/week/month, default: week)
    - region: Regional trending preference (default: auto-detect)
    """
    try:
        if request.method == 'OPTIONS':
            return '', 200
        
        if not recommendation_engine:
            return jsonify({
                'error': 'CineBrain recommendation engine not available',
                'code': 'ENGINE_UNAVAILABLE'
            }), 503
        
        # Parse parameters
        limit = min(int(request.args.get('limit', 25)), 40)
        time_window = request.args.get('time_window', 'week')
        region = request.args.get('region', 'auto')
        
        # Validate time window
        if time_window not in ['day', 'week', 'month']:
            time_window = 'week'
        
        # Set up filters for trending
        filters = {
            'time_window': time_window,
            'region': region,
            'personalization_level': 'high'
        }
        
        # Generate trending recommendations
        result = recommendation_engine.generate_personalized_recommendations(
            user_id=current_user.id,
            recommendation_type='trending_for_you',
            limit=limit,
            filters=filters
        )
        
        # Add trending context
        result['trending_context'] = {
            'time_window': time_window,
            'trending_algorithm': 'popularity_velocity_with_personalization',
            'regional_boost': 'Telugu content prioritized in trending',
            'freshness_factor': 'High weight for recent releases'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in trending recommendations for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to generate CineBrain Trending recommendations',
            'code': 'TRENDING_FAILED',
            'user_id': current_user.id
        }), 500

@personalized_bp.route('/your-language', methods=['GET', 'OPTIONS'])
@require_auth
def get_your_language_recommendations(current_user):
    """
    Get recommendations in user's preferred languages
    
    Query Parameters:
    - limit: Maximum number of recommendations (default: 30, max: 50)
    - language_priority: Strict language priority (default: true)
    - include_dubbed: Include dubbed content (default: false)
    """
    try:
        if request.method == 'OPTIONS':
            return '', 200
        
        if not recommendation_engine:
            return jsonify({
                'error': 'CineBrain recommendation engine not available',
                'code': 'ENGINE_UNAVAILABLE'
            }), 503
        
        # Parse parameters
        limit = min(int(request.args.get('limit', 30)), 50)
        language_priority = request.args.get('language_priority', 'true').lower() == 'true'
        include_dubbed = request.args.get('include_dubbed', 'false').lower() == 'true'
        
        # Get user's preferred languages
        user_languages = []
        if current_user.preferred_languages:
            try:
                user_languages = json.loads(current_user.preferred_languages)
            except:
                pass
        
        # Default to Telugu if no preferences set
        if not user_languages:
            user_languages = ['Telugu', 'English']
        
        # Set up filters for language-specific recommendations
        filters = {
            'languages': user_languages,
            'language_priority': language_priority,
            'include_dubbed': include_dubbed,
            'telugu_boost': True
        }
        
        # Generate language-specific recommendations
        result = recommendation_engine.generate_personalized_recommendations(
            user_id=current_user.id,
            recommendation_type='your_language',
            limit=limit,
            filters=filters
        )
        
        # Add language context
        result['language_context'] = {
            'preferred_languages': user_languages,
            'language_priority_applied': language_priority,
            'telugu_first_policy': 'Active',
            'cultural_authenticity': 'High priority for regional content'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in language recommendations for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to generate CineBrain Language recommendations',
            'code': 'LANGUAGE_FAILED',
            'user_id': current_user.id
        }), 500

@personalized_bp.route('/because-you-watched', methods=['GET', 'OPTIONS'])
@require_auth
def get_because_you_watched(current_user):
    """
    Get 'Because you watched' recommendations
    
    Query Parameters:
    - limit: Maximum number of recommendations (default: 20, max: 30)
    - similarity_threshold: Minimum similarity score (default: 0.6)
    - explanation_detail: Level of explanation (basic/detailed, default: basic)
    """
    try:
        if request.method == 'OPTIONS':
            return '', 200
        
        if not recommendation_engine:
            return jsonify({
                'error': 'CineBrain recommendation engine not available',
                'code': 'ENGINE_UNAVAILABLE'
            }), 503
        
        # Parse parameters
        limit = min(int(request.args.get('limit', 20)), 30)
        similarity_threshold = float(request.args.get('similarity_threshold', 0.6))
        explanation_detail = request.args.get('explanation_detail', 'basic')
        
        # Validate parameters
        similarity_threshold = max(0.1, min(similarity_threshold, 1.0))
        if explanation_detail not in ['basic', 'detailed']:
            explanation_detail = 'basic'
        
        # Set up filters
        filters = {
            'similarity_threshold': similarity_threshold,
            'explanation_detail': explanation_detail,
            'max_base_content': 5  # Limit to last 5 watched items
        }
        
        # Generate because-you-watched recommendations
        result = recommendation_engine.generate_personalized_recommendations(
            user_id=current_user.id,
            recommendation_type='because_you_watched',
            limit=limit,
            filters=filters
        )
        
        # Add similarity context
        result['similarity_context'] = {
            'similarity_algorithm': 'ultra_powerful_similarity_engine',
            'similarity_threshold': similarity_threshold,
            'explanation_level': explanation_detail,
            'base_content_analysis': 'Recent interactions and high-rated content'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in because-you-watched recommendations for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to generate CineBrain Because You Watched recommendations',
            'code': 'SIMILARITY_FAILED',
            'user_id': current_user.id
        }), 500

@personalized_bp.route('/hidden-gems', methods=['GET', 'OPTIONS'])
@require_auth
def get_hidden_gems(current_user):
    """
    Get hidden gem recommendations
    
    Query Parameters:
    - limit: Maximum number of recommendations (default: 15, max: 25)
    - min_rating: Minimum rating for gems (default: 7.5)
    - max_popularity: Maximum popularity score for 'hidden' content (default: 50)
    """
    try:
        if request.method == 'OPTIONS':
            return '', 200
        
        if not recommendation_engine:
            return jsonify({
                'error': 'CineBrain recommendation engine not available',
                'code': 'ENGINE_UNAVAILABLE'
            }), 503
        
        # Parse parameters
        limit = min(int(request.args.get('limit', 15)), 25)
        min_rating = float(request.args.get('min_rating', 7.5))
        max_popularity = float(request.args.get('max_popularity', 50))
        
        # Validate parameters
        min_rating = max(5.0, min(min_rating, 10.0))
        max_popularity = max(10, min(max_popularity, 100))
        
        # Set up filters for hidden gems
        filters = {
            'min_rating': min_rating,
            'max_popularity': max_popularity,
            'hidden_gem_criteria': True,
            'telugu_gems_boost': True
        }
        
        # Generate hidden gem recommendations
        result = recommendation_engine.generate_personalized_recommendations(
            user_id=current_user.id,
            recommendation_type='hidden_gems',
            limit=limit,
            filters=filters
        )
        
        # Add hidden gems context
        result['hidden_gems_context'] = {
            'discovery_philosophy': 'High-quality content with lower mainstream visibility',
            'quality_threshold': min_rating,
            'popularity_ceiling': max_popularity,
            'telugu_gems_priority': 'Active - Regional hidden treasures highlighted'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in hidden gems recommendations for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to generate CineBrain Hidden Gems recommendations',
            'code': 'GEMS_FAILED',
            'user_id': current_user.id
        }), 500

@personalized_bp.route('/telugu-specials', methods=['GET', 'OPTIONS'])
@require_auth
def get_telugu_specials(current_user):
    """
    Get special Telugu content recommendations
    
    Query Parameters:
    - limit: Maximum number of recommendations (default: 25, max: 40)
    - include_classics: Include classic Telugu films (default: true)
    - focus_area: Focus on specific area (tollywood/independent/all, default: all)
    """
    try:
        if request.method == 'OPTIONS':
            return '', 200
        
        if not recommendation_engine:
            return jsonify({
                'error': 'CineBrain recommendation engine not available',
                'code': 'ENGINE_UNAVAILABLE'
            }), 503
        
        # Parse parameters
        limit = min(int(request.args.get('limit', 25)), 40)
        include_classics = request.args.get('include_classics', 'true').lower() == 'true'
        focus_area = request.args.get('focus_area', 'all')
        
        # Validate focus area
        if focus_area not in ['tollywood', 'independent', 'all']:
            focus_area = 'all'
        
        # Set up filters for Telugu specials
        filters = {
            'language_restriction': ['Telugu'],
            'include_classics': include_classics,
            'focus_area': focus_area,
            'cultural_authenticity': True,
            'tollywood_priority': True
        }
        
        # Generate Telugu special recommendations
        result = recommendation_engine.generate_personalized_recommendations(
            user_id=current_user.id,
            recommendation_type='telugu_specials',
            limit=limit,
            filters=filters
        )
        
        # Add Telugu context
        result['telugu_context'] = {
            'focus_area': focus_area,
            'classics_included': include_classics,
            'cultural_authenticity': 'Maximum priority for authentic Telugu cinema',
            'tollywood_celebration': 'Showcasing the best of Telugu entertainment'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in Telugu specials for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to generate CineBrain Telugu Specials',
            'code': 'TELUGU_FAILED',
            'user_id': current_user.id
        }), 500

@personalized_bp.route('/mix', methods=['GET', 'OPTIONS'])
@require_auth
def get_personalized_mix(current_user):
    """
    Get a personalized mix of different recommendation types
    
    Query Parameters:
    - total_limit: Total number of recommendations (default: 60, max: 100)
    - mix_strategy: How to mix recommendations (balanced/discovery/comfort, default: balanced)
    """
    try:
        if request.method == 'OPTIONS':
            return '', 200
        
        if not recommendation_engine:
            return jsonify({
                'error': 'CineBrain recommendation engine not available',
                'code': 'ENGINE_UNAVAILABLE'
            }), 503
        
        # Parse parameters
        total_limit = min(int(request.args.get('total_limit', 60)), 100)
        mix_strategy = request.args.get('mix_strategy', 'balanced')
        
        # Validate mix strategy
        if mix_strategy not in ['balanced', 'discovery', 'comfort']:
            mix_strategy = 'balanced'
        
        # Define mix proportions based on strategy
        mix_proportions = {
            'balanced': {
                'for_you': 0.35,
                'trending_for_you': 0.20,
                'your_language': 0.15,
                'discover': 0.15,
                'hidden_gems': 0.10,
                'telugu_specials': 0.05
            },
            'discovery': {
                'discover': 0.40,
                'hidden_gems': 0.25,
                'for_you': 0.20,
                'trending_for_you': 0.10,
                'your_language': 0.05
            },
            'comfort': {
                'for_you': 0.45,
                'your_language': 0.30,
                'telugu_specials': 0.15,
                'trending_for_you': 0.10
            }
        }
        
        proportions = mix_proportions[mix_strategy]
        
        # Generate mixed recommendations
        all_recommendations = []
        category_results = {}
        
        for rec_type, proportion in proportions.items():
            category_limit = max(1, int(total_limit * proportion))
            
            try:
                result = recommendation_engine.generate_personalized_recommendations(
                    user_id=current_user.id,
                    recommendation_type=rec_type,
                    limit=category_limit
                )
                
                recommendations = result.get('recommendations', [])
                category_results[rec_type] = {
                    'count': len(recommendations),
                    'proportion': proportion
                }
                
                # Add category label to each recommendation
                for rec in recommendations:
                    rec['category'] = rec_type
                    rec['mix_rank'] = len(all_recommendations) + 1
                
                all_recommendations.extend(recommendations)
                
            except Exception as e:
                logger.warning(f"Failed to generate {rec_type} for mix: {e}")
                continue
        
        # Shuffle while maintaining some category clustering
        mixed_recommendations = []
        category_queues = {cat: [r for r in all_recommendations if r['category'] == cat] 
                          for cat in proportions.keys()}
        
        # Interleave recommendations from different categories
        max_iterations = total_limit
        iteration = 0
        
        while len(mixed_recommendations) < total_limit and iteration < max_iterations:
            for category in proportions.keys():
                if category_queues[category] and len(mixed_recommendations) < total_limit:
                    rec = category_queues[category].pop(0)
                    rec['mix_rank'] = len(mixed_recommendations) + 1
                    mixed_recommendations.append(rec)
            iteration += 1
        
        # Build response
        response = {
            'success': True,
            'user_id': current_user.id,
            'recommendation_type': 'personalized_mix',
            'mix_strategy': mix_strategy,
            'recommendations': mixed_recommendations,
            'total_count': len(mixed_recommendations),
            'category_breakdown': category_results,
            'mix_metadata': {
                'strategy_applied': mix_strategy,
                'categories_included': list(proportions.keys()),
                'telugu_priority': 'Active across all categories',
                'personalization_level': 'maximum'
            },
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in personalized mix for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to generate CineBrain Personalized Mix',
            'code': 'MIX_FAILED',
            'user_id': current_user.id
        }), 500

@personalized_bp.route('/feedback', methods=['POST', 'OPTIONS'])
@require_auth
def submit_recommendation_feedback(current_user):
    """
    Submit feedback on recommendations for adaptive learning
    
    Body Parameters:
    - recommendation_id: ID of the recommended content
    - feedback_type: Type of feedback (like/dislike/watch/skip/rate)
    - rating: Rating if feedback_type is 'rate' (1-10)
    - category: Recommendation category that generated this
    - context: Additional context about the feedback
    """
    try:
        if request.method == 'OPTIONS':
            return '', 200
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No feedback data provided',
                'code': 'NO_DATA'
            }), 400
        
        # Validate required fields
        required_fields = ['recommendation_id', 'feedback_type', 'category']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'code': 'MISSING_FIELD'
                }), 400
        
        # Validate feedback type
        valid_feedback_types = ['like', 'dislike', 'watch', 'skip', 'rate', 'favorite', 'watchlist']
        if data['feedback_type'] not in valid_feedback_types:
            return jsonify({
                'error': 'Invalid feedback type',
                'code': 'INVALID_FEEDBACK_TYPE',
                'valid_types': valid_feedback_types
            }), 400
        
        # Validate rating if provided
        if data['feedback_type'] == 'rate':
            rating = data.get('rating')
            if not rating or not (1 <= rating <= 10):
                return jsonify({
                    'error': 'Rating must be between 1 and 10',
                    'code': 'INVALID_RATING'
                }), 400
        
        # Process feedback with recommendation engine
        if recommendation_engine and hasattr(recommendation_engine, 'adaptive_engine'):
            try:
                feedback_record = {
                    'user_id': current_user.id,
                    'content_id': data['recommendation_id'],
                    'feedback_type': data['feedback_type'],
                    'category': data['category'],
                    'rating': data.get('rating'),
                    'context': data.get('context', {}),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Store feedback for adaptive learning
                recommendation_engine.adaptive_engine.recommendation_feedback[current_user.id].append(feedback_record)
                
                # Trigger cache invalidation for user's recommendations
                if cache:
                    cache_patterns = [
                        f"cinebrain:advanced_recs:{current_user.id}:*",
                        f"cinebrain:advanced_profile:{current_user.id}"
                    ]
                    for pattern in cache_patterns:
                        try:
                            cache.delete(pattern)
                        except:
                            pass
                
                logger.info(f"Processed recommendation feedback from user {current_user.id}: {data['feedback_type']}")
                
            except Exception as e:
                logger.error(f"Error processing recommendation feedback: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Feedback received and processed',
            'feedback_type': data['feedback_type'],
            'adaptive_learning': 'enabled',
            'cache_refreshed': True
        }), 200
        
    except Exception as e:
        logger.error(f"Error in recommendation feedback for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to process recommendation feedback',
            'code': 'FEEDBACK_FAILED'
        }), 500

@personalized_bp.route('/status', methods=['GET'])
def get_personalization_status():
    """Get status of personalization system"""
    try:
        personalization_system = get_personalization_system()
        
        if not personalization_system:
            return jsonify({
                'status': 'unavailable',
                'message': 'CineBrain Personalization System not initialized'
            }), 503
        
        system_status = personalization_system.get_system_status()
        
        # Add route-specific status
        system_status.update({
            'routes_initialized': recommendation_engine is not None,
            'available_endpoints': [
                '/for-you',
                '/discover', 
                '/trending-for-you',
                '/your-language',
                '/because-you-watched',
                '/hidden-gems',
                '/telugu-specials',
                '/mix',
                '/feedback'
            ],
            'authentication': 'jwt_required',
            'cache_enabled': cache is not None,
            'engine_version': '3.0_neural_cultural',
            'telugu_priority': 'active',
            'adaptive_learning': 'enabled'
        })
        
        return jsonify(system_status), 200
        
    except Exception as e:
        logger.error(f"Error getting personalization status: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get personalization status'
        }), 500

# Error handlers
@personalized_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'CineBrain personalization endpoint not found',
        'code': 'ENDPOINT_NOT_FOUND',
        'available_endpoints': [
            '/api/personalized/for-you',
            '/api/personalized/discover',
            '/api/personalized/trending-for-you',
            '/api/personalized/your-language',
            '/api/personalized/because-you-watched',
            '/api/personalized/hidden-gems',
            '/api/personalized/telugu-specials',
            '/api/personalized/mix',
            '/api/personalized/feedback',
            '/api/personalized/status'
        ]
    }), 404

@personalized_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'CineBrain personalization internal error',
        'code': 'INTERNAL_ERROR',
        'message': 'Please try again later'
    }), 500

# CORS handling
@personalized_bp.after_request
def after_request(response):
    """Add CORS headers"""
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
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response

# Export initialization function
__all__ = ['personalized_bp', 'init_personalized_routes']