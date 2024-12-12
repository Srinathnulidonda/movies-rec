# backend/personalized/routes.py

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import json
import logging
from typing import Dict, List, Any
import jwt
from functools import wraps

from . import (
    get_profile_analyzer,
    get_embedding_manager,
    get_similarity_engine,
    get_cache_manager
)
from .recommendation_engine import HybridRecommendationEngine
from .utils import TeluguPriorityManager, safe_json_loads

personalized_bp = Blueprint('personalized', __name__, url_prefix='/api/personalized')

logger = logging.getLogger(__name__)

_recommendation_engine = None
_models = None
_db = None
_cache_backend = None

def init_personalized_routes(models, db, cache_backend):
    global _models, _db, _cache_backend
    _models = models
    _db = db
    _cache_backend = cache_backend
    logger.info("✅ CineBrain personalized routes initialized with models and services")

def get_recommendation_engine():
    global _recommendation_engine, _models, _db, _cache_backend
    
    if _recommendation_engine is None:
        if not _models or not _db:
            from flask import current_app
            from app import User, Content, UserInteraction
            _models = {
                'User': User,
                'Content': Content,
                'UserInteraction': UserInteraction
            }
            _db = current_app.extensions.get('sqlalchemy').db
        
        cache_manager = get_cache_manager()
        if not cache_manager.cache_backend and _cache_backend:
            cache_manager.cache_backend = _cache_backend
        
        _recommendation_engine = HybridRecommendationEngine(
            db=_db,
            models=_models,
            cache_manager=cache_manager
        )
        
        logger.info("🚀 CineBrain recommendation engine initialized")
    
    return _recommendation_engine

def auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                'status': 'error',
                'message': 'Authentication required',
                'error_code': 'AUTH_REQUIRED'
            }), 401
        
        token = auth_header.split(' ')[1]
        
        try:
            payload = jwt.decode(
                token,
                current_app.config['SECRET_KEY'],
                algorithms=['HS256']
            )
            request.user_id = payload.get('user_id')
            return f(*args, **kwargs)
            
        except jwt.ExpiredSignatureError:
            return jsonify({
                'status': 'error',
                'message': 'Token expired',
                'error_code': 'TOKEN_EXPIRED'
            }), 401
        except jwt.InvalidTokenError:
            return jsonify({
                'status': 'error',
                'message': 'Invalid token',
                'error_code': 'INVALID_TOKEN'
            }), 401
    
    return decorated_function

@personalized_bp.route('/recommendations', methods=['GET'])
@auth_required
def get_recommendations():
    try:
        user_id = request.user_id
        
        limit = min(int(request.args.get('limit', 20)), 100)
        page = int(request.args.get('page', 1))
        
        context = {}
        context_param = request.args.get('context')
        if context_param:
            try:
                context = json.loads(context_param)
            except json.JSONDecodeError:
                pass
        
        context.update({
            'device': request.headers.get('X-Device-Type', 'unknown'),
            'ip': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', ''),
            'timestamp': datetime.utcnow().isoformat()
        })
        
        cache_manager = get_cache_manager()
        cache_key = f"recommendations:{user_id}:{limit}:{page}"
        cached_result = cache_manager.get(cache_key)
        
        if cached_result and not request.args.get('force_refresh'):
            logger.debug(f"Returning cached recommendations for user {user_id}")
            return jsonify(cached_result), 200
        
        engine = get_recommendation_engine()
        result = engine.generate_recommendations(
            user_id=user_id,
            limit=limit,
            context=context
        )
        
        recommendations = result.get('recommendations', [])
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        
        paginated_result = {
            **result,
            'recommendations': recommendations[start_idx:end_idx],
            'pagination': {
                'page': page,
                'limit': limit,
                'total_items': len(recommendations),
                'total_pages': (len(recommendations) + limit - 1) // limit,
                'has_next': end_idx < len(recommendations),
                'has_prev': page > 1
            }
        }
        
        cache_manager.set(cache_key, paginated_result, ttl=300)
        
        return jsonify(paginated_result), 200
        
    except Exception as e:
        logger.error(f"Error getting recommendations for user {request.user_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to generate recommendations',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@personalized_bp.route('/recommendations/feed', methods=['GET'])
@auth_required
def get_recommendation_feed():
    try:
        user_id = request.user_id
        batch_size = min(int(request.args.get('batch_size', 10)), 50)
        session_id = request.args.get('session_id', 'default')
        
        context = {
            'feed_mode': True,
            'session_id': session_id,
            'device': request.headers.get('X-Device-Type', 'unknown'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        engine = get_recommendation_engine()
        result = engine.generate_recommendations(
            user_id=user_id,
            limit=batch_size * 3,
            context=context
        )
        
        recommendations = result.get('recommendations', [])
        
        if len(recommendations) > batch_size:
            import random
            
            top_items = recommendations[:batch_size//2]
            other_items = recommendations[batch_size//2:]
            random.shuffle(other_items)
            
            batch = top_items + other_items[:batch_size - len(top_items)]
        else:
            batch = recommendations[:batch_size]
        
        feed_response = {
            'status': 'success',
            'feed': batch,
            'session': {
                'session_id': session_id,
                'batch_size': batch_size,
                'has_more': len(recommendations) > batch_size,
                'timestamp': datetime.utcnow().isoformat()
            },
            'metadata': result.get('metadata', {})
        }
        
        return jsonify(feed_response), 200
        
    except Exception as e:
        logger.error(f"Error getting recommendation feed: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to generate feed',
            'error': str(e)
        }), 500

@personalized_bp.route('/recommendations/similar/<int:content_id>', methods=['GET'])
def get_similar_content(content_id):
    try:
        limit = min(int(request.args.get('limit', 10)), 50)
        
        cache_manager = get_cache_manager()
        cache_key = f"similar:{content_id}:{limit}"
        cached_result = cache_manager.get(cache_key)
        
        if cached_result:
            return jsonify(cached_result), 200
        
        engine = get_recommendation_engine()
        similar_content = engine.get_similar_content(
            content_id=content_id,
            limit=limit
        )
        
        response = {
            'status': 'success',
            'base_content_id': content_id,
            'similar_content': similar_content,
            'total_results': len(similar_content),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        cache_manager.set(cache_key, response, ttl=3600)
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error getting similar content for {content_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to find similar content',
            'error': str(e)
        }), 500

@personalized_bp.route('/profile', methods=['GET'])
@auth_required
def get_user_profile():
    try:
        user_id = request.user_id
        include_stats = request.args.get('include_stats', 'false').lower() == 'true'
        
        profile_analyzer = get_profile_analyzer()
        
        profile = profile_analyzer.build_user_profile(user_id)
        
        response = {
            'status': 'success',
            'profile': {
                'user_id': user_id,
                'segment': profile.get('user_segment', 'unknown'),
                'preferences': {
                    'languages': profile.get('explicit_preferences', {}).get('preferred_languages', []),
                    'genres': profile.get('implicit_preferences', {}).get('genre_preferences', {}).get('top_genres', []),
                    'content_types': profile.get('implicit_preferences', {}).get('content_type_preferences', {})
                },
                'cinematic_dna': {
                    'themes': profile.get('cinematic_dna', {}).get('themes', {}),
                    'sophistication': profile.get('cinematic_dna', {}).get('sophistication', 0),
                    'telugu_affinity': profile.get('cinematic_dna', {}).get('telugu_content_ratio', 0)
                },
                'engagement': profile.get('engagement_metrics', {}),
                'profile_strength': {
                    'completeness': profile.get('profile_completeness', 0),
                    'confidence': profile.get('confidence_score', 0),
                    'diversity': profile.get('diversity_score', 0)
                },
                'last_updated': profile.get('last_updated')
            }
        }
        
        if include_stats:
            response['profile']['detailed_stats'] = {
                'behavioral_patterns': profile.get('behavioral_patterns', {}),
                'temporal_patterns': profile.get('temporal_patterns', {}),
                'preference_clusters': profile.get('preference_clusters', {}),
                'recommendation_context': profile.get('recommendation_context', {})
            }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error getting profile for user {request.user_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve profile',
            'error': str(e)
        }), 500

@personalized_bp.route('/feedback', methods=['POST'])
@auth_required
def submit_feedback():
    try:
        user_id = request.user_id
        data = request.get_json()
        
        if not data or 'content_id' not in data or 'feedback_type' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: content_id, feedback_type'
            }), 400
        
        content_id = data['content_id']
        feedback_type = data['feedback_type']
        feedback_value = data.get('feedback_value')
        context = data.get('context', {})
        
        valid_types = ['like', 'dislike', 'view', 'skip', 'rating', 'favorite', 'share']
        if feedback_type not in valid_types:
            return jsonify({
                'status': 'error',
                'message': f'Invalid feedback type. Must be one of: {", ".join(valid_types)}'
            }), 400
        
        if feedback_type == 'rating':
            if not feedback_value or not isinstance(feedback_value, (int, float)):
                return jsonify({
                    'status': 'error',
                    'message': 'Rating feedback requires feedback_value (1-10)'
                }), 400
            
            if feedback_value < 1 or feedback_value > 10:
                return jsonify({
                    'status': 'error',
                    'message': 'Rating must be between 1 and 10'
                }), 400
        
        engine = get_recommendation_engine()
        engine.update_user_feedback(
            user_id=user_id,
            content_id=content_id,
            feedback_type=feedback_type,
            feedback_value=feedback_value
        )
        
        logger.info(f"User {user_id} submitted {feedback_type} feedback for content {content_id}")
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback processed successfully',
            'feedback': {
                'content_id': content_id,
                'feedback_type': feedback_type,
                'feedback_value': feedback_value,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to process feedback',
            'error': str(e)
        }), 500

@personalized_bp.route('/preferences', methods=['PUT'])
@auth_required
def update_preferences():
    try:
        user_id = request.user_id
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No preferences provided'
            }), 400
        
        from flask import current_app
        db = current_app.extensions.get('sqlalchemy').db
        
        if _models and 'User' in _models:
            User = _models['User']
        else:
            from app import User
        
        user = User.query.get(user_id)
        if not user:
            return jsonify({
                'status': 'error',
                'message': 'User not found'
            }), 404
        
        if 'languages' in data:
            user.preferred_languages = json.dumps(data['languages'])
        
        if 'genres' in data:
            user.preferred_genres = json.dumps(data['genres'])
        
        db.session.commit()
        
        cache_manager = get_cache_manager()
        cache_key = cache_manager.get_user_cache_key(user_id, "profile")
        cache_manager.delete(cache_key)
        
        profile_analyzer = get_profile_analyzer()
        updated_profile = profile_analyzer.build_user_profile(user_id, force_refresh=True)
        
        return jsonify({
            'status': 'success',
            'message': 'Preferences updated successfully',
            'preferences': {
                'languages': data.get('languages', safe_json_loads(user.preferred_languages)),
                'genres': data.get('genres', safe_json_loads(user.preferred_genres))
            },
            'profile_confidence': updated_profile.get('confidence_score', 0)
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to update preferences',
            'error': str(e)
        }), 500

@personalized_bp.route('/trending', methods=['GET'])
def get_trending_personalized():
    try:
        limit = min(int(request.args.get('limit', 20)), 50)
        category = request.args.get('category', 'all')
        
        user_id = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            try:
                token = auth_header.split(' ')[1]
                payload = jwt.decode(
                    token,
                    current_app.config['SECRET_KEY'],
                    algorithms=['HS256']
                )
                user_id = payload.get('user_id')
            except:
                pass
        
        from flask import current_app
        db = current_app.extensions.get('sqlalchemy').db
        
        if _models and 'Content' in _models:
            Content = _models['Content']
        else:
            from app import Content
        
        query = Content.query.filter(
            Content.is_trending == True,
            Content.title.isnot(None)
        )
        
        if category != 'all':
            query = query.filter(Content.content_type == category)
        
        trending_content = query.order_by(
            Content.popularity.desc()
        ).limit(limit * 2).all()
        
        user_languages = []
        if user_id:
            profile_analyzer = get_profile_analyzer()
            user_profile = profile_analyzer.build_user_profile(user_id)
            user_languages = user_profile.get('explicit_preferences', {}).get('preferred_languages', [])
        
        sorted_content = TeluguPriorityManager.sort_by_language_priority(
            trending_content,
            user_languages
        )[:limit]
        
        response_items = []
        for rank, content in enumerate(sorted_content):
            response_items.append({
                'rank': rank + 1,
                'content': {
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': safe_json_loads(content.genres or '[]'),
                    'languages': safe_json_loads(content.languages or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None
                },
                'trending_score': content.popularity,
                'personalized': user_id is not None
            })
        
        return jsonify({
            'status': 'success',
            'trending': response_items,
            'metadata': {
                'total_results': len(response_items),
                'personalized': user_id is not None,
                'category': category,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting trending content: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get trending content',
            'error': str(e)
        }), 500

@personalized_bp.route('/health', methods=['GET'])
def health_check():
    try:
        health_status = {
            'status': 'healthy',
            'service': 'cinebrain_personalization',
            'version': '3.0.0',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }
        
        try:
            profile_analyzer = get_profile_analyzer()
            health_status['components']['profile_analyzer'] = 'healthy' if profile_analyzer else 'unavailable'
        except:
            health_status['components']['profile_analyzer'] = 'error'
        
        try:
            embedding_manager = get_embedding_manager()
            health_status['components']['embedding_manager'] = 'healthy' if embedding_manager else 'unavailable'
        except:
            health_status['components']['embedding_manager'] = 'error'
        
        try:
            engine = get_recommendation_engine()
            health_status['components']['recommendation_engine'] = 'healthy' if engine else 'unavailable'
        except:
            health_status['components']['recommendation_engine'] = 'error'
        
        try:
            cache_manager = get_cache_manager()
            test_key = 'health_check_test'
            cache_manager.set(test_key, 'ok', ttl=5)
            cache_value = cache_manager.get(test_key)
            health_status['components']['cache'] = 'healthy' if cache_value == 'ok' else 'degraded'
        except:
            health_status['components']['cache'] = 'error'
        
        component_statuses = health_status['components'].values()
        if all(s == 'healthy' for s in component_statuses):
            health_status['status'] = 'healthy'
        elif any(s == 'error' for s in component_statuses):
            health_status['status'] = 'unhealthy'
        else:
            health_status['status'] = 'degraded'
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        
        return jsonify(health_status), status_code
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'cinebrain_personalization',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 503

@personalized_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'error_code': 'NOT_FOUND'
    }), 404

@personalized_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'error_code': 'INTERNAL_ERROR'
    }), 500

@personalized_bp.before_request
def log_request():
    logger.debug(f"Request: {request.method} {request.path} from {request.remote_addr}")

@personalized_bp.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin')
    allowed_origins = [
        'http://localhost:3000',
        'http://localhost:5000',
        'https://cinebrain.vercel.app'
    ]
    
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response