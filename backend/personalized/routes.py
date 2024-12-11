# backend/personalized/routes.py
"""
CineBrain Personalization API Routes
Flask blueprint for advanced recommendation endpoints
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any
import jwt
from . import get_recommendation_engine
from .metrics import PerformanceTracker
from .feedback import FeedbackProcessor
from .utils import CacheManager

logger = logging.getLogger(__name__)

personalized_bp = Blueprint('personalized', __name__, url_prefix='/api/personalized')

def auth_required(f):
    """Custom auth decorator matching the main app's auth system"""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'CineBrain authentication required'}), 401
        
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user_id = payload.get('user_id')
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'CineBrain token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid CineBrain token'}), 401
    
    return decorated_function

def get_user_id():
    """Get user ID from token, return None if no valid token"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    
    token = auth_header.split(' ')[1]
    try:
        payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload.get('user_id')
    except:
        return None

@personalized_bp.route('/recommendations', methods=['GET'])
def get_recommendations():
    """
    Get personalized recommendations for authenticated user
    
    Query Parameters:
    - categories: Comma-separated list of recommendation categories
    - limit: Number of recommendations per category (default: 20)
    - language_priority: Apply Telugu-first language prioritization (default: true)
    - include_reasons: Include detailed recommendation reasons (default: false)
    - diversity_factor: Control recommendation diversity 0.0-1.0 (default: 0.3)
    """
    try:
        user_id = get_user_id()
        
        # Allow anonymous users with fallback recommendations
        if not user_id:
            logger.info("Anonymous user requesting personalized recommendations")
            return get_anonymous_recommendations_fallback()
        
        # Parse request parameters
        categories = request.args.get('categories')
        if categories:
            categories = [cat.strip() for cat in categories.split(',')]
        
        limit = min(int(request.args.get('limit', 20)), 50)
        language_priority = request.args.get('language_priority', 'true').lower() == 'true'
        include_reasons = request.args.get('include_reasons', 'false').lower() == 'true'
        diversity_factor = float(request.args.get('diversity_factor', 0.3))
        
        # Get recommendation engine
        engine = get_recommendation_engine()
        if not engine:
            return jsonify({
                'error': 'CineBrain recommendation engine not available',
                'fallback': True
            }), 503
        
        # Generate personalized recommendations
        recommendations = engine.generate_personalized_recommendations(
            user_id=user_id,
            categories=categories,
            limit=limit,
            language_priority=language_priority,
            include_reasons=include_reasons,
            diversity_factor=diversity_factor
        )
        
        # Track metrics
        tracker = PerformanceTracker()
        tracker.log_recommendation_served(
            user_id=user_id,
            categories=categories or ['default'],
            recommendation_count=sum(len(recs) for recs in recommendations.get('recommendations', {}).values())
        )
        
        # Add request metadata
        recommendations['request_metadata'] = {
            'user_id': user_id,
            'categories_requested': categories,
            'limit_per_category': limit,
            'language_priority_applied': language_priority,
            'diversity_factor': diversity_factor,
            'generated_at': datetime.utcnow().isoformat(),
            'engine_version': '2.0.0',
            'platform': 'cinebrain'
        }
        
        return jsonify(recommendations), 200
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return jsonify({
            'error': 'Failed to generate CineBrain recommendations',
            'message': str(e)
        }), 500

def get_anonymous_recommendations_fallback():
    """Fallback recommendations for anonymous users"""
    try:
        # Import here to avoid circular imports
        from services.algorithms import RecommendationOrchestrator
        
        orchestrator = RecommendationOrchestrator()
        
        # Get some basic trending content
        recommendations = {
            'trending_movies': [
                {
                    'id': 1,
                    'title': 'RRR',
                    'content_type': 'movie',
                    'genres': ['Action', 'Drama'],
                    'languages': ['Telugu', 'Hindi'],
                    'rating': 8.5,
                    'poster_path': 'https://image.tmdb.org/t/p/w300/placeholder.jpg',
                    'recommendation_reason': 'Popular Telugu blockbuster',
                    'recommendation_score': 0.9
                },
                {
                    'id': 2,
                    'title': 'Baahubali 2',
                    'content_type': 'movie',
                    'genres': ['Action', 'Adventure'],
                    'languages': ['Telugu', 'Tamil'],
                    'rating': 8.7,
                    'poster_path': 'https://image.tmdb.org/t/p/w300/placeholder.jpg',
                    'recommendation_reason': 'Epic Telugu cinema',
                    'recommendation_score': 0.95
                }
            ],
            'popular_content': [
                {
                    'id': 3,
                    'title': 'Pushpa',
                    'content_type': 'movie',
                    'genres': ['Action', 'Crime'],
                    'languages': ['Telugu'],
                    'rating': 7.6,
                    'poster_path': 'https://image.tmdb.org/t/p/w300/placeholder.jpg',
                    'recommendation_reason': 'Trending Telugu film',
                    'recommendation_score': 0.8
                }
            ]
        }
        
        return jsonify({
            'user_id': None,
            'recommendations': recommendations,
            'profile_insights': {
                'status': 'anonymous_user',
                'message': 'Sign in to get personalized CineBrain recommendations!'
            },
            'recommendation_metadata': {
                'type': 'anonymous_fallback',
                'algorithm_version': '2.0.0',
                'language_priority_applied': True,
                'generated_at': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating anonymous recommendations: {e}")
        return jsonify({
            'error': 'Failed to generate recommendations',
            'recommendations': {},
            'message': 'Please try again later'
        }), 500

@personalized_bp.route('/feedback', methods=['POST'])
@auth_required
def record_feedback():
    """
    Record user feedback for real-time learning
    
    Expected JSON payload:
    {
        "content_id": 123,
        "feedback_type": "like|dislike|view|skip|share|rate",
        "feedback_value": 8.5,  // For ratings
        "context": {
            "recommendation_category": "cinebrain_for_you",
            "position_in_list": 3,
            "viewing_time": 120,  // seconds
            "device_type": "mobile"
        }
    }
    """
    try:
        user_id = request.user_id
        feedback_data = request.get_json()
        
        if not feedback_data:
            return jsonify({'error': 'No feedback data provided'}), 400
        
        required_fields = ['content_id', 'feedback_type']
        if not all(field in feedback_data for field in required_fields):
            return jsonify({'error': 'Missing required feedback fields'}), 400
        
        # Process feedback through feedback processor
        processor = FeedbackProcessor()
        result = processor.process_feedback(
            user_id=user_id,
            content_id=feedback_data['content_id'],
            feedback_type=feedback_data['feedback_type'],
            feedback_value=feedback_data.get('feedback_value'),
            context=feedback_data.get('context', {})
        )
        
        if result['success']:
            # Update user profile in real-time
            engine = get_recommendation_engine()
            if engine:
                engine.update_user_preferences_realtime(user_id, feedback_data)
            
            # Track feedback metrics
            tracker = PerformanceTracker()
            tracker.log_feedback_received(
                user_id=user_id,
                feedback_type=feedback_data['feedback_type'],
                content_id=feedback_data['content_id']
            )
            
            return jsonify({
                'success': True,
                'message': 'CineBrain feedback processed successfully',
                'learning_impact': result.get('learning_impact', 'low'),
                'profile_updated': True
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Failed to process feedback')
            }), 400
            
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({
            'error': 'Failed to process CineBrain feedback',
            'message': str(e)
        }), 500

@personalized_bp.route('/profile', methods=['GET'])
@auth_required
def get_user_profile():
    """Get comprehensive user profile and insights"""
    try:
        user_id = request.user_id
        
        engine = get_recommendation_engine()
        if not engine:
            return jsonify({'error': 'CineBrain engine not available'}), 503
        
        # Get user profile
        profile = engine.profile_analyzer.build_comprehensive_user_profile(user_id)
        
        if not profile:
            return jsonify({
                'user_id': user_id,
                'profile_status': 'building',
                'message': 'Start interacting with content to build your CineBrain profile'
            }), 200
        
        # Get recommendation metrics
        metrics = engine.get_user_recommendation_metrics(user_id)
        
        return jsonify({
            'user_id': user_id,
            'profile': profile,
            'metrics': metrics,
            'cinebrain_insights': {
                'profile_completeness': profile.get('profile_completeness', 0) * 100,
                'recommendation_accuracy': metrics.get('recommendation_accuracy', 0),
                'engagement_level': metrics.get('cinebrain_insights', {}).get('user_type', 'new_user'),
                'cinematic_sophistication': profile.get('cinematic_dna', {}).get('cinematic_sophistication', 0.5) * 100,
                'dominant_themes': list(profile.get('cinematic_dna', {}).get('dominant_themes', {}).keys())[:3],
                'language_priority': profile.get('language_preferences', {}).get('preferred_languages', [])[:3]
            },
            'generated_at': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return jsonify({
            'error': 'Failed to get CineBrain user profile',
            'message': str(e)
        }), 500

@personalized_bp.route('/similar/<int:content_id>', methods=['GET'])
def get_similar_content(content_id):
    """Get ultra-similar content using cinematic DNA analysis"""
    try:
        user_id = get_user_id()  # Optional authentication
        
        # Parse parameters
        limit = min(int(request.args.get('limit', 15)), 30)
        strict_mode = request.args.get('strict_mode', 'true').lower() == 'true'
        min_similarity = float(request.args.get('min_similarity', 0.5))
        include_explanations = request.args.get('include_explanations', 'true').lower() == 'true'
        
        engine = get_recommendation_engine()
        if not engine:
            return jsonify({'error': 'CineBrain engine not available'}), 503
        
        # Get similar content using ultra-powerful similarity engine
        similar_content = engine.get_ultra_similar_content(
            base_content_id=content_id,
            limit=limit,
            strict_mode=strict_mode,
            min_similarity=min_similarity,
            include_explanations=include_explanations
        )
        
        # Track similarity request if user is authenticated
        if user_id:
            tracker = PerformanceTracker()
            tracker.log_similarity_request(user_id, content_id, len(similar_content))
        
        return jsonify({
            'base_content_id': content_id,
            'similar_content': similar_content,
            'metadata': {
                'algorithm': 'cinematic_dna_ultra_similarity',
                'total_matches': len(similar_content),
                'min_similarity_threshold': min_similarity,
                'strict_mode_applied': strict_mode,
                'generated_at': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting similar content: {e}")
        return jsonify({
            'error': 'Failed to get similar content',
            'message': str(e)
        }), 500

@personalized_bp.route('/trending-for-you', methods=['GET'])
def get_trending_for_you():
    """Get personalized trending content"""
    try:
        user_id = get_user_id()
        limit = min(int(request.args.get('limit', 25)), 50)
        
        engine = get_recommendation_engine()
        if not engine:
            return jsonify({'error': 'CineBrain engine not available'}), 503
        
        if user_id:
            trending_recs = engine.generate_category_recommendations(
                user_id=user_id,
                category='trending_for_you',
                limit=limit
            )
        else:
            # Anonymous trending recommendations
            trending_recs = get_anonymous_trending()
        
        return jsonify({
            'category': 'trending_for_you',
            'recommendations': trending_recs,
            'total_count': len(trending_recs),
            'explanation': 'Trending CineBrain content personalized for your taste',
            'algorithm': 'hybrid_trending_personalization',
            'generated_at': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting trending for you: {e}")
        return jsonify({
            'error': 'Failed to get trending recommendations',
            'message': str(e)
        }), 500

def get_anonymous_trending():
    """Get trending content for anonymous users"""
    return [
        {
            'id': 1,
            'title': 'Popular Telugu Movie',
            'content_type': 'movie',
            'genres': ['Action', 'Drama'],
            'languages': ['Telugu'],
            'rating': 8.0,
            'poster_path': 'https://image.tmdb.org/t/p/w300/placeholder.jpg',
            'recommendation_reason': 'Trending now'
        }
    ]

@personalized_bp.route('/metrics', methods=['GET'])
@auth_required
def get_recommendation_metrics():
    """Get user's recommendation performance metrics"""
    try:
        user_id = request.user_id
        
        engine = get_recommendation_engine()
        if not engine:
            return jsonify({'error': 'CineBrain engine not available'}), 503
        
        metrics = engine.get_user_recommendation_metrics(user_id)
        
        # Add system-wide metrics for comparison
        tracker = PerformanceTracker()
        system_metrics = tracker.get_system_performance_summary()
        
        return jsonify({
            'user_metrics': metrics,
            'system_benchmarks': system_metrics,
            'performance_comparison': {
                'user_accuracy_vs_avg': metrics.get('recommendation_accuracy', 0) - system_metrics.get('avg_accuracy', 0),
                'user_engagement_vs_avg': metrics.get('engagement_score', 0) - system_metrics.get('avg_engagement', 0),
                'user_tier': 'above_average' if metrics.get('recommendation_accuracy', 0) > system_metrics.get('avg_accuracy', 0) else 'average'
            },
            'generated_at': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting recommendation metrics: {e}")
        return jsonify({
            'error': 'Failed to get recommendation metrics',
            'message': str(e)
        }), 500

@personalized_bp.route('/health', methods=['GET'])
def health_check():
    """Health check for personalization service"""
    try:
        engine = get_recommendation_engine()
        
        health_info = {
            'status': 'healthy',
            'service': 'cinebrain_personalization_v2',
            'timestamp': datetime.utcnow().isoformat(),
            'engine_available': engine is not None,
            'features': {
                'cinematic_dna_analysis': True,
                'telugu_first_prioritization': True,
                'real_time_learning': True,
                'ultra_similarity_engine': True,
                'hybrid_recommendation': True,
                'behavioral_analytics': True,
                'performance_tracking': True,
                'anonymous_support': True
            }
        }
        
        if engine:
            # Test basic functionality
            try:
                test_metrics = engine.get_system_health_metrics()
                health_info['engine_metrics'] = test_metrics
                health_info['last_update'] = test_metrics.get('last_update')
            except Exception as e:
                health_info['engine_warning'] = str(e)
                health_info['status'] = 'degraded'
        else:
            health_info['status'] = 'degraded'
            health_info['error'] = 'Recommendation engine not initialized'
        
        status_code = 200 if health_info['status'] == 'healthy' else 503
        return jsonify(health_info), status_code
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'cinebrain_personalization_v2',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# Error handlers
@personalized_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'CineBrain personalization endpoint not found',
        'available_endpoints': [
            '/recommendations',
            '/feedback', 
            '/profile',
            '/similar/<content_id>',
            '/trending-for-you',
            '/metrics',
            '/health'
        ]
    }), 404

@personalized_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'CineBrain personalization service error',
        'message': 'Internal server error occurred'
    }), 500