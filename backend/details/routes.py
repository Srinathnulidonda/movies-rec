"""
CineBrain Details Routes
RESTful API endpoints for content details
"""

from flask import Blueprint, request, jsonify, current_app
from functools import wraps
import logging
import jwt

from .errors import DetailsError, ValidationError
from .validator import DetailsValidator

logger = logging.getLogger(__name__)
details_bp = Blueprint('details', __name__)

def auth_optional(f):
    """Optional authentication decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            try:
                payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
                user_id = payload.get('user_id')
            except jwt.ExpiredSignatureError:
                pass  # Token expired, continue as anonymous
            except jwt.InvalidTokenError:
                pass  # Invalid token, continue as anonymous
        
        request.user_id = user_id
        return f(*args, **kwargs)
    
    return decorated_function

@details_bp.route('/<slug>', methods=['GET'])
@auth_optional
def get_content_details(slug):
    """Get comprehensive content details by slug"""
    try:
        # Validate slug
        if not DetailsValidator.validate_slug(slug):
            return jsonify({
                'success': False,
                'error': 'Invalid slug format'
            }), 400
        
        # Get query parameters
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        include_trailers = request.args.get('include_trailers', 'true').lower() == 'true'
        include_reviews = request.args.get('include_reviews', 'true').lower() == 'true'
        include_similar = request.args.get('include_similar', 'true').lower() == 'true'
        
        # Get details from service
        details = current_app.details_service.get_comprehensive_details(
            slug=slug,
            user_id=request.user_id,
            force_refresh=force_refresh,
            include_trailers=include_trailers,
            include_reviews=include_reviews,
            include_similar=include_similar
        )
        
        if not details:
            return jsonify({
                'success': False,
                'error': 'Content not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': details,
            'cinebrain_service': 'details'
        }), 200
        
    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except DetailsError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    except Exception as e:
        logger.error(f"Unexpected error in content details: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@details_bp.route('/person/<slug>', methods=['GET'])
@auth_optional
def get_person_details(slug):
    """Get comprehensive person details by slug"""
    try:
        if not DetailsValidator.validate_slug(slug):
            return jsonify({
                'success': False,
                'error': 'Invalid person slug format'
            }), 400
        
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        details = current_app.details_service.get_person_details(
            slug=slug,
            force_refresh=force_refresh
        )
        
        if not details:
            return jsonify({
                'success': False,
                'error': 'Person not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': details,
            'cinebrain_service': 'person_details'
        }), 200
        
    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except DetailsError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    except Exception as e:
        logger.error(f"Unexpected error in person details: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@details_bp.route('/<slug>/trailer', methods=['GET'])
def get_content_trailer(slug):
    """Get trailer for specific content - on-demand fetching"""
    try:
        if not DetailsValidator.validate_slug(slug):
            return jsonify({
                'success': False,
                'error': 'Invalid slug format'
            }), 400
        
        # Force fresh trailer fetch
        trailer = current_app.details_service.get_trailer_for_content(slug)
        
        if not trailer:
            return jsonify({
                'success': False,
                'error': 'Trailer not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': trailer,
            'cinebrain_service': 'trailer'
        }), 200
        
    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except DetailsError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    except Exception as e:
        logger.error(f"Unexpected error fetching trailer: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@details_bp.route('/<slug>/reviews', methods=['GET'])
@auth_optional
def get_content_reviews(slug):
    """Get reviews for specific content"""
    try:
        if not DetailsValidator.validate_slug(slug):
            return jsonify({
                'success': False,
                'error': 'Invalid slug format'
            }), 400
        
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        sort_by = request.args.get('sort_by', 'newest')
        
        if not DetailsValidator.validate_pagination(page, limit):
            return jsonify({
                'success': False,
                'error': 'Invalid pagination parameters'
            }), 400
        
        reviews = current_app.details_service.get_content_reviews(
            slug=slug,
            page=page,
            limit=limit,
            sort_by=sort_by,
            user_id=request.user_id
        )
        
        return jsonify({
            'success': True,
            'data': reviews,
            'cinebrain_service': 'reviews'
        }), 200
        
    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except DetailsError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    except Exception as e:
        logger.error(f"Unexpected error fetching reviews: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@details_bp.route('/<slug>/similar', methods=['GET'])
def get_similar_content(slug):
    """Get similar content recommendations"""
    try:
        if not DetailsValidator.validate_slug(slug):
            return jsonify({
                'success': False,
                'error': 'Invalid slug format'
            }), 400
        
        limit = int(request.args.get('limit', 12))
        algorithm = request.args.get('algorithm', 'genre_based')
        
        if not DetailsValidator.validate_limit(limit, max_limit=24):
            return jsonify({
                'success': False,
                'error': 'Invalid limit parameter'
            }), 400
        
        similar = current_app.details_service.get_similar_content(
            slug=slug,
            limit=limit,
            algorithm=algorithm
        )
        
        return jsonify({
            'success': True,
            'data': similar,
            'cinebrain_service': 'similar_content'
        }), 200
        
    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except DetailsError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    except Exception as e:
        logger.error(f"Unexpected error fetching similar content: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@details_bp.route('/admin/cache/clear', methods=['POST'])
def clear_cache():
    """Admin endpoint to clear details cache"""
    try:
        # Check admin authentication (simplified)
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authentication required'}), 401
        
        cleared_count = current_app.details_service.clear_all_cache()
        
        return jsonify({
            'success': True,
            'message': f'Cleared {cleared_count} cache entries',
            'cinebrain_service': 'cache_management'
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to clear cache'
        }), 500

@details_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for details service"""
    try:
        health_data = current_app.details_service.get_health_status()
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'data': health_data,
            'cinebrain_service': 'details_health'
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e)
        }), 500