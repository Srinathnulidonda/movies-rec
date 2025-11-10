# reviews/routes.py
import logging
import jwt
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app


logger = logging.getLogger(__name__)

reviews_bp = Blueprint('reviews', __name__)

def get_current_user_id():
    """Extract user ID from JWT token"""
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
        
        token = auth_header.split(' ')[1]
        payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload.get('user_id')
    except:
        return None

def require_auth():
    """Check if user is authenticated"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401
    return user_id

def require_admin():
    """Check if user is admin"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401
    
    User = current_app.review_service.User
    user = User.query.get(user_id)
    if not user or not user.is_admin:
        return jsonify({'success': False, 'error': 'Admin access required'}), 403
    
    return user_id

# ============================================================================
# CONTENT REVIEW ROUTES
# ============================================================================

@reviews_bp.route('/api/details/<slug>/reviews', methods=['GET'])
def get_content_reviews_route(slug):
    """Get reviews for content"""
    try:
        page = int(request.args.get('page', 1))
        limit = min(int(request.args.get('limit', 10)), 50)  # Max 50 per page
        sort_by = request.args.get('sort_by', 'newest')
        user_id = get_current_user_id()
        
        service = current_app.review_service
        result = service.get_content_reviews(slug, page, limit, sort_by, user_id)
        
        return jsonify(result), 200 if result['success'] else 404
        
    except Exception as e:
        logger.error(f"Error in get_content_reviews_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to get reviews'}), 500

@reviews_bp.route('/api/details/<slug>/reviews', methods=['POST'])
def submit_review_route(slug):
    """Submit a review"""
    try:
        user_id = require_auth()
        if isinstance(user_id, tuple):  # Error response
            return user_id
        
        review_data = request.get_json()
        if not review_data:
            return jsonify({'success': False, 'error': 'Review data required'}), 400
        
        service = current_app.review_service
        result = service.submit_review(slug, user_id, review_data)
        
        status_code = 201 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in submit_review_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to submit review'}), 500

@reviews_bp.route('/api/details/<slug>/rating', methods=['POST'])
def add_rating_route(slug):
    """Add a quick rating (without review text)"""
    try:
        user_id = require_auth()
        if isinstance(user_id, tuple):  # Error response
            return user_id
        
        rating_data = request.get_json()
        if not rating_data or 'rating' not in rating_data:
            return jsonify({'success': False, 'error': 'Rating required'}), 400
        
        rating = float(rating_data['rating'])
        service = current_app.review_service
        result = service.add_rating(slug, user_id, rating)
        
        status_code = 201 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in add_rating_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to add rating'}), 500

@reviews_bp.route('/api/details/<slug>/rating', methods=['GET'])
def get_user_rating_route(slug):
    """Get user's rating for content"""
    try:
        user_id = require_auth()
        if isinstance(user_id, tuple):  # Error response
            return user_id
        
        service = current_app.review_service
        result = service.get_user_rating(slug, user_id)
        
        status_code = 200 if result['success'] else 404
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in get_user_rating_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to get rating'}), 500

# ============================================================================
# REVIEW MANAGEMENT ROUTES
# ============================================================================

@reviews_bp.route('/api/reviews/<int:review_id>', methods=['PUT'])
def update_review_route(review_id):
    """Update a review"""
    try:
        user_id = require_auth()
        if isinstance(user_id, tuple):  # Error response
            return user_id
        
        review_data = request.get_json()
        if not review_data:
            return jsonify({'success': False, 'error': 'Review data required'}), 400
        
        service = current_app.review_service
        result = service.update_review(review_id, user_id, review_data)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in update_review_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to update review'}), 500

@reviews_bp.route('/api/reviews/<int:review_id>', methods=['DELETE'])
def delete_review_route(review_id):
    """Delete a review"""
    try:
        user_id = require_auth()
        if isinstance(user_id, tuple):  # Error response
            return user_id
        
        service = current_app.review_service
        result = service.delete_review(review_id, user_id)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in delete_review_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to delete review'}), 500

@reviews_bp.route('/api/reviews/<int:review_id>/helpful', methods=['POST'])
def vote_helpful_route(review_id):
    """Vote on review helpfulness"""
    try:
        user_id = require_auth()
        if isinstance(user_id, tuple):  # Error response
            return user_id
        
        vote_data = request.get_json() or {}
        is_helpful = vote_data.get('is_helpful', True)
        
        service = current_app.review_service
        result = service.vote_helpful(review_id, user_id, is_helpful)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in vote_helpful_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to vote on review'}), 500

@reviews_bp.route('/api/reviews/<int:review_id>/flag', methods=['POST'])
def flag_review_route(review_id):
    """Flag a review for moderation"""
    try:
        user_id = require_auth()
        if isinstance(user_id, tuple):  # Error response
            return user_id
        
        flag_data = request.get_json() or {}
        reason = flag_data.get('reason', 'Inappropriate content')
        
        moderation_service = current_app.review_moderation_service
        result = moderation_service.flag_review(review_id, user_id, reason)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in flag_review_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to flag review'}), 500

# ============================================================================
# USER REVIEW ROUTES
# ============================================================================

@reviews_bp.route('/api/user/reviews', methods=['GET'])
def get_user_reviews_route():
    """Get user's reviews"""
    try:
        user_id = require_auth()
        if isinstance(user_id, tuple):  # Error response
            return user_id
        
        page = int(request.args.get('page', 1))
        limit = min(int(request.args.get('limit', 10)), 50)
        include_drafts = request.args.get('include_drafts', 'false').lower() == 'true'
        
        service = current_app.review_service
        result = service.get_user_reviews(user_id, page, limit, include_drafts)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in get_user_reviews_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to get user reviews'}), 500

@reviews_bp.route('/api/user/ratings', methods=['GET'])
def get_user_ratings_route():
    """Get user's ratings (from user module for compatibility)"""
    try:
        user_id = require_auth()
        if isinstance(user_id, tuple):  # Error response
            return user_id
        
        # This maintains compatibility with the user module
        # Could redirect to user module or implement here
        service = current_app.review_service
        result = service.get_user_reviews(user_id, include_drafts=True)
        
        # Filter to only show ratings
        if result['success']:
            ratings = []
            for review_data in result['reviews']:
                if review_data.get('is_rating_only', False) or not review_data.get('review_text', '').strip():
                    ratings.append({
                        'id': review_data['content']['id'],
                        'slug': review_data['content']['slug'],
                        'title': review_data['content']['title'],
                        'content_type': review_data['content']['content_type'],
                        'poster_path': review_data['content']['poster_url'],
                        'user_rating': review_data['rating'],
                        'rated_at': review_data['created_at']
                    })
            
            # Calculate rating stats
            rating_values = [r['user_rating'] for r in ratings]
            stats = {
                'total_ratings': len(rating_values),
                'average_rating': round(sum(rating_values) / len(rating_values), 1) if rating_values else 0,
                'highest_rating': max(rating_values) if rating_values else 0,
                'lowest_rating': min(rating_values) if rating_values else 0
            }
            
            result = {
                'ratings': ratings,
                'stats': stats,
                'last_updated': result.get('reviews', [{}])[0].get('created_at') if result.get('reviews') else None
            }
        
        return jsonify(result), 200 if result else 500
        
    except Exception as e:
        logger.error(f"Error in get_user_ratings_route: {e}")
        return jsonify({'error': 'Failed to get user ratings'}), 500

# ============================================================================
# ADMIN MODERATION ROUTES
# ============================================================================

@reviews_bp.route('/api/admin/reviews', methods=['GET'])
def get_admin_reviews_route():
    """Get reviews for admin moderation"""
    try:
        admin_id = require_admin()
        if isinstance(admin_id, tuple):  # Error response
            return admin_id
        
        page = int(request.args.get('page', 1))
        limit = min(int(request.args.get('limit', 20)), 100)
        status = request.args.get('status', 'all')
        sort_by = request.args.get('sort_by', 'newest')
        
        moderation_service = current_app.review_moderation_service
        result = moderation_service.get_admin_reviews(page, limit, status, sort_by)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in get_admin_reviews_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to get admin reviews'}), 500

@reviews_bp.route('/api/admin/reviews/<int:review_id>/moderate', methods=['POST'])
def moderate_review_route(review_id):
    """Moderate a review"""
    try:
        admin_id = require_admin()
        if isinstance(admin_id, tuple):  # Error response
            return admin_id
        
        mod_data = request.get_json()
        if not mod_data or 'action' not in mod_data:
            return jsonify({'success': False, 'error': 'Action required'}), 400
        
        action = mod_data['action']
        reason = mod_data.get('reason')
        
        moderation_service = current_app.review_moderation_service
        result = moderation_service.moderate_review(review_id, action, admin_id, reason)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in moderate_review_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to moderate review'}), 500

@reviews_bp.route('/api/admin/reviews/bulk-moderate', methods=['POST'])
def bulk_moderate_reviews_route():
    """Bulk moderate reviews"""
    try:
        admin_id = require_admin()
        if isinstance(admin_id, tuple):  # Error response
            return admin_id
        
        bulk_data = request.get_json()
        if not bulk_data or 'review_ids' not in bulk_data or 'action' not in bulk_data:
            return jsonify({'success': False, 'error': 'Review IDs and action required'}), 400
        
        review_ids = bulk_data['review_ids']
        action = bulk_data['action']
        
        if not isinstance(review_ids, list) or len(review_ids) == 0:
            return jsonify({'success': False, 'error': 'Valid review IDs required'}), 400
        
        moderation_service = current_app.review_moderation_service
        result = moderation_service.bulk_moderate_reviews(review_ids, action, admin_id)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in bulk_moderate_reviews_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to bulk moderate reviews'}), 500

@reviews_bp.route('/api/admin/reviews/flagged', methods=['GET'])
def get_flagged_reviews_route():
    """Get flagged reviews for admin review"""
    try:
        admin_id = require_admin()
        if isinstance(admin_id, tuple):  # Error response
            return admin_id
        
        page = int(request.args.get('page', 1))
        limit = min(int(request.args.get('limit', 20)), 100)
        
        moderation_service = current_app.review_moderation_service
        result = moderation_service.get_flagged_reviews(page, limit)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in get_flagged_reviews_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to get flagged reviews'}), 500

# ============================================================================
# ANALYTICS ROUTES
# ============================================================================

@reviews_bp.route('/api/admin/reviews/stats', methods=['GET'])
def get_review_stats_route():
    """Get comprehensive review statistics"""
    try:
        admin_id = require_admin()
        if isinstance(admin_id, tuple):  # Error response
            return admin_id
        
        analytics_service = current_app.review_analytics_service
        result = analytics_service.get_admin_review_stats()
        
        return jsonify({
            'success': True,
            'stats': result
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_review_stats_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to get review stats'}), 500

@reviews_bp.route('/api/admin/reviews/top-reviewers', methods=['GET'])
def get_top_reviewers_route():
    """Get top reviewers"""
    try:
        admin_id = require_admin()
        if isinstance(admin_id, tuple):  # Error response
            return admin_id
        
        limit = min(int(request.args.get('limit', 10)), 50)
        period_days = request.args.get('period_days')
        if period_days:
            period_days = int(period_days)
        
        analytics_service = current_app.review_analytics_service
        result = analytics_service.get_top_reviewers(limit, period_days)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in get_top_reviewers_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to get top reviewers'}), 500

@reviews_bp.route('/api/details/<slug>/reviews/trends', methods=['GET'])
def get_content_rating_trends_route(slug):
    """Get rating trends for content"""
    try:
        # Get content ID from slug
        Content = current_app.review_service.Content
        content = Content.query.filter_by(slug=slug).first()
        if not content:
            return jsonify({'success': False, 'error': 'Content not found'}), 404
        
        days = int(request.args.get('days', 30))
        
        analytics_service = current_app.review_analytics_service
        result = analytics_service.get_content_rating_trends(content.id, days)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in get_content_rating_trends_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to get rating trends'}), 500

@reviews_bp.route('/api/details/<slug>/reviews/sentiment', methods=['GET'])
def get_content_sentiment_route(slug):
    """Get sentiment analysis for content reviews"""
    try:
        # Get content ID from slug
        Content = current_app.review_service.Content
        content = Content.query.filter_by(slug=slug).first()
        if not content:
            return jsonify({'success': False, 'error': 'Content not found'}), 404
        
        analytics_service = current_app.review_analytics_service
        result = analytics_service.get_review_sentiment_analysis(content.id)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in get_content_sentiment_route: {e}")
        return jsonify({'success': False, 'error': 'Failed to get sentiment analysis'}), 500

# ============================================================================
# HEALTH CHECK ROUTE
# ============================================================================

@reviews_bp.route('/api/reviews/health', methods=['GET'])
def reviews_health():
    """Health check for reviews system"""
    try:
        # Test database connection
        Review = current_app.review_service.Review
        review_count = Review.query.count()
        
        health_info = {
            'status': 'healthy',
            'service': 'cinebrain_reviews',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'total_reviews': review_count,
            'components': {
                'review_service': 'active',
                'moderation_service': 'active',
                'analytics_service': 'active'
            }
        }
        
        return jsonify(health_info), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'cinebrain_reviews',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500