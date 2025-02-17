# user/routes.py
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import logging
import jwt
from sqlalchemy import or_

from .utils import init_user_module, add_cors_headers
from . import avatar
from . import profile
from . import watchlist
from . import favorites
from . import activity
from . import dashboard
from . import settings

# Create the user blueprint
user_bp = Blueprint('user', __name__)

logger = logging.getLogger(__name__)

# Global variables (will be set by init function)
db = None
User = None
app = None

def init_user_routes(flask_app, database, models, services):
    """Initialize the user routes with dependencies"""
    global db, User, app
    
    app = flask_app
    db = database
    User = models['User']
    
    # Initialize the user module
    init_user_module(flask_app, database, models, services)
    
    logger.info("✅ CineBrain user routes initialized successfully")

# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@user_bp.route('/api/register', methods=['POST', 'OPTIONS'])
def register():
    """User registration with welcome email"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        if not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing required fields for CineBrain account'}), 400
        
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'CineBrain username already exists'}), 400
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'CineBrain email already exists'}), 400
        
        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=generate_password_hash(data['password']),
            preferred_languages=json.dumps(data.get('preferred_languages', ['Telugu', 'English'])),
            preferred_genres=json.dumps(data.get('preferred_genres', [])),
            location=data.get('location', ''),
            avatar_url=data.get('avatar_url', '')
        )
        
        db.session.add(user)
        db.session.commit()
        
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.secret_key, algorithm='HS256')
        
        # Send welcome email
        welcome_email_sent = False
        try:
            from auth.user_mail_templates import get_professional_template
            from auth.service import email_service
            
            if email_service and email_service.email_enabled:
                # Prepare user languages for email
                user_languages = json.loads(user.preferred_languages or '["Telugu", "English"]')
                
                html_content, text_content = get_professional_template(
                    'registration',
                    user_name=user.username,
                    user_email=user.email,
                    preferred_languages=user_languages
                )
                
                email_service.queue_email(
                    to=user.email,
                    subject="Welcome to CineBrain! 🎬",
                    html=html_content,
                    text=text_content,
                    priority='normal',
                    to_name=user.username
                )
                
                welcome_email_sent = True
                logger.info(f"✅ Welcome email queued for user {user.username} ({user.email})")
            else:
                logger.warning("⚠️ Email service not available for welcome email")
                
        except Exception as e:
            logger.error(f"❌ Failed to send welcome email to {user.email}: {e}")
            # Don't fail registration if email fails - user experience is priority
        
        from .utils import get_enhanced_user_stats
        stats = get_enhanced_user_stats(user.id)
        
        response_data = {
            'message': 'CineBrain user registered successfully',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]'),
                'location': user.location,
                'avatar_url': user.avatar_url,
                'created_at': user.created_at.isoformat(),
                'stats': stats
            },
            'welcome_email_sent': welcome_email_sent
        }
        
        return jsonify(response_data), 201
        
    except Exception as e:
        logger.error(f"CineBrain registration error: {e}")
        db.session.rollback()
        return jsonify({'error': 'CineBrain registration failed'}), 500

@user_bp.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    """User login"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'error': 'Missing username or password'}), 400
        
        user = None
        
        if '@' in username:
            user = User.query.filter(User.email.ilike(username)).first()
        else:
            user = User.query.filter(User.username.ilike(username)).first()
        
        if not user:
            user = User.query.filter(
                or_(
                    User.username.ilike(username),
                    User.email.ilike(username)
                )
            ).first()
        
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        password_valid = check_password_hash(user.password_hash, password)
        
        if not password_valid:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.secret_key, algorithm='HS256')
        
        from .utils import get_enhanced_user_stats, recommendation_engine
        stats = get_enhanced_user_stats(user.id)
        
        rec_effectiveness = {}
        try:
            if recommendation_engine:
                rec_effectiveness = recommendation_engine.get_user_recommendation_metrics(user.id)
        except Exception as e:
            logger.warning(f"Could not get CineBrain recommendation effectiveness: {e}")
        
        return jsonify({
            'message': 'CineBrain login successful',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]'),
                'location': user.location,
                'avatar_url': user.avatar_url,
                'last_active': user.last_active.isoformat() if user.last_active else None,
                'stats': stats,
                'recommendation_effectiveness': rec_effectiveness
            }
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain Login error: {e}")
        return jsonify({'error': 'CineBrain login failed'}), 500

# ============================================================================
# PROFILE ROUTES
# ============================================================================

@user_bp.route('/api/users/profile', methods=['GET', 'OPTIONS'])
def get_profile():
    return profile.get_user_profile()

@user_bp.route('/api/users/profile', methods=['PUT', 'OPTIONS'])
def update_profile():
    return profile.update_user_profile()

@user_bp.route('/api/users/profile/public/<username>', methods=['GET'])
def get_public_profile_route(username):
    return profile.get_public_profile(username)

@user_bp.route('/api/personalized/update-preferences', methods=['POST', 'OPTIONS'])
def update_preferences():
    return profile.update_user_preferences()

# ============================================================================
# AVATAR ROUTES
# ============================================================================

@user_bp.route('/api/users/avatar/upload', methods=['POST', 'OPTIONS'])
def upload_avatar_route():
    return avatar.upload_avatar()

@user_bp.route('/api/users/avatar/delete', methods=['DELETE', 'OPTIONS'])
def delete_avatar_route():
    return avatar.delete_avatar()

@user_bp.route('/api/users/avatar/url', methods=['GET', 'OPTIONS'])
def get_avatar_url_route():
    return avatar.get_avatar_url()

# ============================================================================
# WATCHLIST ROUTES
# ============================================================================

@user_bp.route('/api/user/watchlist', methods=['GET', 'OPTIONS'])
def get_watchlist_route():
    return watchlist.get_watchlist()

@user_bp.route('/api/user/watchlist', methods=['POST', 'OPTIONS'])
def add_to_watchlist_route():
    return watchlist.add_to_watchlist()

@user_bp.route('/api/user/watchlist/<int:content_id>', methods=['DELETE', 'OPTIONS'])
def remove_from_watchlist_route(content_id):
    from .utils import require_auth
    @require_auth
    def wrapper(current_user):
        return watchlist.remove_from_watchlist(current_user, content_id)
    return wrapper()

@user_bp.route('/api/user/watchlist/<int:content_id>', methods=['GET', 'OPTIONS'])
def check_watchlist_status_route(content_id):
    from .utils import require_auth
    @require_auth
    def wrapper(current_user):
        return watchlist.check_watchlist_status(current_user, content_id)
    return wrapper()

# ============================================================================
# FAVORITES ROUTES
# ============================================================================

@user_bp.route('/api/user/favorites', methods=['GET', 'OPTIONS'])
def get_favorites_route():
    from .utils import require_auth
    
    @require_auth
    def wrapper(current_user):
        if request.method == 'OPTIONS':
            return '', 200
        return favorites.get_favorites(current_user)
    
    return wrapper()

@user_bp.route('/api/user/favorites', methods=['POST', 'OPTIONS'])
def add_to_favorites_route():
    from .utils import require_auth
    
    @require_auth
    def wrapper(current_user):
        if request.method == 'OPTIONS':
            return '', 200
        return favorites.add_to_favorites(current_user)
    
    return wrapper()

@user_bp.route('/api/user/favorites/<int:content_id>', methods=['DELETE', 'OPTIONS'])
def remove_from_favorites_route(content_id):
    from .utils import require_auth
    
    @require_auth
    def wrapper(current_user):
        if request.method == 'OPTIONS':
            return '', 200
        return favorites.remove_from_favorites(current_user, content_id)
    
    return wrapper()

@user_bp.route('/api/user/favorites/<int:content_id>', methods=['GET', 'OPTIONS'])
def check_favorite_status_route(content_id):
    from .utils import require_auth
    
    @require_auth
    def wrapper(current_user):
        if request.method == 'OPTIONS':
            return '', 200
        return favorites.check_favorite_status(current_user, content_id)
    
    return wrapper()

# ============================================================================
# RATINGS ROUTES (Delegated to Reviews Module for Compatibility)
# ============================================================================

@user_bp.route('/api/user/ratings', methods=['GET', 'OPTIONS'])
def get_user_ratings_route():
    """Get user's ratings (delegates to reviews module for compatibility)"""
    try:
        from .utils import require_auth
        
        @require_auth
        def wrapper(current_user):
            if request.method == 'OPTIONS':
                return '', 200
            
            try:
                # Delegate to reviews service for ratings
                if hasattr(app, 'review_service') and app.review_service:
                    result = app.review_service.get_user_reviews(current_user.id, include_drafts=True)
                    
                    if result['success']:
                        # Filter to only show ratings and format for compatibility
                        ratings = []
                        for review_data in result['reviews']:
                            content_data = review_data.get('content', {})
                            ratings.append({
                                'id': content_data.get('id'),
                                'slug': content_data.get('slug'),
                                'title': content_data.get('title'),
                                'content_type': content_data.get('content_type'),
                                'poster_path': content_data.get('poster_url'),
                                'user_rating': review_data.get('rating'),
                                'imdb_rating': None,  # Would need to be fetched separately
                                'rated_at': review_data.get('created_at'),
                                'has_review': bool(review_data.get('review_text', '').strip()),
                                'review_id': review_data.get('id')
                            })
                        
                        # Calculate rating stats
                        rating_values = [r['user_rating'] for r in ratings if r['user_rating'] is not None]
                        stats = {
                            'total_ratings': len(rating_values),
                            'average_rating': round(sum(rating_values) / len(rating_values), 1) if rating_values else 0,
                            'highest_rating': max(rating_values) if rating_values else 0,
                            'lowest_rating': min(rating_values) if rating_values else 0
                        }
                        
                        return jsonify({
                            'ratings': ratings,
                            'stats': stats,
                            'last_updated': datetime.utcnow().isoformat()
                        }), 200
                    else:
                        return jsonify({
                            'ratings': [],
                            'stats': {'total_ratings': 0, 'average_rating': 0, 'highest_rating': 0, 'lowest_rating': 0},
                            'last_updated': datetime.utcnow().isoformat()
                        }), 200
                else:
                    # Fallback if reviews service not available
                    from .utils import UserInteraction, Content
                    
                    rating_interactions = UserInteraction.query.filter_by(
                        user_id=current_user.id,
                        interaction_type='rating'
                    ).filter(UserInteraction.rating.isnot(None)).order_by(
                        UserInteraction.timestamp.desc()
                    ).all()
                    
                    content_ids = [interaction.content_id for interaction in rating_interactions]
                    contents = Content.query.filter(Content.id.in_(content_ids)).all()
                    content_map = {content.id: content for content in contents}
                    
                    ratings = []
                    for interaction in rating_interactions:
                        content = content_map.get(interaction.content_id)
                        if content:
                            ratings.append({
                                'id': content.id,
                                'slug': content.slug,
                                'title': content.title,
                                'content_type': content.content_type,
                                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                                'user_rating': interaction.rating,
                                'imdb_rating': content.rating,
                                'rated_at': interaction.timestamp.isoformat(),
                                'has_review': False,
                                'review_id': None
                            })
                    
                    rating_values = [interaction.rating for interaction in rating_interactions]
                    stats = {
                        'total_ratings': len(rating_values),
                        'average_rating': round(sum(rating_values) / len(rating_values), 1) if rating_values else 0,
                        'highest_rating': max(rating_values) if rating_values else 0,
                        'lowest_rating': min(rating_values) if rating_values else 0
                    }
                    
                    return jsonify({
                        'ratings': ratings,
                        'stats': stats,
                        'last_updated': datetime.utcnow().isoformat()
                    }), 200
                    
            except Exception as e:
                logger.error(f"CineBrain user ratings error: {e}")
                return jsonify({'error': 'Failed to get CineBrain user ratings'}), 500
        
        return wrapper()
        
    except Exception as e:
        logger.error(f"Error in get_user_ratings_route: {e}")
        return jsonify({'error': 'Failed to get user ratings'}), 500

@user_bp.route('/api/user/ratings', methods=['POST', 'OPTIONS'])
def add_rating_route():
    """Add or update a rating (delegates to reviews module)"""
    try:
        from .utils import require_auth
        
        @require_auth
        def wrapper(current_user):
            if request.method == 'OPTIONS':
                return '', 200
            
            try:
                data = request.get_json()
                content_id = data.get('content_id')
                rating = data.get('rating')
                
                if not content_id or rating is None:
                    return jsonify({'error': 'Content ID and rating required'}), 400
                
                if not (1 <= rating <= 10):
                    return jsonify({'error': 'Rating must be between 1 and 10'}), 400
                
                # Get content slug for reviews service
                from .utils import Content
                content = Content.query.get(content_id)
                if not content:
                    return jsonify({'error': 'Content not found'}), 404
                
                # Delegate to reviews service if available
                if hasattr(app, 'review_service') and app.review_service:
                    result = app.review_service.add_rating(content.slug, current_user.id, rating)
                    return jsonify(result), 201 if result['success'] else 400
                else:
                    # Fallback to direct UserInteraction
                    from .utils import UserInteraction
                    
                    # Check if rating already exists
                    existing = UserInteraction.query.filter_by(
                        user_id=current_user.id,
                        content_id=content_id,
                        interaction_type='rating'
                    ).first()
                    
                    if existing:
                        existing.rating = rating
                        existing.timestamp = datetime.utcnow()
                        message = 'Rating updated successfully'
                    else:
                        interaction = UserInteraction(
                            user_id=current_user.id,
                            content_id=content_id,
                            interaction_type='rating',
                            rating=rating
                        )
                        db.session.add(interaction)
                        message = 'Rating added successfully'
                    
                    db.session.commit()
                    
                    return jsonify({
                        'success': True,
                        'message': message,
                        'rating': rating
                    }), 201
                    
            except Exception as e:
                logger.error(f"CineBrain add rating error: {e}")
                db.session.rollback()
                return jsonify({'error': 'Failed to add rating'}), 500
        
        return wrapper()
        
    except Exception as e:
        logger.error(f"Error in add_rating_route: {e}")
        return jsonify({'error': 'Failed to add rating'}), 500

@user_bp.route('/api/user/ratings/<int:content_id>', methods=['DELETE', 'OPTIONS'])
def remove_rating_route(content_id):
    """Remove a rating (delegates to reviews module)"""
    try:
        from .utils import require_auth
        
        @require_auth
        def wrapper(current_user):
            if request.method == 'OPTIONS':
                return '', 200
            
            try:
                # Get content slug for reviews service
                from .utils import Content
                content = Content.query.get(content_id)
                if not content:
                    return jsonify({'error': 'Content not found'}), 404
                
                # Delegate to reviews service if available
                if hasattr(app, 'review_service') and app.review_service:
                    # First check if user has a review/rating
                    result = app.review_service.get_user_rating(content.slug, current_user.id)
                    if result['success'] and result['has_rating']:
                        # Delete the review (which includes the rating)
                        if result['review_id']:
                            delete_result = app.review_service.delete_review(result['review_id'], current_user.id)
                            return jsonify(delete_result), 200 if delete_result['success'] else 400
                    
                    return jsonify({
                        'success': False,
                        'message': 'No rating found for this content'
                    }), 404
                else:
                    # Fallback to direct UserInteraction
                    from .utils import UserInteraction
                    
                    interaction = UserInteraction.query.filter_by(
                        user_id=current_user.id,
                        content_id=content_id,
                        interaction_type='rating'
                    ).first()
                    
                    if interaction:
                        db.session.delete(interaction)
                        db.session.commit()
                        
                        return jsonify({
                            'success': True,
                            'message': 'Rating removed successfully'
                        }), 200
                    else:
                        return jsonify({
                            'success': False,
                            'message': 'No rating found for this content'
                        }), 404
                        
            except Exception as e:
                logger.error(f"Remove rating error: {e}")
                db.session.rollback()
                return jsonify({'error': 'Failed to remove rating'}), 500
        
        return wrapper()
        
    except Exception as e:
        logger.error(f"Error in remove_rating_route: {e}")
        return jsonify({'error': 'Failed to remove rating'}), 500

@user_bp.route('/api/user/ratings/<int:content_id>', methods=['GET', 'OPTIONS'])
def get_rating_for_content_route(content_id):
    """Get user's rating for specific content (delegates to reviews module)"""
    try:
        from .utils import require_auth
        
        @require_auth
        def wrapper(current_user):
            if request.method == 'OPTIONS':
                return '', 200
            
            try:
                # Get content slug for reviews service
                from .utils import Content
                content = Content.query.get(content_id)
                if not content:
                    return jsonify({'error': 'Content not found'}), 404
                
                # Delegate to reviews service if available
                if hasattr(app, 'review_service') and app.review_service:
                    result = app.review_service.get_user_rating(content.slug, current_user.id)
                    if result['success']:
                        return jsonify({
                            'has_rating': result['has_rating'],
                            'rating': result['rating'],
                            'rated_at': result['created_at']
                        }), 200
                    else:
                        return jsonify({
                            'has_rating': False,
                            'rating': None,
                            'rated_at': None
                        }), 200
                else:
                    # Fallback to direct UserInteraction
                    from .utils import UserInteraction
                    
                    interaction = UserInteraction.query.filter_by(
                        user_id=current_user.id,
                        content_id=content_id,
                        interaction_type='rating'
                    ).first()
                    
                    return jsonify({
                        'has_rating': interaction is not None,
                        'rating': interaction.rating if interaction else None,
                        'rated_at': interaction.timestamp.isoformat() if interaction else None
                    }), 200
                    
            except Exception as e:
                logger.error(f"Get rating error: {e}")
                return jsonify({'error': 'Failed to get rating'}), 500
        
        return wrapper()
        
    except Exception as e:
        logger.error(f"Error in get_rating_for_content_route: {e}")
        return jsonify({'error': 'Failed to get rating'}), 500

# ============================================================================
# ACTIVITY ROUTES
# ============================================================================

@user_bp.route('/api/interactions', methods=['POST', 'OPTIONS'])
def record_interaction_route():
    from .utils import require_auth
    
    @require_auth
    def wrapper(current_user):
        if request.method == 'OPTIONS':
            return '', 200
        return activity.record_interaction(current_user)
    
    return wrapper()


@user_bp.route('/api/users/<username>/activity/public', methods=['GET'])
def get_public_activity_route(username):
    return activity.get_public_activity(username)

@user_bp.route('/api/users/<username>/stats/public', methods=['GET'])
def get_public_stats_route(username):
    return activity.get_public_stats(username)

# ============================================================================
# DASHBOARD ROUTES
# ============================================================================

@user_bp.route('/api/users/analytics', methods=['GET', 'OPTIONS'])
def get_analytics_route():
    return dashboard.get_user_analytics()

@user_bp.route('/api/personalized/profile-insights', methods=['GET', 'OPTIONS'])
def get_profile_insights_route():
    return dashboard.get_profile_insights()

# ============================================================================
# PERSONALIZED RECOMMENDATION ROUTES
# ============================================================================

@user_bp.route('/api/personalized/', methods=['GET', 'OPTIONS'])
def get_personalized_recommendations():
    from .utils import require_auth, recommendation_engine
    
    @require_auth
    def wrapper(current_user):
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            if not recommendation_engine:
                return jsonify({
                    'error': 'CineBrain recommendation engine not available',
                    'recommendations': {},
                    'fallback': True
                }), 503
            
            limit = min(int(request.args.get('limit', 50)), 100)
            categories = request.args.get('categories')
            
            if categories:
                category_list = [cat.strip() for cat in categories.split(',')]
            else:
                category_list = None
            
            recommendations = recommendation_engine.get_personalized_recommendations(
                user_id=current_user.id,
                limit=limit,
                categories=category_list
            )
            
            recommendations['platform'] = 'cinebrain'
            recommendations['user_tier'] = 'premium'
            recommendations['personalization_level'] = 'high'
            
            return jsonify({
                'success': True,
                'data': recommendations,
                'message': 'CineBrain personalized recommendations generated successfully',
                'user': {
                    'id': current_user.id,
                    'username': current_user.username
                },
                'metadata': {
                    'generated_at': datetime.utcnow().isoformat(),
                    'algorithm_version': '3.0',
                    'personalization_strength': recommendations.get('recommendation_metadata', {}).get('confidence_score', 0.8)
                }
            }), 200
            
        except Exception as e:
            logger.error(f"CineBrain personalized recommendations error for user {current_user.id}: {e}")
            return jsonify({
                'error': 'Failed to generate CineBrain personalized recommendations',
                'success': False,
                'data': {}
            }), 500
    
    return wrapper()

@user_bp.route('/api/personalized/for-you', methods=['GET', 'OPTIONS'])
def get_for_you_recommendations():
    from .utils import require_auth, recommendation_engine
    
    @require_auth
    def wrapper(current_user):
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            if not recommendation_engine:
                return jsonify({'error': 'CineBrain recommendation engine not available'}), 503
            
            limit = min(int(request.args.get('limit', 30)), 50)
            
            recommendations = recommendation_engine.get_personalized_recommendations(
                user_id=current_user.id,
                limit=limit,
                categories=['for_you']
            )
            
            for_you_recs = recommendations.get('recommendations', {}).get('for_you', [])
            
            return jsonify({
                'success': True,
                'recommendations': for_you_recs,
                'total_count': len(for_you_recs),
                'user_insights': recommendations.get('profile_insights', {}),
                'metadata': recommendations.get('recommendation_metadata', {}),
                'generated_at': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            logger.error(f"CineBrain For You recommendations error: {e}")
            return jsonify({'error': 'Failed to get CineBrain For You recommendations'}), 500
    
    return wrapper()

@user_bp.route('/api/personalized/because-you-watched', methods=['GET', 'OPTIONS'])
def get_because_you_watched():
    from .utils import require_auth, recommendation_engine
    
    @require_auth
    def wrapper(current_user):
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            if not recommendation_engine:
                return jsonify({'error': 'CineBrain recommendation engine not available'}), 503
            
            limit = min(int(request.args.get('limit', 20)), 30)
            
            recommendations = recommendation_engine.get_personalized_recommendations(
                user_id=current_user.id,
                limit=limit,
                categories=['because_you_watched']
            )
            
            because_recs = recommendations.get('recommendations', {}).get('because_you_watched', [])
            
            return jsonify({
                'success': True,
                'recommendations': because_recs,
                'total_count': len(because_recs),
                'explanation': 'Based on your recently watched CineBrain content',
                'algorithm': 'content_similarity_and_collaborative_filtering'
            }), 200
            
        except Exception as e:
            logger.error(f"CineBrain Because you watched recommendations error: {e}")
            return jsonify({'error': 'Failed to get CineBrain because you watched recommendations'}), 500
    
    return wrapper()

@user_bp.route('/api/personalized/trending-for-you', methods=['GET', 'OPTIONS'])
def get_trending_for_you():
    from .utils import require_auth, recommendation_engine
    
    @require_auth
    def wrapper(current_user):
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            if not recommendation_engine:
                return jsonify({'error': 'CineBrain recommendation engine not available'}), 503
            
            limit = min(int(request.args.get('limit', 25)), 40)
            
            recommendations = recommendation_engine.get_personalized_recommendations(
                user_id=current_user.id,
                limit=limit,
                categories=['trending_for_you']
            )
            
            trending_recs = recommendations.get('recommendations', {}).get('trending_for_you', [])
            
            return jsonify({
                'success': True,
                'recommendations': trending_recs,
                'total_count': len(trending_recs),
                'explanation': 'Trending CineBrain content personalized for your taste',
                'algorithm': 'hybrid_trending_personalization'
            }), 200
            
        except Exception as e:
            logger.error(f"CineBrain Trending for you recommendations error: {e}")
            return jsonify({'error': 'Failed to get CineBrain trending recommendations'}), 500
    
    return wrapper()

@user_bp.route('/api/personalized/your-language', methods=['GET', 'OPTIONS'])
def get_your_language_recommendations():
    from .utils import require_auth, recommendation_engine
    
    @require_auth
    def wrapper(current_user):
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            if not recommendation_engine:
                return jsonify({'error': 'CineBrain recommendation engine not available'}), 503
            
            limit = min(int(request.args.get('limit', 25)), 40)
            
            recommendations = recommendation_engine.get_personalized_recommendations(
                user_id=current_user.id,
                limit=limit,
                categories=['your_language']
            )
            
            language_recs = recommendations.get('recommendations', {}).get('your_language', [])
            
            preferred_languages = json.loads(current_user.preferred_languages or '[]')
            
            return jsonify({
                'success': True,
                'recommendations': language_recs,
                'total_count': len(language_recs),
                'explanation': 'CineBrain content in your preferred languages',
                'preferred_languages': preferred_languages,
                'algorithm': 'language_preference_filtering'
            }), 200
            
        except Exception as e:
            logger.error(f"CineBrain Language recommendations error: {e}")
            return jsonify({'error': 'Failed to get CineBrain language recommendations'}), 500
    
    return wrapper()

@user_bp.route('/api/personalized/hidden-gems', methods=['GET', 'OPTIONS'])
def get_hidden_gems():
    from .utils import require_auth, recommendation_engine
    
    @require_auth
    def wrapper(current_user):
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            if not recommendation_engine:
                return jsonify({'error': 'CineBrain recommendation engine not available'}), 503
            
            limit = min(int(request.args.get('limit', 15)), 25)
            
            recommendations = recommendation_engine.get_personalized_recommendations(
                user_id=current_user.id,
                limit=limit,
                categories=['hidden_gems']
            )
            
            gems_recs = recommendations.get('recommendations', {}).get('hidden_gems', [])
            
            return jsonify({
                'success': True,
                'recommendations': gems_recs,
                'total_count': len(gems_recs),
                'explanation': 'High-quality CineBrain content you might have missed',
                'algorithm': 'hidden_gem_discovery'
            }), 200
            
        except Exception as e:
            logger.error(f"CineBrain Hidden gems recommendations error: {e}")
            return jsonify({'error': 'Failed to get CineBrain hidden gems recommendations'}), 500
    
    return wrapper()

# ============================================================================
# SETTINGS ROUTES
# ============================================================================

@user_bp.route('/api/users/settings', methods=['GET', 'OPTIONS'])
def get_settings_route():
    return settings.get_user_settings()

@user_bp.route('/api/users/settings/account', methods=['PUT', 'OPTIONS'])
def update_account_settings_route():
    return settings.update_account_settings()

@user_bp.route('/api/users/settings/password', methods=['PUT', 'OPTIONS'])
def change_password_route():
    return settings.change_password()

@user_bp.route('/api/users/settings/delete-account', methods=['DELETE', 'OPTIONS'])
def delete_account_route():
    return settings.delete_account()

@user_bp.route('/api/users/settings/export-data', methods=['GET', 'OPTIONS'])
def export_data_route():
    return settings.export_user_data()

# ============================================================================
# HEALTH ROUTE
# ============================================================================

@user_bp.route('/api/users/health', methods=['GET'])
def users_health():
    try:
        health_info = {
            'status': 'healthy',
            'service': 'cinebrain_users',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '3.1.0'
        }
        
        try:
            User.query.limit(1).first()
            health_info['database'] = 'connected'
        except Exception as e:
            health_info['database'] = f'error: {str(e)}'
            health_info['status'] = 'degraded'
        
        from .utils import recommendation_engine
        health_info['recommendation_engine'] = 'connected' if recommendation_engine else 'not_available'
        
        # Check reviews service integration
        health_info['reviews_integration'] = 'connected' if hasattr(app, 'review_service') and app.review_service else 'not_available'
        
        # Check email service status
        try:
            from auth.service import email_service
            health_info['email_service'] = {
                'enabled': email_service.email_enabled if email_service else False,
                'smtp_enabled': email_service.smtp_enabled if email_service else False,
                'api_enabled': email_service.api_enabled if email_service else False
            }
        except:
            health_info['email_service'] = {'enabled': False}
        
        try:
            total_users = User.query.count()
            active_users = User.query.filter(
                User.last_active >= datetime.utcnow() - timedelta(days=7)
            ).count()
            users_with_avatars = User.query.filter(User.avatar_url.isnot(None)).count()
            
            health_info['user_metrics'] = {
                'total_users': total_users,
                'active_users_7d': active_users,
                'users_with_avatars': users_with_avatars,
                'activity_rate': round((active_users / total_users * 100), 1) if total_users > 0 else 0,
                'avatar_adoption_rate': round((users_with_avatars / total_users * 100), 1) if total_users > 0 else 0
            }
        except Exception as e:
            health_info['user_metrics'] = {'error': str(e)}
        
        health_info['features'] = {
            'personalized_recommendations': True,
            'user_analytics': True,
            'profile_management': True,
            'watchlist_favorites': True,
            'real_time_updates': True,
            'email_username_login': True,
            'avatar_service': True,
            'modular_architecture': True,
            'welcome_emails': True,
            'reviews_integration': hasattr(app, 'review_service') and bool(app.review_service),
            'ratings_delegation': True
        }
        
        return jsonify(health_info), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'cinebrain_users',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# Add CORS headers to all responses
@user_bp.after_request
def after_request(response):
    return add_cors_headers(response)

# Export the initialization function
__all__ = ['user_bp', 'init_user_routes']