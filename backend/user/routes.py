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
from . import ratings
from . import dashboard
from . import settings
from personalized import get_personalization_system

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
    """User registration"""
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
            preferred_languages=json.dumps(data.get('preferred_languages', ['english', 'telugu'])),
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
        
        from .utils import get_enhanced_user_stats
        stats = get_enhanced_user_stats(user.id)
        
        return jsonify({
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
            }
        }), 201
        
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
    return favorites.get_favorites()

@user_bp.route('/api/user/favorites', methods=['POST', 'OPTIONS'])
def add_to_favorites_route():
    return favorites.add_to_favorites()

@user_bp.route('/api/user/favorites/<int:content_id>', methods=['DELETE', 'OPTIONS'])
def remove_from_favorites_route(content_id):
    from .utils import require_auth
    @require_auth
    def wrapper(current_user):
        return favorites.remove_from_favorites(current_user, content_id)
    return wrapper()

@user_bp.route('/api/user/favorites/<int:content_id>', methods=['GET', 'OPTIONS'])
def check_favorite_status_route(content_id):
    from .utils import require_auth
    @require_auth
    def wrapper(current_user):
        return favorites.check_favorite_status(current_user, content_id)
    return wrapper()

# ============================================================================
# RATINGS ROUTES
# ============================================================================

@user_bp.route('/api/user/ratings', methods=['GET', 'OPTIONS'])
def get_ratings_route():
    return ratings.get_user_ratings()

@user_bp.route('/api/user/ratings', methods=['POST', 'OPTIONS'])
def add_rating_route():
    return ratings.add_rating()

@user_bp.route('/api/user/ratings/<int:content_id>', methods=['DELETE', 'OPTIONS'])
def remove_rating_route(content_id):
    from .utils import require_auth
    @require_auth
    def wrapper(current_user):
        return ratings.remove_rating(current_user, content_id)
    return wrapper()

@user_bp.route('/api/user/ratings/<int:content_id>', methods=['GET', 'OPTIONS'])
def get_rating_for_content_route(content_id):
    from .utils import require_auth
    @require_auth
    def wrapper(current_user):
        return ratings.get_rating_for_content(current_user, content_id)
    return wrapper()

# ============================================================================
# ACTIVITY ROUTES
# ============================================================================

@user_bp.route('/api/interactions', methods=['POST', 'OPTIONS'])
def record_interaction_route():
    return activity.record_interaction()

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
            # Try to use the new advanced personalization system first
            personalization_system = get_personalization_system()
            
            if personalization_system and personalization_system.is_ready():
                # Use the advanced system
                if hasattr(personalization_system, 'recommendation_engine'):
                    advanced_engine = personalization_system.recommendation_engine
                    
                    limit = min(int(request.args.get('limit', 50)), 100)
                    categories = request.args.get('categories')
                    
                    if categories:
                        category_list = [cat.strip() for cat in categories.split(',')]
                    else:
                        category_list = ['for_you']  # Default to for_you with new system
                    
                    # Generate recommendations using advanced system
                    result = advanced_engine.generate_personalized_recommendations(
                        user_id=current_user.id,
                        recommendation_type=category_list[0] if len(category_list) == 1 else 'for_you',
                        limit=limit
                    )
                    
                    # Enhanced response format
                    enhanced_result = {
                        'success': True,
                        'data': result,
                        'message': 'CineBrain Advanced Personalization System recommendations',
                        'user': {
                            'id': current_user.id,
                            'username': current_user.username
                        },
                        'system_info': {
                            'engine_version': '3.0_neural_cultural',
                            'personalization_level': 'advanced',
                            'telugu_priority': 'maximum',
                            'cultural_awareness': 'active',
                            'adaptive_learning': 'enabled'
                        },
                        'metadata': {
                            'generated_at': datetime.utcnow().isoformat(),
                            'algorithm_version': '3.0',
                            'personalization_strength': result.get('user_insights', {}).get('personalization_strength', 0.8)
                        }
                    }
                    
                    return jsonify(enhanced_result), 200
            
            # Fallback to legacy system if advanced system not available
            if not recommendation_engine:
                return jsonify({
                    'error': 'CineBrain recommendation engine not available',
                    'recommendations': {},
                    'fallback': True,
                    'message': 'Please try again later or contact support'
                }), 503
            
            # Legacy system logic (existing code)
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
            
            # Enhanced legacy response
            recommendations['platform'] = 'cinebrain'
            recommendations['user_tier'] = 'premium'
            recommendations['personalization_level'] = 'standard'
            recommendations['system_version'] = 'legacy_compatible'
            
            return jsonify({
                'success': True,
                'data': recommendations,
                'message': 'CineBrain personalized recommendations (legacy system)',
                'user': {
                    'id': current_user.id,
                    'username': current_user.username
                },
                'metadata': {
                    'generated_at': datetime.utcnow().isoformat(),
                    'algorithm_version': '2.0_legacy',
                    'personalization_strength': recommendations.get('recommendation_metadata', {}).get('confidence_score', 0.6)
                }
            }), 200
            
        except Exception as e:
            logger.error(f"CineBrain personalized recommendations error for user {current_user.id}: {e}")
            return jsonify({
                'error': 'Failed to generate CineBrain personalized recommendations',
                'success': False,
                'data': {},
                'fallback_available': True
            }), 500
    
    return wrapper()

@user_bp.route('/api/personalized/neural-for-you', methods=['GET', 'OPTIONS'])
def get_neural_for_you():
    from .utils import require_auth
    
    @require_auth
    def wrapper(current_user):
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            personalization_system = get_personalization_system()
            
            if not personalization_system or not personalization_system.is_ready():
                return jsonify({
                    'error': 'CineBrain Neural Personalization not available',
                    'message': 'Use standard /api/personalized/for-you endpoint',
                    'fallback_endpoint': '/api/personalized/for-you'
                }), 503
            
            if hasattr(personalization_system, 'recommendation_engine'):
                advanced_engine = personalization_system.recommendation_engine
                
                limit = min(int(request.args.get('limit', 30)), 50)
                
                result = advanced_engine.generate_personalized_recommendations(
                    user_id=current_user.id,
                    recommendation_type='for_you',
                    limit=limit
                )
                
                # Add neural-specific metadata
                neural_result = {
                    'success': True,
                    'recommendations': result.get('recommendations', []),
                    'total_count': len(result.get('recommendations', [])),
                    'user_insights': result.get('user_insights', {}),
                    'neural_features': {
                        'embedding_dimension': 128,
                        'collaborative_filtering': 'SVD-based',
                        'cultural_awareness': 'Telugu-first priority',
                        'behavioral_intelligence': 'Advanced pattern recognition',
                        'adaptive_learning': 'Real-time feedback integration'
                    },
                    'algorithm_metadata': result.get('algorithm_metadata', {}),
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                return jsonify(neural_result), 200
            else:
                return jsonify({
                    'error': 'Neural recommendation engine not ready',
                    'message': 'System is initializing, please try again'
                }), 503
                
        except Exception as e:
            logger.error(f"Neural for-you recommendations error: {e}")
            return jsonify({
                'error': 'Failed to generate neural recommendations',
                'fallback_endpoint': '/api/personalized/for-you'
            }), 500
    
    return wrapper()

@user_bp.route('/api/personalized/cultural-match', methods=['GET', 'OPTIONS'])
def get_cultural_match_recommendations():
    from .utils import require_auth
    
    @require_auth
    def wrapper(current_user):
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            personalization_system = get_personalization_system()
            
            if not personalization_system or not personalization_system.is_ready():
                return jsonify({
                    'error': 'CineBrain Cultural Awareness Engine not available'
                }), 503
            
            # Get user's cultural preferences
            user_languages = []
            if current_user.preferred_languages:
                try:
                    user_languages = json.loads(current_user.preferred_languages)
                except:
                    pass
            
            # Default to Telugu if no preferences
            if not user_languages:
                user_languages = ['Telugu']
            
            # Focus on cultural matching
            if hasattr(personalization_system, 'recommendation_engine'):
                advanced_engine = personalization_system.recommendation_engine
                
                limit = min(int(request.args.get('limit', 25)), 40)
                
                # Use Telugu specials or language-specific recommendations
                result = advanced_engine.generate_personalized_recommendations(
                    user_id=current_user.id,
                    recommendation_type='telugu_specials' if 'telugu' in [l.lower() for l in user_languages] else 'your_language',
                    limit=limit,
                    filters={'languages': user_languages, 'cultural_authenticity': True}
                )
                
                cultural_result = {
                    'success': True,
                    'recommendations': result.get('recommendations', []),
                    'total_count': len(result.get('recommendations', [])),
                    'cultural_context': {
                        'user_languages': user_languages,
                        'primary_culture': user_languages[0] if user_languages else 'Telugu',
                        'cultural_authenticity': 'High priority',
                        'regional_focus': 'Telugu cinema and Indian content',
                        'cross_cultural_bridge': 'Enabled for discovery'
                    },
                    'algorithm_metadata': result.get('algorithm_metadata', {}),
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                return jsonify(cultural_result), 200
            else:
                return jsonify({
                    'error': 'Cultural recommendation engine not ready'
                }), 503
                
        except Exception as e:
            logger.error(f"Cultural match recommendations error: {e}")
            return jsonify({
                'error': 'Failed to generate cultural match recommendations'
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
        
        # Database check
        try:
            User.query.limit(1).first()
            health_info['database'] = 'connected'
        except Exception as e:
            health_info['database'] = f'error: {str(e)}'
            health_info['status'] = 'degraded'
        
        # Recommendation engines status
        from .utils import recommendation_engine
        health_info['recommendation_engines'] = {
            'legacy_engine': 'connected' if recommendation_engine else 'not_available',
            'advanced_personalization': 'checking...'
        }
        
        # Check advanced personalization system
        try:
            personalization_system = get_personalization_system()
            if personalization_system:
                system_status = personalization_system.get_system_status()
                health_info['recommendation_engines']['advanced_personalization'] = 'connected'
                health_info['advanced_personalization_status'] = system_status
                health_info['neural_features'] = {
                    'collaborative_filtering': system_status.get('recommendation_engine_ready', False),
                    'cultural_awareness': system_status.get('profile_analyzer_ready', False),
                    'adaptive_learning': system_status.get('initialized', False),
                    'telugu_priority': True
                }
            else:
                health_info['recommendation_engines']['advanced_personalization'] = 'not_available'
        except Exception as e:
            health_info['recommendation_engines']['advanced_personalization'] = f'error: {str(e)}'
        
        # User metrics
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
        
        # Enhanced features list
        health_info['features'] = {
            'personalized_recommendations': True,
            'neural_collaborative_filtering': True,
            'cultural_awareness_engine': True,
            'adaptive_learning': True,
            'behavioral_intelligence': True,
            'cinematic_dna_analysis': True,
            'user_analytics': True,
            'profile_management': True,
            'watchlist_favorites': True,
            'real_time_updates': True,
            'email_username_login': True,
            'avatar_service': True,
            'modular_architecture': True,
            'telugu_cinema_prioritization': True
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