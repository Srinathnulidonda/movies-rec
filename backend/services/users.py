# backend/services/users.py
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import logging
import jwt
from functools import wraps
from collections import defaultdict, Counter

users_bp = Blueprint('users', __name__)

logger = logging.getLogger(__name__)

db = None
User = None
Content = None
UserInteraction = None
Review = None
app = None
recommendation_engine = None
cache = None

def init_users(flask_app, database, models, services):
    global db, User, Content, UserInteraction, Review, app, recommendation_engine, cache
    
    app = flask_app
    db = database
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    Review = models.get('Review')
    cache = services.get('cache')
    
    try:
        from services.personalized import get_recommendation_engine
        recommendation_engine = get_recommendation_engine()
        logger.info("✅ Personalized recommendation engine connected to users service")
    except Exception as e:
        logger.warning(f"Could not connect to recommendation engine: {e}")

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return '', 200
            
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': 'Invalid token'}), 401
            
            current_user.last_active = datetime.utcnow()
            try:
                db.session.commit()
            except Exception as e:
                logger.warning(f"Failed to update last_active: {e}")
                db.session.rollback()
                
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return jsonify({'error': 'Authentication failed'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

def get_enhanced_user_stats(user_id):
    try:
        try:
            from services.auth import EnhancedUserAnalytics
            return EnhancedUserAnalytics.get_comprehensive_user_stats(user_id)
        except ImportError:
            logger.warning("Enhanced analytics not available, using basic stats")
            return get_basic_user_stats(user_id)
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return {}

def get_basic_user_stats(user_id):
    try:
        if not UserInteraction:
            return {
                'total_interactions': 0,
                'content_watched': 0,
                'favorites': 0,
                'watchlist_items': 0,
                'ratings_given': 0
            }
        
        interactions = UserInteraction.query.filter_by(user_id=user_id).all()
        
        stats = {
            'total_interactions': len(interactions),
            'content_watched': len([i for i in interactions if i.interaction_type == 'view']),
            'favorites': len([i for i in interactions if i.interaction_type == 'favorite']),
            'watchlist_items': len([i for i in interactions if i.interaction_type == 'watchlist']),
            'ratings_given': len([i for i in interactions if i.interaction_type == 'rating']),
            'likes_given': len([i for i in interactions if i.interaction_type == 'like']),
            'searches_made': len([i for i in interactions if i.interaction_type == 'search'])
        }
        
        ratings = [i.rating for i in interactions if i.rating is not None]
        stats['average_rating'] = round(sum(ratings) / len(ratings), 1) if ratings else 0
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating basic stats: {e}")
        return {}

@users_bp.route('/api/register', methods=['POST', 'OPTIONS'])
def register():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        if not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        
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
        
        stats = get_enhanced_user_stats(user.id)
        
        return jsonify({
            'message': 'User registered successfully',
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
        logger.error(f"Registration error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@users_bp.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        print(f"CineBrain Login attempt - Raw data: {data}")
        
        if not data:
            print("CineBrain Login error: No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        print(f"CineBrain Login attempt - Username: '{username}', Password length: {len(password) if password else 0}")
        
        if not username or not password:
            print("CineBrain Login error: Missing username or password")
            return jsonify({'error': 'Missing username or password'}), 400
        
        # Find user (case-insensitive)
        user = User.query.filter(User.username.ilike(username)).first()
        
        if not user:
            print(f"CineBrain Login error: User '{username}' not found")
            # List all users for debugging (remove in production)
            all_users = User.query.all()
            print(f"Available users: {[u.username for u in all_users]}")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        print(f"CineBrain Login: Found user {user.username} (ID: {user.id})")
        
        # Check password
        password_valid = check_password_hash(user.password_hash, password)
        print(f"CineBrain Login: Password check result: {password_valid}")
        
        if not password_valid:
            print(f"CineBrain Login error: Invalid password for user '{username}'")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Success
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.secret_key, algorithm='HS256')
        
        stats = get_enhanced_user_stats(user.id)
        
        rec_effectiveness = {}
        try:
            if recommendation_engine:
                rec_effectiveness = recommendation_engine.get_user_recommendation_metrics(user.id)
        except Exception as e:
            logger.warning(f"Could not get recommendation effectiveness: {e}")
        
        print(f"CineBrain Login successful for user: {user.username}")
        
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
        print(f"CineBrain Login exception: {e}")
        logger.error(f"CineBrain Login error: {e}")
        return jsonify({'error': 'CineBrain login failed'}), 500
    
@users_bp.route('/api/users/profile', methods=['GET', 'OPTIONS'])
@require_auth
def get_user_profile(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        stats = get_enhanced_user_stats(current_user.id)
        
        recent_interactions = []
        if UserInteraction:
            try:
                recent = UserInteraction.query.filter_by(
                    user_id=current_user.id
                ).order_by(UserInteraction.timestamp.desc()).limit(10).all()
                
                for interaction in recent:
                    content = Content.query.get(interaction.content_id) if Content else None
                    recent_interactions.append({
                        'id': interaction.id,
                        'interaction_type': interaction.interaction_type,
                        'timestamp': interaction.timestamp.isoformat(),
                        'rating': interaction.rating,
                        'content': {
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                            'slug': content.slug
                        } if content else None
                    })
            except Exception as e:
                logger.warning(f"Could not get recent activity: {e}")
        
        rec_effectiveness = {}
        try:
            if recommendation_engine:
                rec_effectiveness = recommendation_engine.get_user_recommendation_metrics(current_user.id)
        except Exception as e:
            logger.warning(f"Could not get recommendation effectiveness: {e}")
        
        profile_fields = {
            'preferred_languages': current_user.preferred_languages,
            'preferred_genres': current_user.preferred_genres,
            'location': current_user.location,
            'avatar_url': current_user.avatar_url
        }
        
        completed_fields = [field for field, value in profile_fields.items() if value]
        completion_score = min(100, len(completed_fields) * 25)
        missing_fields = [field for field, value in profile_fields.items() if not value]
        
        profile_data = {
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'is_admin': current_user.is_admin,
                'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
                'preferred_genres': json.loads(current_user.preferred_genres or '[]'),
                'location': current_user.location,
                'avatar_url': current_user.avatar_url,
                'created_at': current_user.created_at.isoformat(),
                'last_active': current_user.last_active.isoformat() if current_user.last_active else None
            },
            'stats': stats,
            'recent_activity': recent_interactions,
            'recommendation_effectiveness': rec_effectiveness,
            'profile_completion': {
                'score': completion_score,
                'missing_fields': missing_fields,
                'suggestions': [
                    'Add preferred languages to get better recommendations',
                    'Select favorite genres to improve content discovery',
                    'Add your location for regional content suggestions',
                    'Upload an avatar to personalize your profile'
                ][:len(missing_fields)]
            }
        }
        
        return jsonify(profile_data), 200
        
    except Exception as e:
        logger.error(f"Profile error: {e}")
        return jsonify({'error': 'Failed to get user profile'}), 500

@users_bp.route('/api/users/profile', methods=['PUT', 'OPTIONS'])
@require_auth
def update_user_profile(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        updated_fields = []
        
        if 'preferred_languages' in data:
            current_user.preferred_languages = json.dumps(data['preferred_languages'])
            updated_fields.append('preferred_languages')
        
        if 'preferred_genres' in data:
            current_user.preferred_genres = json.dumps(data['preferred_genres'])
            updated_fields.append('preferred_genres')
        
        if 'location' in data:
            current_user.location = data['location']
            updated_fields.append('location')
        
        if 'avatar_url' in data:
            current_user.avatar_url = data['avatar_url']
            updated_fields.append('avatar_url')
        
        db.session.commit()
        
        if recommendation_engine and updated_fields:
            try:
                recommendation_engine.update_user_preferences_realtime(
                    current_user.id,
                    {
                        'interaction_type': 'profile_update',
                        'metadata': {
                            'updated_fields': updated_fields,
                            'data': data
                        }
                    }
                )
                logger.info(f"Updated recommendation engine for user {current_user.id}")
            except Exception as e:
                logger.warning(f"Failed to update recommendation engine: {e}")
        
        return jsonify({
            'success': True,
            'message': f'Profile updated successfully. Updated: {", ".join(updated_fields)}',
            'updated_fields': updated_fields,
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
                'preferred_genres': json.loads(current_user.preferred_genres or '[]'),
                'location': current_user.location,
                'avatar_url': current_user.avatar_url
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update profile'}), 500

@users_bp.route('/api/users/analytics', methods=['GET', 'OPTIONS'])
@require_auth
def get_user_analytics(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
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
                "Interact with more content (like, favorite, rate) to improve recommendations"
            )
        
        if analytics.get('ratings_given', 0) < 5:
            insights['recommendations']['improvement_tips'].append(
                "Rate content to help our AI understand your preferences better"
            )
        
        if analytics.get('content_diversity', {}).get('genre_diversity_count', 0) < 5:
            insights['recommendations']['improvement_tips'].append(
                "Explore different genres to discover new content you might love"
            )
        
        return jsonify({
            'success': True,
            'analytics': analytics,
            'insights': insights,
            'generated_at': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500

@users_bp.route('/api/interactions', methods=['POST', 'OPTIONS'])
@require_auth
def record_interaction(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'interaction_type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if data['interaction_type'] == 'remove_watchlist':
            interaction = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=data['content_id'],
                interaction_type='watchlist'
            ).first()
            
            if interaction:
                db.session.delete(interaction)
                db.session.commit()
                
                if recommendation_engine:
                    try:
                        recommendation_engine.update_user_preferences_realtime(
                            current_user.id,
                            {
                                'content_id': data['content_id'],
                                'interaction_type': 'remove_watchlist',
                                'metadata': data.get('metadata', {})
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update real-time preferences: {e}")
                
                return jsonify({
                    'success': True,
                    'message': 'Removed from watchlist'
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'message': 'Content not in watchlist'
                }), 404
        
        if data['interaction_type'] == 'watchlist':
            existing = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=data['content_id'],
                interaction_type='watchlist'
            ).first()
            
            if existing:
                return jsonify({
                    'success': True,
                    'message': 'Already in watchlist'
                }), 200
        
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=data['content_id'],
            interaction_type=data['interaction_type'],
            rating=data.get('rating'),
            interaction_metadata=json.dumps(data.get('metadata', {}))
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        if recommendation_engine:
            try:
                recommendation_engine.update_user_preferences_realtime(
                    current_user.id,
                    {
                        'content_id': data['content_id'],
                        'interaction_type': data['interaction_type'],
                        'rating': data.get('rating'),
                        'metadata': data.get('metadata', {})
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update real-time preferences: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Interaction recorded successfully',
            'interaction_id': interaction.id
        }), 201
        
    except Exception as e:
        logger.error(f"Interaction recording error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record interaction'}), 500

@users_bp.route('/api/personalized/', methods=['GET', 'OPTIONS'])
@require_auth
def get_personalized_recommendations(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        if not recommendation_engine:
            return jsonify({
                'error': 'Recommendation engine not available',
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
            'message': 'Personalized recommendations generated successfully',
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
        logger.error(f"Personalized recommendations error for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to generate personalized recommendations',
            'success': False,
            'data': {}
        }), 500

@users_bp.route('/api/personalized/for-you', methods=['GET', 'OPTIONS'])
@require_auth
def get_for_you_recommendations(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        if not recommendation_engine:
            return jsonify({'error': 'Recommendation engine not available'}), 503
        
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
        logger.error(f"For You recommendations error: {e}")
        return jsonify({'error': 'Failed to get For You recommendations'}), 500

@users_bp.route('/api/personalized/because-you-watched', methods=['GET', 'OPTIONS'])
@require_auth
def get_because_you_watched(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        if not recommendation_engine:
            return jsonify({'error': 'Recommendation engine not available'}), 503
        
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
            'explanation': 'Based on your recently watched content',
            'algorithm': 'content_similarity_and_collaborative_filtering'
        }), 200
        
    except Exception as e:
        logger.error(f"Because you watched recommendations error: {e}")
        return jsonify({'error': 'Failed to get because you watched recommendations'}), 500

@users_bp.route('/api/personalized/trending-for-you', methods=['GET', 'OPTIONS'])
@require_auth
def get_trending_for_you(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        if not recommendation_engine:
            return jsonify({'error': 'Recommendation engine not available'}), 503
        
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
            'explanation': 'Trending content personalized for your taste',
            'algorithm': 'hybrid_trending_personalization'
        }), 200
        
    except Exception as e:
        logger.error(f"Trending for you recommendations error: {e}")
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

@users_bp.route('/api/personalized/your-language', methods=['GET', 'OPTIONS'])
@require_auth
def get_your_language_recommendations(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        if not recommendation_engine:
            return jsonify({'error': 'Recommendation engine not available'}), 503
        
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
            'explanation': 'Content in your preferred languages',
            'preferred_languages': preferred_languages,
            'algorithm': 'language_preference_filtering'
        }), 200
        
    except Exception as e:
        logger.error(f"Language recommendations error: {e}")
        return jsonify({'error': 'Failed to get language recommendations'}), 500

@users_bp.route('/api/personalized/hidden-gems', methods=['GET', 'OPTIONS'])
@require_auth
def get_hidden_gems(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        if not recommendation_engine:
            return jsonify({'error': 'Recommendation engine not available'}), 503
        
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
            'explanation': 'High-quality content you might have missed',
            'algorithm': 'hidden_gem_discovery'
        }), 200
        
    except Exception as e:
        logger.error(f"Hidden gems recommendations error: {e}")
        return jsonify({'error': 'Failed to get hidden gems recommendations'}), 500

@users_bp.route('/api/personalized/profile-insights', methods=['GET', 'OPTIONS'])
@require_auth
def get_profile_insights(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        if not recommendation_engine:
            return jsonify({'error': 'Recommendation engine not available'}), 503
        
        user_profile = recommendation_engine.user_profiler.build_comprehensive_user_profile(current_user.id)
        
        if not user_profile:
            return jsonify({
                'success': False,
                'message': 'Could not build user profile - insufficient interaction data',
                'suggestion': 'Interact with more content to build your profile'
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
        logger.error(f"Profile insights error: {e}")
        return jsonify({'error': 'Failed to get profile insights'}), 500

def _get_improvement_suggestion(user_profile):
    completeness = user_profile.get('profile_completeness', 0)
    total_interactions = user_profile.get('implicit_preferences', {}).get('total_interactions', 0)
    ratings_count = user_profile.get('explicit_preferences', {}).get('ratings_count', 0)
    
    if completeness < 0.3:
        return 'Interact with more content (like, favorite, add to watchlist) to improve accuracy'
    elif ratings_count < 5:
        return 'Rate more content to help our AI understand your taste better'
    elif completeness < 0.8:
        return 'Explore different genres to get more diverse recommendations'
    else:
        return 'Your recommendations are highly accurate! Keep discovering new content'

@users_bp.route('/api/personalized/update-preferences', methods=['POST', 'OPTIONS'])
@require_auth
def update_user_preferences(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        if 'preferred_languages' in data:
            current_user.preferred_languages = json.dumps(data['preferred_languages'])
        
        if 'preferred_genres' in data:
            current_user.preferred_genres = json.dumps(data['preferred_genres'])
        
        db.session.commit()
        
        if recommendation_engine:
            try:
                recommendation_engine.update_user_preferences_realtime(
                    current_user.id,
                    {
                        'interaction_type': 'preference_update',
                        'metadata': {
                            'updated_languages': data.get('preferred_languages'),
                            'updated_genres': data.get('preferred_genres'),
                            'source': 'explicit_preference_update'
                        }
                    }
                )
                logger.info(f"Successfully updated preferences for user {current_user.id}")
            except Exception as e:
                logger.warning(f"Failed to update recommendation engine: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Preferences updated successfully',
            'user': {
                'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
                'preferred_genres': json.loads(current_user.preferred_genres or '[]')
            },
            'recommendation_refresh': 'triggered'
        }), 200
        
    except Exception as e:
        logger.error(f"Update preferences error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update preferences'}), 500

@users_bp.route('/api/user/watchlist', methods=['GET', 'OPTIONS'])
@require_auth
def get_watchlist(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        watchlist_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='watchlist'
        ).order_by(UserInteraction.timestamp.desc()).all()
        
        content_ids = [interaction.content_id for interaction in watchlist_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        content_map = {content.id: content for content in contents}
        
        result = []
        for interaction in watchlist_interactions:
            content = content_map.get(interaction.content_id)
            if content:
                youtube_url = None
                if content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                result.append({
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'youtube_trailer': youtube_url,
                    'added_at': interaction.timestamp.isoformat(),
                    'is_new_release': content.is_new_release,
                    'is_trending': content.is_trending
                })
        
        return jsonify({
            'watchlist': result,
            'total_count': len(result),
            'last_updated': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Watchlist error: {e}")
        return jsonify({'error': 'Failed to get watchlist'}), 500

@users_bp.route('/api/user/watchlist/<int:content_id>', methods=['DELETE', 'OPTIONS'])
@require_auth
def remove_from_watchlist(current_user, content_id):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='watchlist'
        ).first()
        
        if interaction:
            db.session.delete(interaction)
            db.session.commit()
            
            if recommendation_engine:
                try:
                    recommendation_engine.update_user_preferences_realtime(
                        current_user.id,
                        {
                            'content_id': content_id,
                            'interaction_type': 'remove_watchlist'
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to update recommendations: {e}")
            
            return jsonify({
                'success': True,
                'message': 'Removed from watchlist'
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Content not in watchlist'
            }), 404
            
    except Exception as e:
        logger.error(f"Remove from watchlist error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to remove from watchlist'}), 500

@users_bp.route('/api/user/watchlist/<int:content_id>', methods=['GET', 'OPTIONS'])
@require_auth
def check_watchlist_status(current_user, content_id):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='watchlist'
        ).first()
        
        return jsonify({
            'in_watchlist': interaction is not None,
            'added_at': interaction.timestamp.isoformat() if interaction else None
        }), 200
        
    except Exception as e:
        logger.error(f"Check watchlist status error: {e}")
        return jsonify({'error': 'Failed to check watchlist status'}), 500

@users_bp.route('/api/user/favorites', methods=['GET', 'OPTIONS'])
@require_auth
def get_favorites(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        favorite_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='favorite'
        ).order_by(UserInteraction.timestamp.desc()).all()
        
        content_ids = [interaction.content_id for interaction in favorite_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        content_map = {content.id: content for content in contents}
        
        result = []
        for interaction in favorite_interactions:
            content = content_map.get(interaction.content_id)
            if content:
                youtube_url = None
                if content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                result.append({
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'youtube_trailer': youtube_url,
                    'favorited_at': interaction.timestamp.isoformat(),
                    'user_rating': interaction.rating
                })
        
        return jsonify({
            'favorites': result,
            'total_count': len(result),
            'last_updated': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Favorites error: {e}")
        return jsonify({'error': 'Failed to get favorites'}), 500

@users_bp.route('/api/user/ratings', methods=['GET', 'OPTIONS'])
@require_auth
def get_user_ratings(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        rating_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='rating'
        ).filter(UserInteraction.rating.isnot(None)).order_by(
            UserInteraction.timestamp.desc()
        ).all()
        
        content_ids = [interaction.content_id for interaction in rating_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        content_map = {content.id: content for content in contents}
        
        result = []
        for interaction in rating_interactions:
            content = content_map.get(interaction.content_id)
            if content:
                result.append({
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'user_rating': interaction.rating,
                    'imdb_rating': content.rating,
                    'rated_at': interaction.timestamp.isoformat()
                })
        
        ratings = [interaction.rating for interaction in rating_interactions]
        stats = {
            'total_ratings': len(ratings),
            'average_rating': round(sum(ratings) / len(ratings), 1) if ratings else 0,
            'highest_rating': max(ratings) if ratings else 0,
            'lowest_rating': min(ratings) if ratings else 0
        }
        
        return jsonify({
            'ratings': result,
            'stats': stats,
            'last_updated': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"User ratings error: {e}")
        return jsonify({'error': 'Failed to get user ratings'}), 500

@users_bp.after_request
def after_request(response):
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

@users_bp.route('/api/users/health', methods=['GET'])
def users_health():
    try:
        health_info = {
            'status': 'healthy',
            'service': 'users',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '3.0.0'
        }
        
        try:
            User.query.limit(1).first()
            health_info['database'] = 'connected'
        except Exception as e:
            health_info['database'] = f'error: {str(e)}'
            health_info['status'] = 'degraded'
        
        health_info['recommendation_engine'] = 'connected' if recommendation_engine else 'not_available'
        
        try:
            total_users = User.query.count()
            active_users = User.query.filter(
                User.last_active >= datetime.utcnow() - timedelta(days=7)
            ).count()
            
            health_info['user_metrics'] = {
                'total_users': total_users,
                'active_users_7d': active_users,
                'activity_rate': round((active_users / total_users * 100), 1) if total_users > 0 else 0
            }
        except Exception as e:
            health_info['user_metrics'] = {'error': str(e)}
        
        health_info['features'] = {
            'personalized_recommendations': True,
            'user_analytics': True,
            'profile_management': True,
            'watchlist_favorites': True,
            'real_time_updates': True
        }
        
        return jsonify(health_info), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'users',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

__all__ = ['users_bp', 'init_users']