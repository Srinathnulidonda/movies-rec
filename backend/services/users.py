# backend/services/users.py
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import logging
import jwt
from functools import wraps

users_bp = Blueprint('users', __name__)

logger = logging.getLogger(__name__)

db = None
User = None
Content = None
UserInteraction = None
app = None
recommendation_engine = None

def init_users(flask_app, database, models, services):
    global db, User, Content, UserInteraction, app, recommendation_engine
    
    app = flask_app
    db = database
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    
    # Import recommendation engine
    try:
        from services.personalized import get_recommendation_engine
        recommendation_engine = get_recommendation_engine()
        logger.info("Personalized recommendation engine connected to users service")
    except Exception as e:
        logger.warning(f"Could not connect to recommendation engine: {e}")

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': 'Invalid token'}), 401
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

@users_bp.route('/api/register', methods=['POST'])
def register():
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
            preferred_languages=json.dumps(data.get('preferred_languages', [])),
            preferred_genres=json.dumps(data.get('preferred_genres', []))
        )
        
        db.session.add(user)
        db.session.commit()
        
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.secret_key, algorithm='HS256')
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@users_bp.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Missing username or password'}), 400
        
        user = User.query.filter_by(username=data['username']).first()
        
        if not user or not check_password_hash(user.password_hash, data['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.secret_key, algorithm='HS256')
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@users_bp.route('/api/interactions', methods=['POST'])
@require_auth
def record_interaction(current_user):
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
                
                # Update recommendations in real-time
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
                
                return jsonify({'message': 'Removed from watchlist'}), 200
            else:
                return jsonify({'message': 'Content not in watchlist'}), 404
        
        if data['interaction_type'] == 'watchlist':
            existing = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=data['content_id'],
                interaction_type='watchlist'
            ).first()
            
            if existing:
                return jsonify({'message': 'Already in watchlist'}), 200
        
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=data['content_id'],
            interaction_type=data['interaction_type'],
            rating=data.get('rating'),
            interaction_metadata=json.dumps(data.get('metadata', {}))
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        # Update recommendations in real-time
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
        
        return jsonify({'message': 'Interaction recorded successfully'}), 201
        
    except Exception as e:
        logger.error(f"Interaction recording error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record interaction'}), 500

# Personalized Recommendation Endpoints

@users_bp.route('/api/personalized/', methods=['GET'])
@require_auth
def get_personalized_recommendations(current_user):
    """
    Get Netflix-level personalized recommendations for cinbrain users
    
    Analyzes user interactions (search history, views, favorites, watchlist, ratings)
    and content metadata (storylines, synopsis, genres) using hybrid recommendation 
    techniques to deliver highly accurate, real-time personalized recommendations.
    """
    try:
        if not recommendation_engine:
            return jsonify({
                'error': 'Recommendation engine not available',
                'recommendations': {},
                'fallback': True
            }), 503
        
        # Get query parameters
        limit = min(int(request.args.get('limit', 50)), 100)  # Cap at 100
        categories = request.args.get('categories')
        
        if categories:
            category_list = [cat.strip() for cat in categories.split(',')]
        else:
            category_list = None
        
        # Generate Netflix-level personalized recommendations
        recommendations = recommendation_engine.get_personalized_recommendations(
            user_id=current_user.id,
            limit=limit,
            categories=category_list
        )
        
        # Add cinbrain branding
        recommendations['platform'] = 'cinbrain'
        recommendations['user_tier'] = 'premium'  # All registered users get premium experience
        
        return jsonify({
            'success': True,
            'data': recommendations,
            'message': 'Personalized recommendations generated successfully',
            'user': {
                'id': current_user.id,
                'username': current_user.username
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Personalized recommendations error for user {current_user.id}: {e}")
        return jsonify({
            'error': 'Failed to generate personalized recommendations',
            'success': False,
            'data': {}
        }), 500

@users_bp.route('/api/personalized/for-you', methods=['GET'])
@require_auth
def get_for_you_recommendations(current_user):
    """Get main 'For You' personalized feed"""
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
            'metadata': recommendations.get('recommendation_metadata', {})
        }), 200
        
    except Exception as e:
        logger.error(f"For You recommendations error: {e}")
        return jsonify({'error': 'Failed to get For You recommendations'}), 500

@users_bp.route('/api/personalized/because-you-watched', methods=['GET'])
@require_auth
def get_because_you_watched(current_user):
    """Get 'Because you watched X' recommendations"""
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
            'explanation': 'Based on your recently watched content'
        }), 200
        
    except Exception as e:
        logger.error(f"Because you watched recommendations error: {e}")
        return jsonify({'error': 'Failed to get because you watched recommendations'}), 500

@users_bp.route('/api/personalized/trending-for-you', methods=['GET'])
@require_auth
def get_trending_for_you(current_user):
    """Get personalized trending recommendations"""
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
            'explanation': 'Trending content personalized for your taste'
        }), 200
        
    except Exception as e:
        logger.error(f"Trending for you recommendations error: {e}")
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

@users_bp.route('/api/personalized/your-language', methods=['GET'])
@require_auth
def get_your_language_recommendations(current_user):
    """Get language-specific personalized recommendations"""
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
        
        return jsonify({
            'success': True,
            'recommendations': language_recs,
            'total_count': len(language_recs),
            'explanation': 'Content in your preferred languages'
        }), 200
        
    except Exception as e:
        logger.error(f"Language recommendations error: {e}")
        return jsonify({'error': 'Failed to get language recommendations'}), 500

@users_bp.route('/api/personalized/hidden-gems', methods=['GET'])
@require_auth
def get_hidden_gems(current_user):
    """Get hidden gem recommendations"""
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
            'explanation': 'High-quality content you might have missed'
        }), 200
        
    except Exception as e:
        logger.error(f"Hidden gems recommendations error: {e}")
        return jsonify({'error': 'Failed to get hidden gems recommendations'}), 500

@users_bp.route('/api/personalized/profile-insights', methods=['GET'])
@require_auth
def get_profile_insights(current_user):
    """Get user profile insights and recommendation analytics"""
    try:
        if not recommendation_engine:
            return jsonify({'error': 'Recommendation engine not available'}), 503
        
        # Get user profile
        user_profile = recommendation_engine.user_profiler.build_comprehensive_user_profile(current_user.id)
        
        if not user_profile:
            return jsonify({
                'success': False,
                'message': 'Could not build user profile'
            }), 404
        
        # Extract key insights
        insights = {
            'profile_strength': {
                'completeness': user_profile.get('profile_completeness', 0),
                'confidence': user_profile.get('confidence_score', 0),
                'status': 'strong' if user_profile.get('confidence_score', 0) > 0.7 else 'building'
            },
            'preferences': {
                'top_genres': user_profile.get('genre_preferences', {}).get('top_genres', [])[:5],
                'preferred_languages': user_profile.get('language_preferences', {}).get('preferred_languages', [])[:3],
                'content_types': user_profile.get('content_type_preferences', {}).get('content_type_scores', {})
            },
            'behavior': {
                'engagement_score': user_profile.get('engagement_score', 0),
                'viewing_style': user_profile.get('implicit_preferences', {}).get('most_common_interaction'),
                'exploration_tendency': user_profile.get('exploration_tendency', 0),
                'total_interactions': user_profile.get('implicit_preferences', {}).get('total_interactions', 0)
            },
            'recent_activity': user_profile.get('recent_activity', {}),
            'recommendations_quality': {
                'accuracy_estimate': min(user_profile.get('confidence_score', 0) * 100, 95),
                'next_improvement': 'Rate more content to improve accuracy' if user_profile.get('profile_completeness', 0) < 0.8 else 'Your recommendations are highly accurate!'
            }
        }
        
        return jsonify({
            'success': True,
            'insights': insights,
            'last_updated': user_profile.get('last_updated', datetime.utcnow()).isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Profile insights error: {e}")
        return jsonify({'error': 'Failed to get profile insights'}), 500

@users_bp.route('/api/personalized/update-preferences', methods=['POST'])
@require_auth
def update_user_preferences(current_user):
    """Update user preferences and trigger recommendation refresh"""
    try:
        data = request.get_json()
        
        # Update user preferences
        if 'preferred_languages' in data:
            current_user.preferred_languages = json.dumps(data['preferred_languages'])
        
        if 'preferred_genres' in data:
            current_user.preferred_genres = json.dumps(data['preferred_genres'])
        
        db.session.commit()
        
        # Trigger recommendation engine update
        if recommendation_engine:
            try:
                recommendation_engine.update_user_preferences_realtime(
                    current_user.id,
                    {
                        'interaction_type': 'preference_update',
                        'metadata': {
                            'updated_languages': data.get('preferred_languages'),
                            'updated_genres': data.get('preferred_genres')
                        }
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update recommendation engine: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Preferences updated successfully',
            'user': {
                'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
                'preferred_genres': json.loads(current_user.preferred_genres or '[]')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Update preferences error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update preferences'}), 500

# Existing endpoints (watchlist, favorites, etc.)

@users_bp.route('/api/user/watchlist', methods=['GET'])
@require_auth
def get_watchlist(current_user):
    try:
        watchlist_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='watchlist'
        ).all()
        
        content_ids = [interaction.content_id for interaction in watchlist_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        
        result = []
        for content in contents:
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
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'youtube_trailer': youtube_url
            })
        
        return jsonify({'watchlist': result}), 200
        
    except Exception as e:
        logger.error(f"Watchlist error: {e}")
        return jsonify({'error': 'Failed to get watchlist'}), 500

@users_bp.route('/api/user/watchlist/<int:content_id>', methods=['DELETE'])
@require_auth
def remove_from_watchlist(current_user, content_id):
    try:
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='watchlist'
        ).first()
        
        if interaction:
            db.session.delete(interaction)
            db.session.commit()
            
            # Update recommendations
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
            
            return jsonify({'message': 'Removed from watchlist'}), 200
        else:
            return jsonify({'message': 'Content not in watchlist'}), 404
            
    except Exception as e:
        logger.error(f"Remove from watchlist error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to remove from watchlist'}), 500

@users_bp.route('/api/user/watchlist/<int:content_id>', methods=['GET'])
@require_auth
def check_watchlist_status(current_user, content_id):
    try:
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='watchlist'
        ).first()
        
        return jsonify({'in_watchlist': interaction is not None}), 200
        
    except Exception as e:
        logger.error(f"Check watchlist status error: {e}")
        return jsonify({'error': 'Failed to check watchlist status'}), 500

@users_bp.route('/api/user/favorites', methods=['GET'])
@require_auth
def get_favorites(current_user):
    try:
        favorite_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='favorite'
        ).all()
        
        content_ids = [interaction.content_id for interaction in favorite_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        
        result = []
        for content in contents:
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
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'youtube_trailer': youtube_url
            })
        
        return jsonify({'favorites': result}), 200
        
    except Exception as e:
        logger.error(f"Favorites error: {e}")
        return jsonify({'error': 'Failed to get favorites'}), 500