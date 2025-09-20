from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import logging
import jwt
import sys
import os
from functools import wraps

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from ml_services.recommendation import PersonalizedRecommendationEngine
except ImportError:
    PersonalizedRecommendationEngine = None

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
    
    if PersonalizedRecommendationEngine:
        try:
            recommendation_engine = PersonalizedRecommendationEngine(db, models)
            print("Enhanced Personalized Recommendation Engine initialized")
        except Exception as e:
            print(f"Warning: Failed to initialize Enhanced Recommendation Engine: {e}")
            recommendation_engine = None

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
            preferred_languages=json.dumps(data.get('preferred_languages', ['english', 'telugu'])),
            preferred_genres=json.dumps(data.get('preferred_genres', []))
        )
        
        db.session.add(user)
        db.session.commit()
        
        if recommendation_engine:
            recommendation_engine.initialize_user_profile(user.id)
        
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
                if recommendation_engine:
                    recommendation_engine.update_user_behavior_profile(current_user.id)
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
        
        metadata = data.get('metadata', {})
        if data['interaction_type'] == 'search':
            metadata['query'] = data.get('search_query', '')
            metadata['search_timestamp'] = datetime.utcnow().isoformat()
        elif data['interaction_type'] == 'view':
            metadata['view_duration'] = data.get('view_duration', 0)
            metadata['completion_percentage'] = data.get('completion_percentage', 0)
        
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=data['content_id'],
            interaction_type=data['interaction_type'],
            rating=data.get('rating'),
            interaction_metadata=metadata
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        if recommendation_engine:
            recommendation_engine.update_user_behavior_profile(current_user.id)
            recommendation_engine.process_real_time_interaction(
                current_user.id, 
                data['content_id'], 
                data['interaction_type'],
                data.get('rating'),
                metadata
            )
        
        return jsonify({'message': 'Interaction recorded successfully'}), 201
        
    except Exception as e:
        logger.error(f"Interaction recording error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record interaction'}), 500

@users_bp.route('/api/interactions/batch', methods=['POST'])
@require_auth
def record_batch_interactions(current_user):
    try:
        data = request.get_json()
        interactions_data = data.get('interactions', [])
        
        if not interactions_data:
            return jsonify({'error': 'No interactions provided'}), 400
        
        recorded_count = 0
        
        for interaction_data in interactions_data:
            if 'content_id' in interaction_data and 'interaction_type' in interaction_data:
                metadata = interaction_data.get('metadata', {})
                
                interaction = UserInteraction(
                    user_id=current_user.id,
                    content_id=interaction_data['content_id'],
                    interaction_type=interaction_data['interaction_type'],
                    rating=interaction_data.get('rating'),
                    interaction_metadata=metadata
                )
                
                db.session.add(interaction)
                recorded_count += 1
        
        db.session.commit()
        
        if recommendation_engine and recorded_count > 0:
            recommendation_engine.update_user_behavior_profile(current_user.id)
        
        return jsonify({
            'message': f'Recorded {recorded_count} interactions successfully',
            'recorded_count': recorded_count
        }), 201
        
    except Exception as e:
        logger.error(f"Batch interaction error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record batch interactions'}), 500

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
                'slug': getattr(content, 'slug', None),
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
                'slug': getattr(content, 'slug', None),
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

@users_bp.route('/api/recommendations/personalized', methods=['GET'])
@require_auth
def get_personalized_recommendations(current_user):
    try:
        if not recommendation_engine:
            return jsonify({
                'recommendations': [],
                'error': 'Recommendation engine not available'
            }), 200
        
        limit = int(request.args.get('limit', 30))
        content_type = request.args.get('content_type', 'all')
        strategy = request.args.get('strategy', 'intelligent_hybrid')
        include_explanations = request.args.get('include_explanations', 'true').lower() == 'true'
        
        recommendations = recommendation_engine.get_ultra_personalized_recommendations(
            user_id=current_user.id,
            limit=limit,
            content_type=content_type,
            strategy=strategy,
            include_explanations=include_explanations
        )
        
        result = []
        for rec in recommendations:
            content = rec['content']
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'slug': getattr(content, 'slug', None),
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url,
                'personalization_score': rec['personalization_score'],
                'match_reason': rec['match_reason'],
                'confidence_level': rec['confidence_level'],
                'behavioral_match': rec['behavioral_match'],
                'preference_alignment': rec['preference_alignment']
            })
        
        user_profile = recommendation_engine.get_comprehensive_user_profile(current_user.id)
        
        return jsonify({
            'recommendations': result,
            'user_profile': user_profile,
            'recommendation_strategy': strategy,
            'total_analyzed_interactions': user_profile.get('total_interactions', 0),
            'personalization_strength': user_profile.get('profile_strength', 'unknown')
        }), 200
        
    except Exception as e:
        logger.error(f"Personalized recommendations error: {e}")
        return jsonify({'recommendations': [], 'error': 'Failed to get recommendations'}), 500

@users_bp.route('/api/recommendations/behavior-based', methods=['GET'])
@require_auth
def get_behavior_based_recommendations(current_user):
    try:
        if not recommendation_engine:
            return jsonify({
                'recommendations': [],
                'error': 'Recommendation engine not available'
            }), 200
        
        limit = int(request.args.get('limit', 25))
        behavior_focus = request.args.get('behavior_focus', 'comprehensive')
        temporal_weight = float(request.args.get('temporal_weight', 0.7))
        
        recommendations = recommendation_engine.get_behavior_driven_recommendations(
            user_id=current_user.id,
            limit=limit,
            behavior_focus=behavior_focus,
            temporal_weight=temporal_weight
        )
        
        result = []
        for rec in recommendations:
            content = rec['content']
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'slug': getattr(content, 'slug', None),
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url,
                'behavior_score': rec['behavior_score'],
                'behavior_pattern': rec['behavior_pattern'],
                'temporal_relevance': rec['temporal_relevance'],
                'interaction_affinity': rec['interaction_affinity']
            })
        
        behavior_analysis = recommendation_engine.analyze_user_behavior_patterns(current_user.id)
        
        return jsonify({
            'recommendations': result,
            'behavior_analysis': behavior_analysis,
            'behavior_focus': behavior_focus,
            'temporal_weight': temporal_weight
        }), 200
        
    except Exception as e:
        logger.error(f"Behavior-based recommendations error: {e}")
        return jsonify({'recommendations': [], 'error': 'Failed to get recommendations'}), 500

@users_bp.route('/api/user/profile/update-preferences', methods=['POST'])
@require_auth
def update_user_preferences(current_user):
    try:
        data = request.get_json()
        
        if 'preferred_languages' in data:
            current_user.preferred_languages = json.dumps(data['preferred_languages'])
        
        if 'preferred_genres' in data:
            current_user.preferred_genres = json.dumps(data['preferred_genres'])
        
        db.session.commit()
        
        if recommendation_engine:
            recommendation_engine.update_user_behavior_profile(current_user.id)
        
        return jsonify({'message': 'Preferences updated successfully'}), 200
        
    except Exception as e:
        logger.error(f"Update preferences error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update preferences'}), 500

@users_bp.route('/api/user/interaction-history', methods=['GET'])
@require_auth
def get_interaction_history(current_user):
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        interaction_type = request.args.get('type', 'all')
        
        query = UserInteraction.query.filter_by(user_id=current_user.id)
        
        if interaction_type != 'all':
            query = query.filter_by(interaction_type=interaction_type)
        
        interactions = query.order_by(UserInteraction.timestamp.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        result = []
        for interaction in interactions.items:
            content = Content.query.get(interaction.content_id)
            if content:
                result.append({
                    'interaction_id': interaction.id,
                    'content_id': content.id,
                    'content_title': content.title,
                    'content_type': content.content_type,
                    'interaction_type': interaction.interaction_type,
                    'rating': interaction.rating,
                    'timestamp': interaction.timestamp.isoformat(),
                    'metadata': getattr(interaction, 'interaction_metadata', {}) or {}
                })
        
        return jsonify({
            'interactions': result,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': interactions.total,
                'pages': interactions.pages,
                'has_next': interactions.has_next,
                'has_prev': interactions.has_prev
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Interaction history error: {e}")
        return jsonify({'error': 'Failed to get interaction history'}), 500

@users_bp.route('/api/user/recommendation-feedback', methods=['POST'])
@require_auth
def record_recommendation_feedback(current_user):
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'feedback_type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if recommendation_engine:
            recommendation_engine.record_detailed_feedback(
                user_id=current_user.id,
                content_id=data['content_id'],
                feedback_type=data['feedback_type'],
                feedback_value=data.get('feedback_value', 1.0),
                feedback_context=data.get('feedback_context', {})
            )
        
        return jsonify({'message': 'Feedback recorded successfully'}), 201
        
    except Exception as e:
        logger.error(f"Recommendation feedback error: {e}")
        return jsonify({'error': 'Failed to record feedback'}), 500

@users_bp.route('/api/user/profile/analytics', methods=['GET'])
@require_auth
def get_user_profile_analytics(current_user):
    try:
        if not recommendation_engine:
            return jsonify({'error': 'Analytics not available'}), 503
        
        analytics = recommendation_engine.get_user_analytics(current_user.id)
        
        return jsonify(analytics), 200
        
    except Exception as e:
        logger.error(f"Profile analytics error: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500