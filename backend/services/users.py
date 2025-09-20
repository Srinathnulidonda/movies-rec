from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import logging
import jwt
from functools import wraps
from ..ml_services.recommendation import RecommendationEngine

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
    
    recommendation_engine = RecommendationEngine(db, models)

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
                recommendation_engine.update_user_profile(current_user.id)
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
            interaction_metadata=data.get('metadata', {})
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        recommendation_engine.update_user_profile(current_user.id)
        
        return jsonify({'message': 'Interaction recorded successfully'}), 201
        
    except Exception as e:
        logger.error(f"Interaction recording error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record interaction'}), 500

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
            recommendation_engine.update_user_profile(current_user.id)
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

@users_bp.route('/api/recommendations/personalized', methods=['GET'])
@require_auth
def get_personalized_recommendations(current_user):
    try:
        limit = int(request.args.get('limit', 20))
        content_type = request.args.get('content_type', 'all')
        strategy = request.args.get('strategy', 'hybrid')
        
        recommendations = recommendation_engine.get_personalized_recommendations(
            user_id=current_user.id,
            limit=limit,
            content_type=content_type,
            strategy=strategy
        )
        
        result = []
        for rec in recommendations:
            content = rec['content']
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
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url,
                'recommendation_score': rec['score'],
                'recommendation_reason': rec['reason'],
                'algorithm_used': rec['algorithm'],
                'confidence': rec['confidence']
            })
        
        return jsonify({
            'recommendations': result,
            'strategy': strategy,
            'total_interactions': recommendation_engine.get_user_interaction_count(current_user.id),
            'user_profile_strength': recommendation_engine.get_user_profile_strength(current_user.id)
        }), 200
        
    except Exception as e:
        logger.error(f"Personalized recommendations error: {e}")
        return jsonify({'recommendations': [], 'error': 'Failed to get recommendations'}), 200

@users_bp.route('/api/recommendations/ml-personalized', methods=['GET'])
@require_auth
def get_ml_personalized_recommendations(current_user):
    try:
        limit = int(request.args.get('limit', 20))
        include_explanations = request.args.get('include_explanations', 'true').lower() == 'true'
        diversity_factor = float(request.args.get('diversity_factor', 0.3))
        
        recommendations = recommendation_engine.get_advanced_recommendations(
            user_id=current_user.id,
            limit=limit,
            include_explanations=include_explanations,
            diversity_factor=diversity_factor
        )
        
        result = []
        for rec in recommendations:
            content = rec['content']
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
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url,
                'ml_score': rec['ml_score'],
                'ml_reason': rec['explanation'],
                'algorithm_mix': rec['algorithm_mix'],
                'confidence': rec['confidence'],
                'novelty_score': rec['novelty_score'],
                'diversity_contribution': rec['diversity_contribution']
            })
        
        metrics = recommendation_engine.get_recommendation_metrics(current_user.id)
        
        return jsonify({
            'recommendations': result,
            'ml_strategy': 'advanced_hybrid',
            'user_metrics': metrics,
            'diversity_applied': diversity_factor,
            'recommendation_quality': 'high_precision'
        }), 200
        
    except Exception as e:
        logger.error(f"ML personalized recommendations error: {e}")
        return jsonify({'recommendations': [], 'error': 'Failed to get recommendations'}), 200

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
                    'metadata': interaction.interaction_metadata or {}
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
        
        required_fields = ['content_id', 'feedback_type', 'recommendation_id']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        recommendation_engine.record_recommendation_feedback(
            user_id=current_user.id,
            content_id=data['content_id'],
            feedback_type=data['feedback_type'],
            recommendation_id=data['recommendation_id'],
            feedback_value=data.get('feedback_value', 1.0)
        )
        
        return jsonify({'message': 'Feedback recorded successfully'}), 201
        
    except Exception as e:
        logger.error(f"Recommendation feedback error: {e}")
        return jsonify({'error': 'Failed to record feedback'}), 500