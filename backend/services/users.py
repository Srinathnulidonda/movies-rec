#backend/services/users.py
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import logging
import jwt
from functools import wraps
import os

users_bp = Blueprint('users', __name__)

logger = logging.getLogger(__name__)

db = None
User = None
Content = None
UserInteraction = None
http_session = None
app = None
cache = None

ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-aksq.onrender.com')

def init_users(flask_app, database, models, services):
    global db, User, Content, UserInteraction
    global http_session, app, cache
    
    app = flask_app
    db = database
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    
    http_session = services['http_session']
    cache = services['cache']

class MLServiceClient:
    
    @staticmethod
    def call_ml_service(endpoint, params=None, timeout=10, use_cache=True):
        try:
            if not ML_SERVICE_URL:
                return None
            
            cache_key = f"ml:{endpoint}:{json.dumps(params, sort_keys=True)}"
            
            if use_cache and cache:
                cached_result = cache.get(cache_key)
                if cached_result:
                    logger.info(f"ML service cache hit for {endpoint}")
                    return cached_result
            
            url = f"{ML_SERVICE_URL}{endpoint}"
            response = http_session.get(url, params=params, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                if use_cache and cache:
                    cache.set(cache_key, result, timeout=1800)
                return result
            else:
                logger.warning(f"ML service returned {response.status_code} for {endpoint}")
                return None
                
        except Exception as e:
            logger.warning(f"ML service call failed for {endpoint}: {e}")
            return None
    
    @staticmethod
    def call_ml_service_post(endpoint, data=None, timeout=30, use_cache=True):
        try:
            if not ML_SERVICE_URL:
                return None
            
            cache_key = f"ml_post:{endpoint}:{json.dumps(data, sort_keys=True)}"
            
            if use_cache and cache:
                cached_result = cache.get(cache_key)
                if cached_result:
                    logger.info(f"ML service POST cache hit for {endpoint}")
                    return cached_result
            
            url = f"{ML_SERVICE_URL}{endpoint}"
            response = http_session.post(url, json=data, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                if use_cache and cache:
                    cache.set(cache_key, result, timeout=1800)
                return result
            else:
                logger.warning(f"ML service POST returned {response.status_code} for {endpoint}")
                return None
                
        except Exception as e:
            logger.warning(f"ML service POST call failed for {endpoint}: {e}")
            return None
    
    @staticmethod
    def process_ml_recommendations(ml_response, limit=20):
        try:
            if not ml_response or 'recommendations' not in ml_response:
                return []
            
            recommendations = []
            ml_recs = ml_response['recommendations'][:limit]
            
            content_ids = []
            for rec in ml_recs:
                if isinstance(rec, dict) and 'content_id' in rec:
                    content_ids.append(rec['content_id'])
                elif isinstance(rec, int):
                    content_ids.append(rec)
            
            if not content_ids:
                return []
            
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            content_dict = {content.id: content for content in contents}
            
            for i, rec in enumerate(ml_recs):
                content_id = rec['content_id'] if isinstance(rec, dict) else rec
                content = content_dict.get(content_id)
                
                if content:
                    if not content.slug:
                        content.ensure_slug()
                    
                    content_data = {
                        'content': content,
                        'ml_score': rec.get('score', 0) if isinstance(rec, dict) else 0,
                        'ml_reason': rec.get('reason', '') if isinstance(rec, dict) else '',
                        'ml_rank': i + 1
                    }
                    recommendations.append(content_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error processing ML recommendations: {e}")
            return []
    
    @staticmethod
    def get_personalized_recommendations(user_id, user_data, limit=20):
        try:
            endpoint = "/api/recommendations"
            ml_response = MLServiceClient.call_ml_service_post(endpoint, user_data, timeout=30)
            
            if ml_response:
                return MLServiceClient.process_ml_recommendations(ml_response, limit)
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            return []
    
    @staticmethod
    def get_similar_content(content_id, limit=10):
        try:
            endpoint = f"/api/similar/{content_id}"
            params = {'limit': limit}
            ml_response = MLServiceClient.call_ml_service(endpoint, params)
            
            if ml_response:
                return MLServiceClient.process_ml_recommendations(ml_response, limit)
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting similar content: {e}")
            return []
    
    @staticmethod
    def get_trending_recommendations(limit=20, region='US'):
        try:
            endpoint = "/api/trending"
            params = {'limit': limit, 'region': region}
            ml_response = MLServiceClient.call_ml_service(endpoint, params)
            
            if ml_response:
                return MLServiceClient.process_ml_recommendations(ml_response, limit)
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []

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
            rating=data.get('rating')
        )
        
        db.session.add(interaction)
        db.session.commit()
        
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
        
        interactions = UserInteraction.query.filter_by(user_id=current_user.id).all()
        
        user_data = {
            'user_id': current_user.id,
            'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
            'preferred_genres': json.loads(current_user.preferred_genres or '[]'),
            'interactions': [
                {
                    'content_id': interaction.content_id,
                    'interaction_type': interaction.interaction_type,
                    'rating': interaction.rating,
                    'timestamp': interaction.timestamp.isoformat()
                }
                for interaction in interactions
            ]
        }
        
        ml_recommendations = MLServiceClient.get_personalized_recommendations(
            current_user.id, user_data, limit
        )
        
        if ml_recommendations:
            result = []
            for rec in ml_recommendations:
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
                    'ml_reason': rec['ml_reason'],
                    'ml_rank': rec['ml_rank'],
                    'recommendation_source': 'ml_service'
                })
            
            return jsonify({
                'recommendations': result,
                'total_interactions': len(interactions),
                'source': 'ml_service'
            }), 200
        
        return jsonify({
            'recommendations': [], 
            'source': 'fallback',
            'message': 'ML service unavailable'
        }), 200
        
    except Exception as e:
        logger.error(f"Personalized recommendations error: {e}")
        return jsonify({
            'recommendations': [], 
            'error': 'Failed to get recommendations'
        }), 200

@users_bp.route('/api/recommendations/ml-similar/<int:content_id>', methods=['GET'])
def get_ml_similar_recommendations(content_id):
    try:
        limit = int(request.args.get('limit', 10))
        
        ml_recommendations = MLServiceClient.get_similar_content(content_id, limit)
        
        if ml_recommendations:
            result = []
            for rec in ml_recommendations:
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
                    'ml_reason': rec['ml_reason'],
                    'similarity_score': rec['ml_score'],
                    'match_type': 'ml_enhanced'
                })
            
            return jsonify({
                'similar_content': result,
                'source': 'ml_service',
                'algorithm': 'ml_enhanced_similarity'
            }), 200
        
        return jsonify({
            'similar_content': [],
            'source': 'fallback',
            'message': 'ML service unavailable'
        }), 200
        
    except Exception as e:
        logger.error(f"ML similar recommendations error: {e}")
        return jsonify({
            'similar_content': [],
            'error': 'Failed to get similar recommendations'
        }), 500

@users_bp.route('/api/recommendations/ml-trending', methods=['GET'])
def get_ml_trending_recommendations():
    try:
        limit = int(request.args.get('limit', 20))
        region = request.args.get('region', 'IN')
        
        ml_recommendations = MLServiceClient.get_trending_recommendations(limit, region)
        
        if ml_recommendations:
            result = []
            for rec in ml_recommendations:
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
                    'ml_reason': rec['ml_reason'],
                    'trending_score': rec['ml_score']
                })
            
            return jsonify({
                'recommendations': result,
                'source': 'ml_service',
                'region': region,
                'algorithm': 'ml_enhanced_trending'
            }), 200
        
        return jsonify({
            'recommendations': [],
            'source': 'fallback',
            'message': 'ML service unavailable'
        }), 200
        
    except Exception as e:
        logger.error(f"ML trending recommendations error: {e}")
        return jsonify({
            'recommendations': [],
            'error': 'Failed to get trending recommendations'
        }), 500

@users_bp.route('/api/ml/health', methods=['GET'])
def check_ml_service_health():
    try:
        health_response = MLServiceClient.call_ml_service('/health', use_cache=False, timeout=5)
        
        if health_response:
            return jsonify({
                'ml_service_status': 'healthy',
                'ml_service_url': ML_SERVICE_URL,
                'response': health_response
            }), 200
        else:
            return jsonify({
                'ml_service_status': 'unhealthy',
                'ml_service_url': ML_SERVICE_URL,
                'error': 'No response from ML service'
            }), 503
            
    except Exception as e:
        logger.error(f"ML service health check error: {e}")
        return jsonify({
            'ml_service_status': 'error',
            'ml_service_url': ML_SERVICE_URL,
            'error': str(e)
        }), 503