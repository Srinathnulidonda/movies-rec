#backend/users.py
from flask import Blueprint, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from functools import wraps
import logging
import json
import jwt
import hashlib
import time
from collections import Counter
import requests

# Create users blueprint
users_bp = Blueprint('users', __name__, url_prefix='/api')

# Initialize logger
logger = logging.getLogger(__name__)

# Global variables (will be initialized from main app)
db = None
cache = None
app_instance = None
http_session = None
ML_SERVICE_URL = None

def init_users(app, database, cache_instance):
    """Initialize users module with app instance"""
    global db, cache, app_instance, http_session, ML_SERVICE_URL
    
    db = database
    cache = cache_instance
    app_instance = app
    ML_SERVICE_URL = app.config.get('ML_SERVICE_URL')
    
    # Create HTTP session
    http_session = create_http_session()
    
    logger.info("Users module initialized")

def create_http_session():
    """Create HTTP session with retry logic"""
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    
    session = requests.Session()
    retry = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 504)
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_session_id():
    """Get or create session ID for anonymous users"""
    if 'session_id' not in session:
        session['session_id'] = hashlib.md5(f"{request.remote_addr}{time.time()}".encode()).hexdigest()
    return session['session_id']

def get_user_location(ip_address):
    """Get user location from IP address"""
    cache_key = f"location:{ip_address}"
    cached_location = cache.get(cache_key)
    
    if cached_location:
        return cached_location
    
    try:
        response = http_session.get(f'http://ip-api.com/json/{ip_address}', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                location = {
                    'country': data.get('country'),
                    'region': data.get('regionName'),
                    'city': data.get('city'),
                    'lat': data.get('lat'),
                    'lon': data.get('lon')
                }
                cache.set(cache_key, location, timeout=86400)
                return location
    except Exception as e:
        logger.error(f"Error getting location for IP {ip_address}: {e}")
    return None

def require_auth(f):
    """Decorator to require user authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app_instance.secret_key, algorithms=['HS256'])
            
            # Import User model dynamically to avoid circular imports
            from app import User
            current_user = User.query.get(data['user_id'])
            
            if not current_user:
                return jsonify({'error': 'Invalid token'}), 401
            
            # Update last active
            current_user.last_active = datetime.utcnow()
            db.session.commit()
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return jsonify({'error': 'Authentication failed'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

class UserService:
    """Service class for user-related operations"""
    
    @staticmethod
    def create_token(user_id, expiry_days=30):
        """Create JWT token for user"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(days=expiry_days)
        }
        return jwt.encode(payload, app_instance.secret_key, algorithm='HS256')
    
    @staticmethod
    def verify_token(token):
        """Verify JWT token"""
        try:
            data = jwt.decode(token, app_instance.secret_key, algorithms=['HS256'])
            return data
        except:
            return None
    
    @staticmethod
    def format_user_data(user):
        """Format user data for response"""
        return {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'is_admin': user.is_admin,
            'preferred_languages': json.loads(user.preferred_languages or '[]'),
            'preferred_genres': json.loads(user.preferred_genres or '[]'),
            'location': user.location,
            'created_at': user.created_at.isoformat() if user.created_at else None,
            'last_active': user.last_active.isoformat() if user.last_active else None
        }

class InteractionService:
    """Service for handling user interactions"""
    
    @staticmethod
    def record_anonymous_interaction(session_id, content_id, interaction_type, ip_address):
        """Record interaction for anonymous users"""
        from app import AnonymousInteraction
        
        try:
            interaction = AnonymousInteraction(
                session_id=session_id,
                content_id=content_id,
                interaction_type=interaction_type,
                ip_address=ip_address,
                timestamp=datetime.utcnow()
            )
            db.session.add(interaction)
            db.session.commit()
            return True
        except Exception as e:
            logger.error(f"Error recording anonymous interaction: {e}")
            db.session.rollback()
            return False
    
    @staticmethod
    def get_user_interaction_stats(user_id):
        """Get user interaction statistics"""
        from app import UserInteraction, Content
        
        try:
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            
            stats = {
                'total_interactions': len(interactions),
                'watched': 0,
                'watchlist': 0,
                'favorites': 0,
                'rated': 0,
                'average_rating': 0,
                'genres_watched': {},
                'content_types': {'movie': 0, 'tv': 0, 'anime': 0}
            }
            
            total_rating = 0
            rating_count = 0
            
            for interaction in interactions:
                # Count by type
                if interaction.interaction_type == 'view':
                    stats['watched'] += 1
                elif interaction.interaction_type == 'watchlist':
                    stats['watchlist'] += 1
                elif interaction.interaction_type == 'favorite':
                    stats['favorites'] += 1
                
                # Rating stats
                if interaction.rating:
                    stats['rated'] += 1
                    total_rating += interaction.rating
                    rating_count += 1
                
                # Get content details for genre stats
                content = Content.query.get(interaction.content_id)
                if content:
                    # Content type stats
                    if content.content_type in stats['content_types']:
                        stats['content_types'][content.content_type] += 1
                    
                    # Genre stats
                    if content.genres:
                        genres = json.loads(content.genres)
                        for genre in genres:
                            stats['genres_watched'][genre] = stats['genres_watched'].get(genre, 0) + 1
            
            if rating_count > 0:
                stats['average_rating'] = round(total_rating / rating_count, 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting user interaction stats: {e}")
            return None

class RecommendationService:
    """Service for user-specific recommendations"""
    
    @staticmethod
    def get_personalized_recommendations(user, limit=20):
        """Get personalized recommendations for user"""
        from app import UserInteraction, Content, MLServiceClient, RecommendationEngine
        
        try:
            interactions = UserInteraction.query.filter_by(user_id=user.id).all()
            
            user_data = {
                'user_id': user.id,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]'),
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
            
            # Try ML service first
            try:
                response = http_session.post(f"{ML_SERVICE_URL}/api/recommendations", 
                                           json=user_data, timeout=30)
                
                if response.status_code == 200:
                    ml_recommendations = response.json().get('recommendations', [])
                    
                    content_ids = [rec['content_id'] for rec in ml_recommendations]
                    contents = Content.query.filter(Content.id.in_(content_ids)).all()
                    
                    result = []
                    content_dict = {content.id: content for content in contents}
                    
                    for rec in ml_recommendations:
                        content = content_dict.get(rec['content_id'])
                        if content:
                            result.append({
                                'content': content,
                                 'score': rec.get('score', 0),
                                'reason': rec.get('reason', '')
                            })
                    
                    return result[:limit]
            except Exception as e:
                logger.warning(f"ML service failed, using fallback: {e}")
            
            # Fallback to content-based recommendations
            return RecommendationService.get_content_based_recommendations(user, limit)
            
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            return []
    
    @staticmethod
    def get_content_based_recommendations(user, limit=20):
        """Get content-based recommendations based on user preferences"""
        from app import Content, UserInteraction, RecommendationEngine
        
        try:
            # Get user's interaction history
            interactions = UserInteraction.query.filter_by(user_id=user.id).all()
            
            if not interactions:
                # No history, use preferences
                preferred_genres = json.loads(user.preferred_genres or '[]')
                preferred_languages = json.loads(user.preferred_languages or '[]')
                
                recommendations = []
                for genre in preferred_genres[:3]:
                    genre_recs = RecommendationEngine.get_genre_recommendations(genre, limit=7)
                    recommendations.extend(genre_recs)
                
                return recommendations[:limit]
            
            # Analyze user's content preferences
            viewed_content_ids = [i.content_id for i in interactions]
            viewed_contents = Content.query.filter(Content.id.in_(viewed_content_ids)).all()
            
            # Extract genres and languages
            all_genres = []
            all_languages = []
            for content in viewed_contents:
                if content.genres:
                    all_genres.extend(json.loads(content.genres))
                if content.languages:
                    all_languages.extend(json.loads(content.languages))
            
            # Get most common genres and languages
            genre_counts = Counter(all_genres)
            language_counts = Counter(all_languages)
            
            top_genres = [genre for genre, _ in genre_counts.most_common(3)]
            top_languages = [lang for lang, _ in language_counts.most_common(2)]
            
            recommendations = []
            
            # Get recommendations based on top genres
            for genre in top_genres:
                genre_recs = RecommendationEngine.get_genre_recommendations(genre, limit=10)
                recommendations.extend(genre_recs)
            
            # Filter out already viewed content
            recommendations = [rec for rec in recommendations 
                             if rec.id not in viewed_content_ids]
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting content-based recommendations: {e}")
            return []
    
    @staticmethod
    def get_anonymous_recommendations(session_id, ip_address, limit=20):
        """Get recommendations for anonymous users"""
        from app import AnonymousInteraction, Content, RecommendationEngine
        
        try:
            location = get_user_location(ip_address)
            interactions = AnonymousInteraction.query.filter_by(session_id=session_id).all()
            
            recommendations = []
            
            if interactions:
                # User has some interaction history
                viewed_content_ids = [interaction.content_id for interaction in interactions]
                viewed_contents = Content.query.filter(Content.id.in_(viewed_content_ids)).all()
                
                # Extract genres from viewed content
                all_genres = []
                for content in viewed_contents:
                    if content.genres:
                        all_genres.extend(json.loads(content.genres))
                
                genre_counts = Counter(all_genres)
                top_genres = [genre for genre, _ in genre_counts.most_common(3)]
                
                # Get recommendations based on genres
                for genre in top_genres:
                    genre_recs = RecommendationEngine.get_genre_recommendations(genre, limit=7)
                    recommendations.extend(genre_recs)
            
            # Add regional recommendations if location is available
            if location and location.get('country') == 'India':
                regional_recs = RecommendationEngine.get_regional_recommendations('hindi', limit=5)
                recommendations.extend(regional_recs)
            
            # Add trending to fill remaining slots
            trending_recs = RecommendationEngine.get_trending_recommendations(limit=10)
            recommendations.extend(trending_recs)
            
            # Remove duplicates and limit
            seen_ids = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec.id not in seen_ids:
                    seen_ids.add(rec.id)
                    unique_recommendations.append(rec)
                    if len(unique_recommendations) >= limit:
                        break
            
            return unique_recommendations
            
        except Exception as e:
            logger.error(f"Error getting anonymous recommendations: {e}")
            return []

# User Routes

@users_bp.route('/register', methods=['POST'])
def register():
    """User registration"""
    try:
        from app import User
        
        data = request.get_json()
        
        # Validate required fields
        if not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if username exists
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        # Check if email exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        # Validate email format
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, data['email']):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Validate password strength
        if len(data['password']) < 8:
            return jsonify({'error': 'Password must be at least 8 characters long'}), 400
        
        # Create new user
        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=generate_password_hash(data['password']),
            preferred_languages=json.dumps(data.get('preferred_languages', [])),
            preferred_genres=json.dumps(data.get('preferred_genres', [])),
            location=data.get('location'),
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow()
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Create token
        token = UserService.create_token(user.id)
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': UserService.format_user_data(user)
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@users_bp.route('/login', methods=['POST'])
def login():
    """User login"""
    try:
        from app import User
        
        data = request.get_json()
        
        if not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Missing username or password'}), 400
        
        # Find user by username or email
        user = User.query.filter(
            (User.username == data['username']) | 
            (User.email == data['username'])
        ).first()
        
        if not user or not check_password_hash(user.password_hash, data['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Update last active
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        # Create token
        token = UserService.create_token(user.id)
        
        # Track login location
        if request.remote_addr:
            location = get_user_location(request.remote_addr)
            if location and location.get('city'):
                user.location = f"{location['city']}, {location['country']}"
                db.session.commit()
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': UserService.format_user_data(user)
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@users_bp.route('/user/profile', methods=['GET'])
@require_auth
def get_user_profile(current_user):
    """Get user profile with statistics"""
    try:
        user_data = UserService.format_user_data(current_user)
        stats = InteractionService.get_user_interaction_stats(current_user.id)
        
        return jsonify({
            'user': user_data,
            'statistics': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Profile error: {e}")
        return jsonify({'error': 'Failed to get profile'}), 500

@users_bp.route('/user/profile', methods=['PUT'])
@require_auth
def update_user_profile(current_user):
    """Update user profile"""
    try:
        data = request.get_json()
        
        # Update allowed fields
        if 'email' in data:
            # Check if email is already taken
            from app import User
            existing = User.query.filter_by(email=data['email']).first()
            if existing and existing.id != current_user.id:
                return jsonify({'error': 'Email already in use'}), 400
            current_user.email = data['email']
        
        if 'preferred_languages' in data:
            current_user.preferred_languages = json.dumps(data['preferred_languages'])
        
        if 'preferred_genres' in data:
            current_user.preferred_genres = json.dumps(data['preferred_genres'])
        
        if 'location' in data:
            current_user.location = data['location']
        
        db.session.commit()
        
        return jsonify({
            'message': 'Profile updated successfully',
            'user': UserService.format_user_data(current_user)
        }), 200
        
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update profile'}), 500

@users_bp.route('/user/change-password', methods=['POST'])
@require_auth
def change_password(current_user):
    """Change user password"""
    try:
        data = request.get_json()
        
        if not data.get('current_password') or not data.get('new_password'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Verify current password
        if not check_password_hash(current_user.password_hash, data['current_password']):
            return jsonify({'error': 'Current password is incorrect'}), 401
        
        # Validate new password
        if len(data['new_password']) < 8:
            return jsonify({'error': 'New password must be at least 8 characters long'}), 400
        
        # Update password
        current_user.password_hash = generate_password_hash(data['new_password'])
        db.session.commit()
        
        # Create new token
        token = UserService.create_token(current_user.id)
        
        return jsonify({
            'message': 'Password changed successfully',
            'token': token
        }), 200
        
    except Exception as e:
        logger.error(f"Password change error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to change password'}), 500

@users_bp.route('/interactions', methods=['POST'])
@require_auth
def record_interaction(current_user):
    """Record user interaction with content"""
    try:
        from app import UserInteraction, Content
        
        data = request.get_json()
        
        required_fields = ['content_id', 'interaction_type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Validate content exists
        content = Content.query.get(data['content_id'])
        if not content:
            return jsonify({'error': 'Content not found'}), 404
        
        # Check for existing interaction of same type
        existing = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=data['content_id'],
            interaction_type=data['interaction_type']
        ).first()
        
        if existing:
            # Update existing interaction
            if 'rating' in data:
                existing.rating = data['rating']
            existing.timestamp = datetime.utcnow()
            message = 'Interaction updated successfully'
        else:
            # Create new interaction
            interaction = UserInteraction(
                user_id=current_user.id,
                content_id=data['content_id'],
                interaction_type=data['interaction_type'],
                rating=data.get('rating'),
                timestamp=datetime.utcnow()
            )
            db.session.add(interaction)
            message = 'Interaction recorded successfully'
        
        db.session.commit()
        
        return jsonify({'message': message}), 201
        
    except Exception as e:
        logger.error(f"Interaction recording error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record interaction'}), 500

@users_bp.route('/user/watchlist', methods=['GET'])
@require_auth
def get_watchlist(current_user):
    """Get user's watchlist"""
    try:
        from app import UserInteraction, Content
        
        watchlist_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='watchlist'
        ).order_by(UserInteraction.timestamp.desc()).all()
        
        content_ids = [interaction.content_id for interaction in watchlist_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        
        # Create a map for ordering
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
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'youtube_trailer': youtube_url,
                    'added_at': interaction.timestamp.isoformat()
                })
        
        return jsonify({
            'watchlist': result,
            'total': len(result)
        }), 200
        
    except Exception as e:
        logger.error(f"Watchlist error: {e}")
        return jsonify({'error': 'Failed to get watchlist'}), 500

@users_bp.route('/user/watchlist/<int:content_id>', methods=['DELETE'])
@require_auth
def remove_from_watchlist(current_user, content_id):
    """Remove content from watchlist"""
    try:
        from app import UserInteraction
        
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='watchlist'
        ).first()
        
        if not interaction:
            return jsonify({'error': 'Content not in watchlist'}), 404
        
        db.session.delete(interaction)
        db.session.commit()
        
        return jsonify({'message': 'Removed from watchlist'}), 200
        
    except Exception as e:
        logger.error(f"Remove from watchlist error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to remove from watchlist'}), 500

@users_bp.route('/user/favorites', methods=['GET'])
@require_auth
def get_favorites(current_user):
    """Get user's favorite content"""
    try:
        from app import UserInteraction, Content
        
        favorite_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='favorite'
        ).order_by(UserInteraction.timestamp.desc()).all()
        
        content_ids = [interaction.content_id for interaction in favorite_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        
        # Create a map for ordering
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
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'youtube_trailer': youtube_url,
                    'user_rating': interaction.rating,
                    'added_at': interaction.timestamp.isoformat()
                })
        
        return jsonify({
            'favorites': result,
            'total': len(result)
        }), 200
        
    except Exception as e:
        logger.error(f"Favorites error: {e}")
        return jsonify({'error': 'Failed to get favorites'}), 500

@users_bp.route('/user/favorites/<int:content_id>', methods=['DELETE'])
@require_auth
def remove_from_favorites(current_user, content_id):
    """Remove content from favorites"""
    try:
        from app import UserInteraction
        
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='favorite'
        ).first()
        
        if not interaction:
            return jsonify({'error': 'Content not in favorites'}), 404
        
        db.session.delete(interaction)
        db.session.commit()
        
        return jsonify({'message': 'Removed from favorites'}), 200
        
    except Exception as e:
        logger.error(f"Remove from favorites error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to remove from favorites'}), 500

@users_bp.route('/user/history', methods=['GET'])
@require_auth
def get_watch_history(current_user):
    """Get user's watch history"""
    try:
        from app import UserInteraction, Content
        
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        history = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='view'
        ).order_by(UserInteraction.timestamp.desc())\
         .paginate(page=page, per_page=per_page, error_out=False)
        
        content_ids = [interaction.content_id for interaction in history.items]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        
        content_map = {content.id: content for content in contents}
        
        result = []
        for interaction in history.items:
            content = content_map.get(interaction.content_id)
            if content:
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'watched_at': interaction.timestamp.isoformat(),
                    'user_rating': interaction.rating
                })
        
        return jsonify({
            'history': result,
            'total': history.total,
            'pages': history.pages,
            'current_page': page
        }), 200
        
    except Exception as e:
        logger.error(f"Watch history error: {e}")
        return jsonify({'error': 'Failed to get watch history'}), 500

@users_bp.route('/recommendations/personalized', methods=['GET'])
@require_auth
def get_personalized_recommendations(current_user):
    """Get personalized recommendations for logged-in user"""
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationService.get_personalized_recommendations(
            current_user, limit
        )
        
        result = []
        for rec in recommendations:
            content = rec.get('content') if isinstance(rec, dict) else rec
            
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url,
                'recommendation_score': rec.get('score', 0) if isinstance(rec, dict) else 0,
                'recommendation_reason': rec.get('reason', '') if isinstance(rec, dict) else 'Based on your preferences'
            })
        
        return jsonify({
            'recommendations': result,
            'personalization_level': 'high' if len(result) > 10 else 'medium'
        }), 200
        
    except Exception as e:
        logger.error(f"Personalized recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@users_bp.route('/recommendations/anonymous', methods=['GET'])
def get_anonymous_recommendations():
    """Get recommendations for anonymous users"""
    try:
        session_id = get_session_id()
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationService.get_anonymous_recommendations(
            session_id, request.remote_addr, limit
        )
        
        # Record anonymous view
        if recommendations:
            InteractionService.record_anonymous_interaction(
                session_id, 
                recommendations[0].id if recommendations else None,
                'recommendation_view',
                request.remote_addr
            )
        
        result = []
        for content in recommendations:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url
            })
        
        return jsonify({
            'recommendations': result,
            'session_id': session_id,
            'personalization_level': 'low'
        }), 200
        
    except Exception as e:
        logger.error(f"Anonymous recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@users_bp.route('/user/delete-account', methods=['DELETE'])
@require_auth
def delete_account(current_user):
    """Delete user account and all associated data"""
    try:
        from app import UserInteraction
        
        # Verify password for security
        data = request.get_json()
        if not data.get('password'):
            return jsonify({'error': 'Password required for account deletion'}), 400
        
        if not check_password_hash(current_user.password_hash, data['password']):
            return jsonify({'error': 'Invalid password'}), 401
        
        # Delete all user interactions
        UserInteraction.query.filter_by(user_id=current_user.id).delete()
        
        # Delete user account
        db.session.delete(current_user)
        db.session.commit()
        
        return jsonify({'message': 'Account deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Account deletion error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to delete account'}), 500

@users_bp.route('/verify-token', methods=['POST'])
def verify_token():
    """Verify if a token is valid"""
    try:
        data = request.get_json()
        token = data.get('token')
        
        if not token:
            return jsonify({'valid': False, 'error': 'No token provided'}), 400
        
        token_data = UserService.verify_token(token)
        if token_data:
            from app import User
            user = User.query.get(token_data['user_id'])
            
            if user:
                return jsonify({
                    'valid': True,
                    'user': UserService.format_user_data(user)
                }), 200
        
        return jsonify({'valid': False, 'error': 'Invalid token'}), 401
        
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return jsonify({'valid': False, 'error': 'Verification failed'}), 500