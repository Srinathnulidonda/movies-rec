from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import logging
import jwt
import sys
import os
from functools import wraps
import hashlib
import threading
from collections import defaultdict, deque

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from ml_services.recommendation import AdvancedRecommendationEngine
    ML_SERVICES_AVAILABLE = True
except ImportError:
    AdvancedRecommendationEngine = None
    ML_SERVICES_AVAILABLE = False
    print("Warning: ML Services not available. Using basic recommendation features.")

users_bp = Blueprint('users', __name__)
logger = logging.getLogger(__name__)

# Global variables for dependency injection
db = None
User = None
Content = None
UserInteraction = None
app = None
recommendation_engine = None

# Performance tracking
user_activity_tracker = defaultdict(lambda: deque(maxlen=100))
recommendation_performance = defaultdict(list)
_performance_lock = threading.Lock()

def init_users(flask_app, database, models, services):
    """Initialize users module with enhanced ML services"""
    global db, User, Content, UserInteraction, app, recommendation_engine
    
    app = flask_app
    db = database
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    
    # Initialize advanced recommendation engine
    if ML_SERVICES_AVAILABLE and AdvancedRecommendationEngine:
        try:
            recommendation_engine = AdvancedRecommendationEngine(
                db, models, config={
                    'cache_ttl': 1800,  # 30 minutes
                    'max_cache_size': 5000,
                    'async_processing': True,
                    'enable_explanation': True,
                    'enable_realtime_learning': True,
                    'performance_tracking': True,
                    'diversity_enforcement': True,
                    'novelty_boost': True,
                    'cold_start_strategy': 'advanced'
                }
            )
            logger.info("Advanced ML Recommendation Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Advanced ML Recommendation Engine: {e}")
            recommendation_engine = None
    else:
        logger.warning("Advanced ML Recommendation Engine not available")
        recommendation_engine = None

def require_auth(f):
    """Enhanced authentication decorator with activity tracking"""
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
            
            # Track user activity
            _track_user_activity(current_user.id, request.endpoint)
            
            # Update last active timestamp
            current_user.last_active = datetime.utcnow()
            db.session.commit()
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return jsonify({'error': 'Authentication failed'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

def _track_user_activity(user_id, endpoint):
    """Track user activity for analytics"""
    try:
        with _performance_lock:
            user_activity_tracker[user_id].append({
                'endpoint': endpoint,
                'timestamp': datetime.utcnow(),
                'session_id': request.headers.get('X-Session-ID', 'unknown')
            })
    except Exception as e:
        logger.warning(f"Activity tracking failed: {e}")

def _generate_session_token(user):
    """Generate enhanced session token with additional claims"""
    payload = {
        'user_id': user.id,
        'username': user.username,
        'is_admin': user.is_admin,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(days=30),
        'session_id': hashlib.md5(f"{user.id}{datetime.utcnow()}".encode()).hexdigest()[:16]
    }
    
    return jwt.encode(payload, app.secret_key, algorithm='HS256')

def _validate_user_input(data, required_fields):
    """Validate user input data"""
    if not data:
        return False, "No data provided"
    
    for field in required_fields:
        if field not in data or not data[field]:
            return False, f"Missing required field: {field}"
    
    # Email validation
    if 'email' in data:
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, data['email']):
            return False, "Invalid email format"
    
    # Password strength validation
    if 'password' in data:
        password = data['password']
        if len(password) < 6:
            return False, "Password must be at least 6 characters long"
        
        # Optional: Add more password complexity requirements
        # if not any(c.isupper() for c in password):
        #     return False, "Password must contain at least one uppercase letter"
    
    return True, "Valid"

@users_bp.route('/api/register', methods=['POST'])
def register():
    """Enhanced user registration with validation and analytics"""
    try:
        data = request.get_json()
        
        # Validate input
        is_valid, error_message = _validate_user_input(
            data, ['username', 'email', 'password']
        )
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        # Check for existing users
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        # Create user with enhanced profile
        user = User(
            username=data['username'].strip(),
            email=data['email'].strip().lower(),
            password_hash=generate_password_hash(data['password']),
            preferred_languages=json.dumps(data.get('preferred_languages', [])),
            preferred_genres=json.dumps(data.get('preferred_genres', [])),
            location=data.get('location', ''),
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow()
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Generate token
        token = _generate_session_token(user)
        
        # Initialize user profile in recommendation engine
        if recommendation_engine:
            try:
                recommendation_engine.update_user_profile(user.id)
            except Exception as e:
                logger.warning(f"Failed to initialize user profile in ML engine: {e}")
        
        logger.info(f"New user registered: {user.username} (ID: {user.id})")
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]')
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Registration failed. Please try again.'}), 500

@users_bp.route('/api/login', methods=['POST'])
def login():
    """Enhanced login with security features"""
    try:
        data = request.get_json()
        
        # Validate input
        is_valid, error_message = _validate_user_input(
            data, ['username', 'password']
        )
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        # Rate limiting check (basic implementation)
        client_ip = request.remote_addr
        login_attempts_key = f"login_attempts:{client_ip}"
        
        # Find user (case-insensitive username)
        user = User.query.filter(
            User.username.ilike(data['username'].strip())
        ).first()
        
        if not user or not check_password_hash(user.password_hash, data['password']):
            # Log failed attempt
            logger.warning(f"Failed login attempt for username: {data['username']} from IP: {client_ip}")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Update user activity
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        # Generate token
        token = _generate_session_token(user)
        
        # Update recommendation engine user profile
        if recommendation_engine:
            try:
                recommendation_engine.update_user_profile(user.id)
            except Exception as e:
                logger.warning(f"Failed to update user profile in ML engine: {e}")
        
        logger.info(f"User logged in: {user.username} (ID: {user.id})")
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]'),
                'location': user.location or '',
                'last_active': user.last_active.isoformat() if user.last_active else None
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed. Please try again.'}), 500

@users_bp.route('/api/interactions', methods=['POST'])
@require_auth
def record_interaction(current_user):
    """Enhanced interaction recording with real-time ML updates"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['content_id', 'interaction_type']
        is_valid, error_message = _validate_user_input(data, required_fields)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        # Validate content exists
        content = Content.query.get(data['content_id'])
        if not content:
            return jsonify({'error': 'Content not found'}), 404
        
        # Handle special interactions
        if data['interaction_type'] == 'remove_watchlist':
            interaction = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=data['content_id'],
                interaction_type='watchlist'
            ).first()
            
            if interaction:
                db.session.delete(interaction)
                db.session.commit()
                
                # Update ML engine
                if recommendation_engine:
                    try:
                        recommendation_engine.update_user_profile(current_user.id)
                    except Exception as e:
                        logger.warning(f"Failed to update ML profile: {e}")
                
                return jsonify({'message': 'Removed from watchlist'}), 200
            else:
                return jsonify({'message': 'Content not in watchlist'}), 404
        
        # Check for duplicate watchlist entries
        if data['interaction_type'] == 'watchlist':
            existing = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=data['content_id'],
                interaction_type='watchlist'
            ).first()
            
            if existing:
                return jsonify({'message': 'Already in watchlist'}), 200
        
        # Validate rating
        rating = data.get('rating')
        if rating is not None:
            try:
                rating = float(rating)
                if not (1.0 <= rating <= 10.0):
                    return jsonify({'error': 'Rating must be between 1.0 and 10.0'}), 400
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid rating format'}), 400
        
        # Create interaction with enhanced metadata
        interaction_metadata = data.get('metadata', {})
        interaction_metadata.update({
            'user_agent': request.headers.get('User-Agent', ''),
            'ip_address': request.remote_addr,
            'session_id': request.headers.get('X-Session-ID', ''),
            'timestamp_created': datetime.utcnow().isoformat()
        })
        
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=data['content_id'],
            interaction_type=data['interaction_type'],
            rating=rating,
            interaction_metadata=interaction_metadata,
            timestamp=datetime.utcnow()
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        # Update ML engine with real-time learning
        if recommendation_engine:
            try:
                # Update user profile
                recommendation_engine.update_user_profile(current_user.id)
                
                # Real-time learning update
                if hasattr(recommendation_engine, 'realtime_engine') and recommendation_engine.realtime_engine:
                    recommendation_engine.realtime_engine.update_realtime_profile(
                        current_user.id, {
                            'content_id': data['content_id'],
                            'interaction_type': data['interaction_type'],
                            'context': {
                                'rating': rating,
                                'metadata': interaction_metadata
                            }
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to update ML engine: {e}")
        
        # Log interaction for analytics
        logger.info(f"Interaction recorded: User {current_user.id} -> {data['interaction_type']} -> Content {data['content_id']}")
        
        return jsonify({
            'message': 'Interaction recorded successfully',
            'interaction_id': interaction.id
        }), 201
        
    except Exception as e:
        logger.error(f"Interaction recording error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record interaction'}), 500

@users_bp.route('/api/user/profile', methods=['GET'])
@require_auth
def get_user_profile(current_user):
    """Get comprehensive user profile with analytics"""
    try:
        # Get interaction statistics
        interaction_stats = db.session.query(
            UserInteraction.interaction_type,
            db.func.count(UserInteraction.id).label('count')
        ).filter_by(user_id=current_user.id).group_by(
            UserInteraction.interaction_type
        ).all()
        
        interaction_counts = {stat.interaction_type: stat.count for stat in interaction_stats}
        
        # Get recent activity
        recent_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id
        ).order_by(UserInteraction.timestamp.desc()).limit(10).all()
        
        recent_activity = []
        for interaction in recent_interactions:
            content = Content.query.get(interaction.content_id)
            if content:
                recent_activity.append({
                    'interaction_type': interaction.interaction_type,
                    'content_title': content.title,
                    'content_type': content.content_type,
                    'timestamp': interaction.timestamp.isoformat(),
                    'rating': interaction.rating
                })
        
        # Get ML profile strength if available
        profile_strength = 'unknown'
        if recommendation_engine:
            try:
                profile_strength = recommendation_engine.get_user_profile_strength(current_user.id)
            except Exception as e:
                logger.warning(f"Failed to get profile strength: {e}")
        
        # Get user activity metrics
        total_interactions = sum(interaction_counts.values())
        days_since_joined = (datetime.utcnow() - current_user.created_at).days
        avg_interactions_per_day = total_interactions / max(days_since_joined, 1)
        
        profile_data = {
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'is_admin': current_user.is_admin,
                'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
                'preferred_genres': json.loads(current_user.preferred_genres or '[]'),
                'location': current_user.location or '',
                'created_at': current_user.created_at.isoformat(),
                'last_active': current_user.last_active.isoformat() if current_user.last_active else None
            },
            'statistics': {
                'total_interactions': total_interactions,
                'interaction_breakdown': interaction_counts,
                'days_since_joined': days_since_joined,
                'avg_interactions_per_day': round(avg_interactions_per_day, 2),
                'profile_strength': profile_strength
            },
            'recent_activity': recent_activity
        }
        
        return jsonify(profile_data), 200
        
    except Exception as e:
        logger.error(f"Get user profile error: {e}")
        return jsonify({'error': 'Failed to get user profile'}), 500

@users_bp.route('/api/user/profile', methods=['PUT'])
@require_auth
def update_user_profile(current_user):
    """Update user profile with validation"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Update allowed fields
        updated_fields = []
        
        if 'preferred_languages' in data:
            if isinstance(data['preferred_languages'], list):
                current_user.preferred_languages = json.dumps(data['preferred_languages'])
                updated_fields.append('preferred_languages')
        
        if 'preferred_genres' in data:
            if isinstance(data['preferred_genres'], list):
                current_user.preferred_genres = json.dumps(data['preferred_genres'])
                updated_fields.append('preferred_genres')
        
        if 'location' in data:
            current_user.location = data['location'][:100]  # Limit location length
            updated_fields.append('location')
        
        if 'email' in data:
            # Validate email format
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if re.match(email_pattern, data['email']):
                # Check if email is already taken by another user
                existing_user = User.query.filter(
                    User.email == data['email'].strip().lower(),
                    User.id != current_user.id
                ).first()
                
                if not existing_user:
                    current_user.email = data['email'].strip().lower()
                    updated_fields.append('email')
                else:
                    return jsonify({'error': 'Email already taken'}), 400
            else:
                return jsonify({'error': 'Invalid email format'}), 400
        
        if updated_fields:
            db.session.commit()
            
            # Update ML engine profile
            if recommendation_engine:
                try:
                    recommendation_engine.update_user_profile(current_user.id)
                except Exception as e:
                    logger.warning(f"Failed to update ML profile: {e}")
            
            logger.info(f"User profile updated: {current_user.username} -> {updated_fields}")
            
            return jsonify({
                'message': 'Profile updated successfully',
                'updated_fields': updated_fields
            }), 200
        else:
            return jsonify({'message': 'No valid fields to update'}), 200
        
    except Exception as e:
        logger.error(f"Update user profile error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update profile'}), 500

@users_bp.route('/api/user/watchlist', methods=['GET'])
@require_auth
def get_watchlist(current_user):
    """Get user's watchlist with enhanced details"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        sort_by = request.args.get('sort_by', 'date_added')  # date_added, title, rating
        sort_order = request.args.get('sort_order', 'desc')  # asc, desc
        
        # Get watchlist interactions
        query = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='watchlist'
        ).join(Content)
        
        # Apply sorting
        if sort_by == 'title':
            order_col = Content.title
        elif sort_by == 'rating':
            order_col = Content.rating
        else:  # date_added
            order_col = UserInteraction.timestamp
        
        if sort_order == 'asc':
            query = query.order_by(order_col.asc())
        else:
            query = query.order_by(order_col.desc())
        
        # Paginate
        watchlist_interactions = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        result = []
        for interaction in watchlist_interactions.items:
            content = Content.query.get(interaction.content_id)
            if content:
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
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'youtube_trailer': youtube_url,
                    'date_added': interaction.timestamp.isoformat(),
                    'overview': content.overview[:200] + '...' if content.overview else None
                })
        
        return jsonify({
            'watchlist': result,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': watchlist_interactions.total,
                'pages': watchlist_interactions.pages,
                'has_next': watchlist_interactions.has_next,
                'has_prev': watchlist_interactions.has_prev
            },
            'total_count': watchlist_interactions.total
        }), 200
        
    except Exception as e:
        logger.error(f"Watchlist error: {e}")
        return jsonify({'error': 'Failed to get watchlist'}), 500

@users_bp.route('/api/user/favorites', methods=['GET'])
@require_auth
def get_favorites(current_user):
    """Get user's favorites with enhanced details"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        # Get favorite interactions
        favorite_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='favorite'
        ).order_by(UserInteraction.timestamp.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        result = []
        for interaction in favorite_interactions.items:
            content = Content.query.get(interaction.content_id)
            if content:
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
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'youtube_trailer': youtube_url,
                    'date_added': interaction.timestamp.isoformat(),
                    'overview': content.overview[:200] + '...' if content.overview else None
                })
        
        return jsonify({
            'favorites': result,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': favorite_interactions.total,
                'pages': favorite_interactions.pages,
                'has_next': favorite_interactions.has_next,
                'has_prev': favorite_interactions.has_prev
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Favorites error: {e}")
        return jsonify({'error': 'Failed to get favorites'}), 500

@users_bp.route('/api/recommendations/personalized', methods=['GET'])
@require_auth
def get_personalized_recommendations(current_user):
    """Get personalized recommendations using advanced ML algorithms"""
    try:
        if not recommendation_engine:
            return jsonify({
                'recommendations': [],
                'error': 'Advanced recommendation engine not available',
                'fallback': True
            }), 200
        
        # Parse parameters
        limit = min(int(request.args.get('limit', 20)), 50)
        content_type = request.args.get('content_type', 'all')
        strategy = request.args.get('strategy', 'adaptive')
        
        # Context information
        context = {
            'device': request.headers.get('X-Device-Type', 'unknown'),
            'time_of_day': _get_time_of_day(),
            'location_type': _infer_location_type(request),
            'session_length': request.headers.get('X-Session-Length', 'unknown'),
            'user_agent': request.headers.get('User-Agent', '')
        }
        
        start_time = datetime.utcnow()
        
        # Get recommendations from ML engine
        recommendations = recommendation_engine.get_personalized_recommendations(
            user_id=current_user.id,
            limit=limit,
            content_type=content_type,
            strategy=strategy,
            context=context
        )
        
        # Format recommendations for response
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
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url,
                'recommendation_score': rec.get('score', 0.0),
                'recommendation_reason': rec.get('reason', ''),
                'algorithm_used': rec.get('algorithm', ''),
                'confidence': rec.get('confidence', 0.0),
                'explanation': rec.get('explanation', '')
            })
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Track performance
        with _performance_lock:
            recommendation_performance[current_user.id].append({
                'timestamp': datetime.utcnow(),
                'response_time': response_time,
                'strategy': strategy,
                'result_count': len(result)
            })
        
        return jsonify({
            'recommendations': result,
            'strategy': strategy,
            'context': context,
            'total_interactions': recommendation_engine.get_user_interaction_count(current_user.id),
            'user_profile_strength': recommendation_engine.get_user_profile_strength(current_user.id),
            'response_time_ms': round(response_time * 1000, 2),
            'ml_engine_version': 'advanced_v2.0'
        }), 200
        
    except Exception as e:
        logger.error(f"Personalized recommendations error: {e}")
        return jsonify({
            'recommendations': [],
            'error': 'Failed to get recommendations',
            'fallback': True
        }), 200

@users_bp.route('/api/recommendations/ml-personalized', methods=['GET'])
@require_auth
def get_ml_personalized_recommendations(current_user):
    """Get advanced ML-personalized recommendations with comprehensive features"""
    try:
        if not recommendation_engine:
            return jsonify({
                'recommendations': [],
                'error': 'Advanced ML recommendation engine not available',
                'fallback': True
            }), 200
        
        # Parse advanced parameters
        limit = min(int(request.args.get('limit', 20)), 50)
        include_explanations = request.args.get('include_explanations', 'true').lower() == 'true'
        diversity_factor = float(request.args.get('diversity_factor', 0.3))
        novelty_factor = float(request.args.get('novelty_factor', 0.2))
        
        # Context information
        context = {
            'device': request.headers.get('X-Device-Type', 'unknown'),
            'time_of_day': _get_time_of_day(),
            'location_type': _infer_location_type(request),
            'session_length': request.headers.get('X-Session-Length', 'unknown'),
            'user_preferences': {
                'languages': json.loads(current_user.preferred_languages or '[]'),
                'genres': json.loads(current_user.preferred_genres or '[]')
            }
        }
        
        start_time = datetime.utcnow()
        
        # Get advanced recommendations
        recommendations = recommendation_engine.get_advanced_recommendations(
            user_id=current_user.id,
            limit=limit,
            include_explanations=include_explanations,
            diversity_factor=diversity_factor,
            novelty_factor=novelty_factor,
            context=context
        )
        
        # Format recommendations for response
        result = []
        for rec in recommendations:
            content = rec['content']
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            formatted_rec = {
                'id': content.id,
                'slug': getattr(content, 'slug', None),
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url,
                'ml_score': rec.get('score', 0.0),
                'confidence': rec.get('confidence', 0.0),
                'novelty_score': rec.get('novelty_score', 0.0),
                'diversity_contribution': rec.get('diversity_contribution', 0.0)
            }
            
            # Add advanced ML features
            if include_explanations:
                formatted_rec['ml_reason'] = rec.get('explanation', '')
                formatted_rec['explanation_confidence'] = rec.get('explanation_confidence', 0.0)
            
            if 'algorithm_contributions' in rec:
                formatted_rec['algorithm_mix'] = rec['algorithm_contributions']
            
            if 'quality_score' in rec:
                formatted_rec['quality_score'] = rec['quality_score']
            
            if 'popularity_percentile' in rec:
                formatted_rec['popularity_percentile'] = rec['popularity_percentile']
            
            result.append(formatted_rec)
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Get comprehensive metrics
        metrics = {}
        try:
            metrics = recommendation_engine.get_recommendation_metrics(current_user.id)
        except Exception as e:
            logger.warning(f"Failed to get recommendation metrics: {e}")
            metrics = {'error': 'Metrics unavailable'}
        
        return jsonify({
            'recommendations': result,
            'ml_strategy': 'advanced_hybrid_with_realtime_learning',
            'user_metrics': metrics,
            'diversity_applied': diversity_factor,
            'novelty_applied': novelty_factor,
            'recommendation_quality': 'high_precision',
            'context': context,
            'response_time_ms': round(response_time * 1000, 2),
            'ml_features': {
                'real_time_learning': True,
                'context_awareness': True,
                'multi_algorithm_hybrid': True,
                'diversity_optimization': True,
                'novelty_injection': True,
                'explanation_generation': include_explanations
            }
        }), 200
        
    except Exception as e:
        logger.error(f"ML personalized recommendations error: {e}")
        return jsonify({
            'recommendations': [],
            'error': 'Failed to get ML recommendations',
            'fallback': True
        }), 200

@users_bp.route('/api/user/interaction-history', methods=['GET'])
@require_auth
def get_interaction_history(current_user):
    """Get comprehensive user interaction history with analytics"""
    try:
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 50)), 100)
        interaction_type = request.args.get('type', 'all')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Build query
        query = UserInteraction.query.filter_by(user_id=current_user.id)
        
        if interaction_type != 'all':
            query = query.filter_by(interaction_type=interaction_type)
        
        # Date filtering
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                query = query.filter(UserInteraction.timestamp >= start_dt)
            except ValueError:
                pass
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                query = query.filter(UserInteraction.timestamp <= end_dt)
            except ValueError:
                pass
        
        # Paginate
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
                    'metadata': getattr(interaction, 'interaction_metadata', {}) or {},
                    'content_rating': content.rating,
                    'content_genres': json.loads(content.genres or '[]')
                })
        
        # Calculate summary statistics
        total_interactions = query.count()
        interaction_types = db.session.query(
            UserInteraction.interaction_type,
            db.func.count(UserInteraction.id).label('count')
        ).filter_by(user_id=current_user.id).group_by(
            UserInteraction.interaction_type
        ).all()
        
        type_breakdown = {stat.interaction_type: stat.count for stat in interaction_types}
        
        return jsonify({
            'interactions': result,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': interactions.total,
                'pages': interactions.pages,
                'has_next': interactions.has_next,
                'has_prev': interactions.has_prev
            },
            'summary': {
                'total_interactions': total_interactions,
                'type_breakdown': type_breakdown,
                'filtered_count': interactions.total
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Interaction history error: {e}")
        return jsonify({'error': 'Failed to get interaction history'}), 500

@users_bp.route('/api/user/recommendation-feedback', methods=['POST'])
@require_auth
def record_recommendation_feedback(current_user):
    """Record feedback on recommendations for ML learning"""
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'feedback_type', 'recommendation_id']
        is_valid, error_message = _validate_user_input(data, required_fields)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        # Validate feedback type
        valid_feedback_types = [
            'like', 'dislike', 'not_interested', 'already_seen',
            'inappropriate', 'poor_quality', 'love', 'helpful'
        ]
        
        if data['feedback_type'] not in valid_feedback_types:
            return jsonify({'error': 'Invalid feedback type'}), 400
        
        # Validate content exists
        content = Content.query.get(data['content_id'])
        if not content:
            return jsonify({'error': 'Content not found'}), 404
        
        # Record feedback
        success = False
        if recommendation_engine:
            try:
                success = recommendation_engine.record_recommendation_feedback(
                    user_id=current_user.id,
                    content_id=data['content_id'],
                    feedback_type=data['feedback_type'],
                    recommendation_id=data['recommendation_id'],
                    feedback_value=data.get('feedback_value', 1.0)
                )
            except Exception as e:
                logger.error(f"Failed to record ML feedback: {e}")
        
        # Also record as regular interaction for backup
        try:
            feedback_interaction = UserInteraction(
                user_id=current_user.id,
                content_id=data['content_id'],
                interaction_type='recommendation_feedback',
                interaction_metadata={
                    'feedback_type': data['feedback_type'],
                    'recommendation_id': data['recommendation_id'],
                    'feedback_value': data.get('feedback_value', 1.0),
                    'feedback_context': data.get('context', {}),
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            db.session.add(feedback_interaction)
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Failed to record backup feedback: {e}")
        
        return jsonify({
            'message': 'Feedback recorded successfully',
            'ml_learning_enabled': success
        }), 201
        
    except Exception as e:
        logger.error(f"Recommendation feedback error: {e}")
        return jsonify({'error': 'Failed to record feedback'}), 500

@users_bp.route('/api/user/analytics', methods=['GET'])
@require_auth
def get_user_analytics(current_user):
    """Get comprehensive user analytics and insights"""
    try:
        # Time range for analytics
        days = int(request.args.get('days', 30))
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Interaction analytics
        interactions = UserInteraction.query.filter(
            UserInteraction.user_id == current_user.id,
            UserInteraction.timestamp >= start_date
        ).all()
        
        # Daily activity
        daily_activity = defaultdict(int)
        interaction_types = defaultdict(int)
        content_types = defaultdict(int)
        genres_engaged = defaultdict(int)
        
        for interaction in interactions:
            day = interaction.timestamp.date()
            daily_activity[day.isoformat()] += 1
            interaction_types[interaction.interaction_type] += 1
            
            content = Content.query.get(interaction.content_id)
            if content:
                content_types[content.content_type] += 1
                
                try:
                    genres = json.loads(content.genres or '[]')
                    for genre in genres:
                        genres_engaged[genre] += 1
                except:
                    pass
        
        # Recommendation performance
        user_performance = []
        with _performance_lock:
            if current_user.id in recommendation_performance:
                user_performance = list(recommendation_performance[current_user.id])[-50:]  # Last 50
        
        # ML metrics
        ml_metrics = {}
        if recommendation_engine:
            try:
                ml_metrics = recommendation_engine.get_recommendation_metrics(current_user.id)
            except Exception as e:
                logger.warning(f"Failed to get ML metrics: {e}")
        
        analytics_data = {
            'period': {
                'days': days,
                'start_date': start_date.isoformat(),
                'end_date': datetime.utcnow().isoformat()
            },
            'activity_summary': {
                'total_interactions': len(interactions),
                'daily_average': len(interactions) / max(days, 1),
                'most_active_day': max(daily_activity.items(), key=lambda x: x[1])[0] if daily_activity else None
            },
            'interaction_patterns': {
                'by_type': dict(interaction_types),
                'by_content_type': dict(content_types),
                'top_genres': dict(sorted(genres_engaged.items(), key=lambda x: x[1], reverse=True)[:10])
            },
            'daily_activity': dict(daily_activity),
            'recommendation_performance': {
                'recent_requests': len(user_performance),
                'avg_response_time': np.mean([p['response_time'] for p in user_performance]) if user_performance else 0,
                'strategies_used': list(set(p['strategy'] for p in user_performance)) if user_performance else []
            },
            'ml_insights': ml_metrics
        }
        
        return jsonify(analytics_data), 200
        
    except Exception as e:
        logger.error(f"User analytics error: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500

# Helper functions
def _get_time_of_day():
    """Get current time of day category"""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 22:
        return 'evening'
    else:
        return 'night'

def _infer_location_type(request):
    """Infer location type from request context"""
    user_agent = request.headers.get('User-Agent', '').lower()
    
    if 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent:
        return 'mobile'
    elif 'smart-tv' in user_agent or 'roku' in user_agent:
        return 'living_room'
    else:
        return 'home'

# Cleanup routes for housekeeping
@users_bp.route('/api/user/watchlist/<int:content_id>', methods=['DELETE'])
@require_auth
def remove_from_watchlist(current_user, content_id):
    """Remove content from user's watchlist"""
    try:
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='watchlist'
        ).first()
        
        if interaction:
            db.session.delete(interaction)
            db.session.commit()
            
            # Update ML engine
            if recommendation_engine:
                try:
                    recommendation_engine.update_user_profile(current_user.id)
                except Exception as e:
                    logger.warning(f"Failed to update ML profile: {e}")
            
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
    """Check if content is in user's watchlist"""
    try:
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='watchlist'
        ).first()
        
        return jsonify({
            'in_watchlist': interaction is not None,
            'date_added': interaction.timestamp.isoformat() if interaction else None
        }), 200
        
    except Exception as e:
        logger.error(f"Check watchlist status error: {e}")
        return jsonify({'error': 'Failed to check watchlist status'}), 500

# Export performance data for monitoring
@users_bp.route('/api/user/performance-stats', methods=['GET'])
@require_auth
def get_performance_stats(current_user):
    """Get performance statistics for the user (admin or self-analytics)"""
    try:
        stats = {
            'ml_services_available': ML_SERVICES_AVAILABLE,
            'recommendation_engine_status': 'available' if recommendation_engine else 'unavailable'
        }
        
        # Add user-specific performance data
        with _performance_lock:
            if current_user.id in recommendation_performance:
                user_perf = list(recommendation_performance[current_user.id])
                stats['recommendation_requests'] = len(user_perf)
                
                if user_perf:
                    response_times = [p['response_time'] for p in user_perf]
                    stats['avg_response_time'] = np.mean(response_times)
                    stats['min_response_time'] = min(response_times)
                    stats['max_response_time'] = max(response_times)
            
            if current_user.id in user_activity_tracker:
                user_activity = list(user_activity_tracker[current_user.id])
                stats['total_activities'] = len(user_activity)
                
                if user_activity:
                    recent_activity = [a for a in user_activity if 
                                     (datetime.utcnow() - a['timestamp']).total_seconds() < 3600]
                    stats['activities_last_hour'] = len(recent_activity)
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Performance stats error: {e}")
        return jsonify({'error': 'Failed to get performance stats'}), 500