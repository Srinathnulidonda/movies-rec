#backend/services/users.py
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import logging
import jwt
import sys
import os
from functools import wraps
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from ml_services.recommendation import AdvancedRecommendationEngine
except ImportError:
    AdvancedRecommendationEngine = None
    print("Warning: Advanced ML Services not available. Recommendation features will be limited.")

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
    
    if AdvancedRecommendationEngine:
        try:
            recommendation_engine = AdvancedRecommendationEngine(db, models)
            print("Advanced ML Recommendation Engine initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize Advanced ML Recommendation Engine: {e}")
            recommendation_engine = None
    else:
        print("Warning: AdvancedRecommendationEngine not available")
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
        
        # Initialize user profile in recommendation engine
        if recommendation_engine:
            try:
                recommendation_engine.initialize_user_profile(user.id, {
                    'preferred_languages': data.get('preferred_languages', ['english', 'telugu']),
                    'preferred_genres': data.get('preferred_genres', [])
                })
            except Exception as e:
                logger.warning(f"Failed to initialize user profile: {e}")
        
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

@users_bp.route('/api/interactions/advanced', methods=['POST'])
@require_auth
def record_advanced_interaction(current_user):
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'interaction_type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Enhanced interaction metadata
        enhanced_metadata = {
            'timestamp': datetime.utcnow().isoformat(),
            'session_duration': data.get('session_duration', 0),
            'view_percentage': data.get('view_percentage', 0),
            'interaction_quality': data.get('interaction_quality', 'medium'),
            'device_type': data.get('device_type', 'unknown'),
            'source': data.get('source', 'direct'),
            'context': data.get('context', {}),
            'engagement_score': data.get('engagement_score', 1.0)
        }
        
        # Handle special interaction types
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
                        recommendation_engine.update_user_behavior_profile(
                            current_user.id, 'remove_watchlist', data['content_id'], enhanced_metadata
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update behavior profile: {e}")
                
                return jsonify({'message': 'Removed from watchlist'}), 200
            else:
                return jsonify({'message': 'Content not in watchlist'}), 404
        
        # Check for existing interactions
        if data['interaction_type'] == 'watchlist':
            existing = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=data['content_id'],
                interaction_type='watchlist'
            ).first()
            
            if existing:
                return jsonify({'message': 'Already in watchlist'}), 200
        
        # Create enhanced interaction
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=data['content_id'],
            interaction_type=data['interaction_type'],
            rating=data.get('rating'),
            interaction_metadata=enhanced_metadata
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        # Update recommendation engine with enhanced data
        if recommendation_engine:
            try:
                recommendation_engine.process_real_time_interaction(
                    user_id=current_user.id,
                    content_id=data['content_id'],
                    interaction_type=data['interaction_type'],
                    rating=data.get('rating'),
                    metadata=enhanced_metadata
                )
            except Exception as e:
                logger.warning(f"Failed to process real-time interaction: {e}")
        
        return jsonify({'message': 'Interaction recorded successfully'}), 201
        
    except Exception as e:
        logger.error(f"Advanced interaction recording error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record interaction'}), 500

@users_bp.route('/api/interactions/search', methods=['POST'])
@require_auth
def record_search_interaction(current_user):
    try:
        data = request.get_json()
        
        search_metadata = {
            'query': data.get('query', ''),
            'results_count': data.get('results_count', 0),
            'clicked_position': data.get('clicked_position', -1),
            'search_context': data.get('context', ''),
            'filters_applied': data.get('filters', {}),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Record search interaction
        if data.get('content_id'):
            interaction = UserInteraction(
                user_id=current_user.id,
                content_id=data['content_id'],
                interaction_type='search_click',
                interaction_metadata=search_metadata
            )
            db.session.add(interaction)
        
        # Record general search behavior
        search_behavior = UserInteraction(
            user_id=current_user.id,
            content_id=None,
            interaction_type='search_query',
            interaction_metadata=search_metadata
        )
        db.session.add(search_behavior)
        db.session.commit()
        
        # Update search behavior in recommendation engine
        if recommendation_engine:
            try:
                recommendation_engine.update_search_behavior(
                    current_user.id, data.get('query', ''), search_metadata
                )
            except Exception as e:
                logger.warning(f"Failed to update search behavior: {e}")
        
        return jsonify({'message': 'Search interaction recorded'}), 201
        
    except Exception as e:
        logger.error(f"Search interaction error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record search interaction'}), 500

@users_bp.route('/api/recommendations/ultra-personalized', methods=['GET'])
@require_auth
def get_ultra_personalized_recommendations(current_user):
    try:
        if not recommendation_engine:
            return jsonify({
                'recommendations': [],
                'error': 'Advanced recommendation engine not available',
                'fallback': True
            }), 200
        
        # Get parameters
        limit = int(request.args.get('limit', 20))
        content_type = request.args.get('content_type', 'all')
        diversity_factor = float(request.args.get('diversity_factor', 0.4))
        novelty_factor = float(request.args.get('novelty_factor', 0.3))
        include_explanations = request.args.get('include_explanations', 'true').lower() == 'true'
        time_context = request.args.get('time_context', 'any')  # morning, afternoon, evening, weekend
        mood_context = request.args.get('mood', 'neutral')  # happy, sad, excited, relaxed
        
        # Get ultra-personalized recommendations
        recommendations = recommendation_engine.get_ultra_personalized_recommendations(
            user_id=current_user.id,
            limit=limit,
            content_type=content_type,
            diversity_factor=diversity_factor,
            novelty_factor=novelty_factor,
            include_explanations=include_explanations,
            time_context=time_context,
            mood_context=mood_context
        )
        
        # Format response
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
                
                # Enhanced recommendation data
                'personalization_score': rec['personalization_score'],
                'confidence_score': rec['confidence_score'],
                'novelty_score': rec['novelty_score'],
                'diversity_contribution': rec['diversity_contribution'],
                'explanation': rec['explanation'],
                'matching_factors': rec['matching_factors'],
                'predicted_rating': rec['predicted_rating'],
                'recommendation_strength': rec['recommendation_strength'],
                'algorithm_breakdown': rec['algorithm_breakdown'],
                'behavioral_match': rec['behavioral_match'],
                'temporal_relevance': rec['temporal_relevance']
            })
        
        # Get user behavior insights
        behavior_insights = recommendation_engine.get_user_behavior_insights(current_user.id)
        
        return jsonify({
            'recommendations': result,
            'recommendation_metadata': {
                'strategy': 'ultra_personalized_ml',
                'user_profile_strength': recommendation_engine.get_user_profile_strength(current_user.id),
                'total_interactions': recommendation_engine.get_user_interaction_count(current_user.id),
                'personalization_accuracy': behavior_insights.get('accuracy_score', 0.0),
                'diversity_applied': diversity_factor,
                'novelty_applied': novelty_factor,
                'context_factors': {
                    'time_context': time_context,
                    'mood_context': mood_context
                }
            },
            'user_insights': behavior_insights,
            'recommendation_quality': 'ultra_high_precision'
        }), 200
        
    except Exception as e:
        logger.error(f"Ultra-personalized recommendations error: {e}")
        return jsonify({'recommendations': [], 'error': 'Failed to get recommendations'}), 200

@users_bp.route('/api/recommendations/contextual', methods=['GET'])
@require_auth
def get_contextual_recommendations(current_user):
    try:
        if not recommendation_engine:
            return jsonify({'recommendations': [], 'error': 'Engine not available'}), 200
        
        # Context parameters
        viewing_time = request.args.get('viewing_time', 'evening')
        available_time = int(request.args.get('available_time', 120))  # minutes
        viewing_companions = request.args.get('companions', 'alone')  # alone, family, friends
        viewing_device = request.args.get('device', 'tv')  # tv, mobile, laptop
        mood = request.args.get('mood', 'neutral')
        occasion = request.args.get('occasion', 'regular')  # weekend, holiday, date_night
        
        recommendations = recommendation_engine.get_contextual_recommendations(
            user_id=current_user.id,
            context={
                'viewing_time': viewing_time,
                'available_time': available_time,
                'viewing_companions': viewing_companions,
                'viewing_device': viewing_device,
                'mood': mood,
                'occasion': occasion
            },
            limit=int(request.args.get('limit', 15))
        )
        
        result = []
        for rec in recommendations:
            content = rec['content']
            result.append({
                'id': content.id,
                'slug': getattr(content, 'slug', None),
                'title': content.title,
                'content_type': content.content_type,
                'runtime': content.runtime,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'context_match_score': rec['context_match_score'],
                'context_explanation': rec['context_explanation'],
                'suitability_factors': rec['suitability_factors']
            })
        
        return jsonify({
            'recommendations': result,
            'context_applied': {
                'viewing_time': viewing_time,
                'available_time': available_time,
                'companions': viewing_companions,
                'device': viewing_device,
                'mood': mood,
                'occasion': occasion
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Contextual recommendations error: {e}")
        return jsonify({'recommendations': [], 'error': str(e)}), 200

@users_bp.route('/api/user/behavior-analysis', methods=['GET'])
@require_auth
def get_user_behavior_analysis(current_user):
    try:
        if not recommendation_engine:
            return jsonify({'error': 'Engine not available'}), 503
        
        analysis = recommendation_engine.get_comprehensive_user_analysis(current_user.id)
        
        return jsonify({
            'user_id': current_user.id,
            'analysis': analysis,
            'last_updated': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Behavior analysis error: {e}")
        return jsonify({'error': 'Failed to analyze behavior'}), 500

@users_bp.route('/api/user/recommendation-feedback/advanced', methods=['POST'])
@require_auth
def record_advanced_feedback(current_user):
    try:
        data = request.get_json()
        
        feedback_data = {
            'content_id': data['content_id'],
            'feedback_type': data['feedback_type'],  # loved, liked, neutral, disliked, hated
            'feedback_reasons': data.get('reasons', []),  # array of reasons
            'recommendation_quality': data.get('quality', 'good'),  # excellent, good, fair, poor
            'explanation_helpfulness': data.get('explanation_helpful', True),
            'surprise_factor': data.get('surprise_factor', 'expected'),  # surprising, expected, boring
            'discovery_value': data.get('discovery_value', 'medium'),  # high, medium, low
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if recommendation_engine:
            recommendation_engine.process_advanced_feedback(current_user.id, feedback_data)
        
        return jsonify({'message': 'Advanced feedback recorded successfully'}), 201
        
    except Exception as e:
        logger.error(f"Advanced feedback error: {e}")
        return jsonify({'error': 'Failed to record feedback'}), 500

@users_bp.route('/api/user/watchlist', methods=['GET'])
@require_auth
def get_watchlist(current_user):
    try:
        watchlist_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='watchlist'
        ).order_by(UserInteraction.timestamp.desc()).all()
        
        content_ids = [interaction.content_id for interaction in watchlist_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        
        result = []
        for content in contents:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            # Get predicted rating for this user
            predicted_rating = None
            if recommendation_engine:
                try:
                    predicted_rating = recommendation_engine.predict_user_rating(current_user.id, content.id)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'slug': getattr(content, 'slug', None),
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'predicted_rating': predicted_rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'youtube_trailer': youtube_url,
                'added_date': next((i.timestamp.isoformat() for i in watchlist_interactions if i.content_id == content.id), None)
            })
        
        return jsonify({
            'watchlist': result,
            'total_count': len(result),
            'predicted_watch_time': sum([c.get('runtime', 0) or 0 for c in result if isinstance(c, dict)])
        }), 200
        
    except Exception as e:
        logger.error(f"Watchlist error: {e}")
        return jsonify({'error': 'Failed to get watchlist'}), 500

@users_bp.route('/api/user/smart-recommendations', methods=['GET'])
@require_auth
def get_smart_recommendations(current_user):
    """Get recommendations based on current user activity patterns and preferences"""
    try:
        if not recommendation_engine:
            return jsonify({'recommendations': [], 'error': 'Engine not available'}), 200
        
        # Analyze current user state
        current_hour = datetime.now().hour
        day_of_week = datetime.now().weekday()
        
        # Get smart recommendations based on user patterns
        recommendations = recommendation_engine.get_smart_recommendations(
            user_id=current_user.id,
            current_context={
                'hour': current_hour,
                'day_of_week': day_of_week,
                'is_weekend': day_of_week >= 5
            },
            limit=int(request.args.get('limit', 12))
        )
        
        result = []
        for rec in recommendations:
            content = rec['content']
            result.append({
                'id': content.id,
                'slug': getattr(content, 'slug', None),
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'smart_score': rec['smart_score'],
                'timing_relevance': rec['timing_relevance'],
                'pattern_match': rec['pattern_match'],
                'recommendation_reason': rec['reason']
            })
        
        return jsonify({
            'recommendations': result,
            'context': {
                'current_hour': current_hour,
                'day_type': 'weekend' if day_of_week >= 5 else 'weekday',
                'recommendation_strategy': 'smart_pattern_based'
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Smart recommendations error: {e}")
        return jsonify({'recommendations': [], 'error': str(e)}), 200