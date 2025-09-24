# backend/services/users.py
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import logging
import jwt
from functools import wraps
from collections import defaultdict, Counter
from sqlalchemy import func, and_, or_, desc
from typing import Optional, Dict, List, Any

# Import personalized recommendation functions
from services.personalized import (
    get_personalized_recommendations_for_user,
    update_user_profile,
    record_recommendation_feedback
)

# Create users blueprint
users_bp = Blueprint('users', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Will be initialized by main app
db = None
User = None
Content = None
UserInteraction = None
AnonymousInteraction = None
Review = None
UserPreference = None
RecommendationFeedback = None
UserSession = None
http_session = None
cache = None
app = None

def init_users(flask_app, database, models, services):
    """Initialize users module with app context and models"""
    global db, User, Content, UserInteraction, AnonymousInteraction, Review
    global UserPreference, RecommendationFeedback, UserSession
    global http_session, cache, app
    
    app = flask_app
    db = database
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    AnonymousInteraction = models.get('AnonymousInteraction')
    Review = models.get('Review')
    UserPreference = models.get('UserPreference')
    RecommendationFeedback = models.get('RecommendationFeedback')
    UserSession = models.get('UserSession')
    
    http_session = services['http_session']
    cache = services.get('cache')

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
            
            # Update last active
            current_user.last_active = datetime.utcnow()
            
            # Track session
            _track_user_session(current_user.id, request)
            
            db.session.commit()
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return jsonify({'error': 'Authentication failed'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

def _track_user_session(user_id: int, request_obj) -> None:
    """Track user session activity"""
    try:
        session_id = request_obj.headers.get('X-Session-Id')
        if not session_id:
            return
        
        # Find or create session
        session = UserSession.query.filter_by(
            user_id=user_id,
            session_id=session_id
        ).first()
        
        if not session:
            session = UserSession(
                user_id=user_id,
                session_id=session_id,
                device_type=_detect_device_type(request_obj.headers.get('User-Agent', '')),
                ip_address=request_obj.remote_addr,
                user_agent=request_obj.headers.get('User-Agent', '')
            )
            db.session.add(session)
        
        # Update session activity
        session.interactions_count += 1
        session.end_time = datetime.utcnow()
        
        if session.start_time:
            session.duration = (session.end_time - session.start_time).seconds
        
    except Exception as e:
        logger.error(f"Error tracking session: {e}")

def _detect_device_type(user_agent: str) -> str:
    """Detect device type from user agent"""
    user_agent_lower = user_agent.lower()
    
    if 'mobile' in user_agent_lower or 'android' in user_agent_lower:
        return 'mobile'
    elif 'tablet' in user_agent_lower or 'ipad' in user_agent_lower:
        return 'tablet'
    elif 'tv' in user_agent_lower or 'smart' in user_agent_lower:
        return 'smart_tv'
    else:
        return 'desktop'

class UserAnalytics:
    """Enhanced user analytics and insights"""
    
    @staticmethod
    def get_user_stats(user_id: int) -> dict:
        """Get comprehensive user statistics"""
        try:
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            
            if not interactions:
                return UserAnalytics._get_empty_stats()
            
            # Basic stats
            stats = {
                'total_interactions': len(interactions),
                'content_watched': len([i for i in interactions if i.interaction_type == 'view']),
                'favorites': len([i for i in interactions if i.interaction_type == 'favorite']),
                'watchlist_items': len([i for i in interactions if i.interaction_type == 'watchlist']),
                'ratings_given': len([i for i in interactions if i.interaction_type == 'rating']),
                'likes_given': len([i for i in interactions if i.interaction_type == 'like'])
            }
            
            # Average rating
            ratings = [i.rating for i in interactions if i.rating is not None]
            stats['average_rating'] = round(sum(ratings) / len(ratings), 1) if ratings else 0
            
            # Get content for analysis
            content_ids = list(set([i.content_id for i in interactions]))
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            content_map = {c.id: c for c in contents}
            
            # Genre and content type analysis
            genre_counts = defaultdict(int)
            content_type_counts = defaultdict(int)
            
            for interaction in interactions:
                content = content_map.get(interaction.content_id)
                if content:
                    content_type_counts[content.content_type] += 1
                    
                    try:
                        genres = json.loads(content.genres or '[]')
                        for genre in genres:
                            genre_counts[genre.lower()] += 1
                    except (json.JSONDecodeError, TypeError):
                        pass
            
            stats['most_watched_genre'] = max(genre_counts, key=genre_counts.get) if genre_counts else None
            stats['preferred_content_type'] = max(content_type_counts, key=content_type_counts.get) if content_type_counts else None
            
            # Viewing streak
            stats['viewing_streak'] = UserAnalytics._calculate_viewing_streak(interactions)
            
            # Discovery score
            stats['discovery_score'] = UserAnalytics._calculate_discovery_score(interactions, contents)
            
            # Monthly activity
            stats['monthly_activity'] = UserAnalytics._get_monthly_activity(interactions)
            
            # Content quality preference
            quality_ratings = [content_map[i.content_id].rating for i in interactions 
                             if i.content_id in content_map and content_map[i.content_id].rating]
            stats['preferred_content_quality'] = round(sum(quality_ratings) / len(quality_ratings), 1) if quality_ratings else 0
            
            # Personalization readiness
            stats['personalization_ready'] = stats['total_interactions'] >= 10
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return UserAnalytics._get_empty_stats()
    
    @staticmethod
    def _get_empty_stats() -> dict:
        """Return empty stats structure"""
        return {
            'total_interactions': 0,
            'content_watched': 0,
            'favorites': 0,
            'watchlist_items': 0,
            'ratings_given': 0,
            'average_rating': 0,
            'most_watched_genre': None,
            'preferred_content_type': None,
            'viewing_streak': 0,
            'discovery_score': 0.0,
            'monthly_activity': [],
            'preferred_content_quality': 0,
            'personalization_ready': False
        }
    
    @staticmethod
    def _calculate_viewing_streak(interactions):
        """Calculate consecutive days with viewing activity"""
        if not interactions:
            return 0
        
        dates = set()
        for interaction in interactions:
            dates.add(interaction.timestamp.date())
        
        if not dates:
            return 0
        
        sorted_dates = sorted(dates, reverse=True)
        streak = 1
        
        for i in range(1, len(sorted_dates)):
            if (sorted_dates[i-1] - sorted_dates[i]).days == 1:
                streak += 1
            else:
                break
        
        return streak
    
    @staticmethod
    def _calculate_discovery_score(interactions, contents):
        """Calculate how much new/diverse content user explores"""
        if not interactions or not contents:
            return 0.0
        
        all_genres = set()
        for content in contents:
            try:
                genres = json.loads(content.genres or '[]')
                all_genres.update([g.lower() for g in genres])
            except (json.JSONDecodeError, TypeError):
                pass
        
        genre_diversity = len(all_genres) / 20.0 if all_genres else 0
        
        popularities = [c.popularity for c in contents if c.popularity]
        avg_popularity = sum(popularities) / len(popularities) if popularities else 100
        popularity_exploration = max(0, (200 - avg_popularity) / 200) if avg_popularity else 0.5
        
        return min((genre_diversity + popularity_exploration) / 2, 1.0)
    
    @staticmethod
    def _get_monthly_activity(interactions):
        """Get monthly activity breakdown"""
        monthly_counts = defaultdict(int)
        
        for interaction in interactions:
            month_key = interaction.timestamp.strftime('%Y-%m')
            monthly_counts[month_key] += 1
        
        current_date = datetime.utcnow()
        last_6_months = []
        
        for i in range(6):
            date = current_date - timedelta(days=30*i)
            month_key = date.strftime('%Y-%m')
            last_6_months.append({
                'month': month_key,
                'count': monthly_counts.get(month_key, 0)
            })
        
        return list(reversed(last_6_months))

# Enhanced User Profile Management
@users_bp.route('/api/user/profile', methods=['GET'])
@require_auth
def get_user_profile(current_user):
    """Get comprehensive user profile with personalization insights"""
    try:
        # Get basic stats
        stats = UserAnalytics.get_user_stats(current_user.id)
        
        # Get recent activity
        recent_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id
        ).order_by(UserInteraction.timestamp.desc()).limit(10).all()
        
        recent_activity = []
        for interaction in recent_interactions:
            content = Content.query.get(interaction.content_id)
            if content:
                recent_activity.append({
                    'content_id': content.id,
                    'content_title': content.title,
                    'content_type': content.content_type,
                    'interaction_type': interaction.interaction_type,
                    'timestamp': interaction.timestamp.isoformat(),
                    'rating': interaction.rating
                })
        
        # Get personalization status
        user_pref = UserPreference.query.filter_by(user_id=current_user.id).first()
        personalization_status = {
            'enabled': user_pref is not None,
            'profile_strength': user_pref.profile_strength if user_pref else 0,
            'confidence_score': user_pref.confidence_score if user_pref else 0,
            'current_mode': user_pref.current_mode if user_pref else 'discovery_mode',
            'last_updated': user_pref.updated_at.isoformat() if user_pref and user_pref.updated_at else None
        }
        
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
            'recent_activity': recent_activity,
            'personalization_status': personalization_status
        }
        
        return jsonify(profile_data), 200
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return jsonify({'error': 'Failed to get user profile'}), 500

@users_bp.route('/api/user/profile', methods=['PUT'])
@require_auth
def update_user_profile_endpoint(current_user):
    """Update user profile and preferences"""
    try:
        data = request.get_json()
        
        # Update basic info
        if 'preferred_languages' in data:
            current_user.preferred_languages = json.dumps(data['preferred_languages'])
        
        if 'preferred_genres' in data:
            current_user.preferred_genres = json.dumps(data['preferred_genres'])
        
        if 'location' in data:
            current_user.location = data['location']
        
        if 'avatar_url' in data:
            current_user.avatar_url = data['avatar_url']
        
        db.session.commit()
        
        # Update personalization profile
        update_user_profile(current_user.id, force_update=True)
        
        # Clear user caches
        if cache:
            cache.delete(f"user_profile:{current_user.id}")
            cache.delete(f"advanced_profile:{current_user.id}")
        
        return jsonify({'message': 'Profile updated successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update profile'}), 500

# Personalized Recommendations (Integrated)
@users_bp.route('/api/user/recommendations', methods=['GET'])
@require_auth
def get_user_recommendations(current_user):
    """Get hyper-personalized recommendations for the user"""
    try:
        content_type = request.args.get('type')  # movie, tv, anime
        limit = min(int(request.args.get('limit', 20)), 50)
        
        # Get personalized recommendations
        recommendations = get_personalized_recommendations_for_user(
            current_user.id, content_type, limit
        )
        
        return jsonify(recommendations), 200
        
    except Exception as e:
        logger.error(f"Error getting personalized recommendations: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@users_bp.route('/api/user/recommendations/categories', methods=['GET'])
@require_auth
def get_user_recommendation_categories(current_user):
    """Get personalized recommendations grouped by categories"""
    try:
        categories = {
            'for_you': get_personalized_recommendations_for_user(current_user.id, None, 15),
            'movies': get_personalized_recommendations_for_user(current_user.id, 'movie', 12),
            'tv_shows': get_personalized_recommendations_for_user(current_user.id, 'tv', 12),
            'anime': get_personalized_recommendations_for_user(current_user.id, 'anime', 12)
        }
        
        return jsonify({
            'categories': categories,
            'metadata': {
                'personalized': True,
                'user_id': current_user.id,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting recommendation categories: {e}")
        return jsonify({'error': 'Failed to get recommendation categories'}), 500

# Enhanced Interaction Routes
@users_bp.route('/api/interactions', methods=['POST'])
@require_auth
def record_interaction(current_user):
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'interaction_type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        content_id = data['content_id']
        interaction_type = data['interaction_type']
        
        # Handle remove_watchlist specially
        if interaction_type == 'remove_watchlist':
            interaction = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=content_id,
                interaction_type='watchlist'
            ).first()
            
            if interaction:
                db.session.delete(interaction)
                db.session.commit()
                
                # Update profile
                update_user_profile(current_user.id, force_update=False)
                
                return jsonify({'message': 'Removed from watchlist'}), 200
            else:
                return jsonify({'message': 'Content not in watchlist'}), 404
        
        # For adding to watchlist, check if already exists
        if interaction_type == 'watchlist':
            existing = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=content_id,
                interaction_type='watchlist'
            ).first()
            
            if existing:
                return jsonify({'message': 'Already in watchlist'}), 200
        
        # Enhanced interaction recording with metadata
        interaction_metadata = {
            'from_recommendation': data.get('from_recommendation', False),
            'recommendation_score': data.get('recommendation_score'),
            'recommendation_method': data.get('recommendation_method'),
            'user_agent': request.headers.get('User-Agent', ''),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type=interaction_type,
            rating=data.get('rating'),
            interaction_metadata=json.dumps(interaction_metadata)
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        # Record recommendation feedback if from recommendation
        if data.get('from_recommendation'):
            feedback_type = 'clicked' if interaction_type == 'view' else interaction_type
            record_recommendation_feedback(
                current_user.id, 
                content_id, 
                feedback_type,
                data.get('rating')
            )
        
        # Update user profile (async)
        update_user_profile(current_user.id, force_update=False)
        
        # Clear user caches
        if cache:
            cache.delete(f"user_profile:{current_user.id}")
        
        return jsonify({'message': 'Interaction recorded successfully'}), 201
        
    except Exception as e:
        logger.error(f"Interaction recording error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record interaction'}), 500

# Enhanced Watchlist Management
@users_bp.route('/api/user/watchlist', methods=['GET'])
@require_auth
def get_watchlist(current_user):
    try:
        sort_by = request.args.get('sort_by', 'added_date')
        order = request.args.get('order', 'desc')
        
        watchlist_query = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='watchlist'
        )
        
        if sort_by == 'added_date':
            if order == 'desc':
                watchlist_query = watchlist_query.order_by(UserInteraction.timestamp.desc())
            else:
                watchlist_query = watchlist_query.order_by(UserInteraction.timestamp.asc())
        
        watchlist_interactions = watchlist_query.all()
        content_ids = [interaction.content_id for interaction in watchlist_interactions]
        
        if not content_ids:
            return jsonify({'watchlist': [], 'count': 0}), 200
        
        contents_query = Content.query.filter(Content.id.in_(content_ids))
        
        # Apply additional sorting
        if sort_by == 'title':
            if order == 'desc':
                contents_query = contents_query.order_by(Content.title.desc())
            else:
                contents_query = contents_query.order_by(Content.title.asc())
        elif sort_by == 'rating':
            if order == 'desc':
                contents_query = contents_query.order_by(Content.rating.desc())
            else:
                contents_query = contents_query.order_by(Content.rating.asc())
        elif sort_by == 'release_date':
            if order == 'desc':
                contents_query = contents_query.order_by(Content.release_date.desc())
            else:
                contents_query = contents_query.order_by(Content.release_date.asc())
        
        contents = contents_query.all()
        
        # Create interaction mapping
        interaction_map = {i.content_id: i for i in watchlist_interactions}
        
        result = []
        for content in contents:
            interaction = interaction_map.get(content.id)
            
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            content_data = {
                'id': content.id,
                'slug': content.slug,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'languages': json.loads(content.languages or '[]'),
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'runtime': content.runtime,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:200] + '...' if content.overview and len(content.overview) > 200 else content.overview,
                'youtube_trailer': youtube_url,
                'added_to_watchlist': interaction.timestamp.isoformat() if interaction else None
            }
            
            result.append(content_data)
        
        # Get personalized recommendations based on watchlist
        watchlist_based_recs = []
        if len(result) >= 3:
            try:
                # Get recommendations similar to watchlist items
                recent_watchlist_ids = [c['id'] for c in result[:5]]
                similar_recs = get_personalized_recommendations_for_user(
                    current_user.id, None, 5
                )
                watchlist_based_recs = similar_recs.get('recommendations', [])[:5]
            except Exception as e:
                logger.warning(f"Could not get watchlist-based recommendations: {e}")
        
        return jsonify({
            'watchlist': result,
            'count': len(result),
            'sort_by': sort_by,
            'order': order,
            'personalized_suggestions': watchlist_based_recs
        }), 200
        
    except Exception as e:
        logger.error(f"Watchlist error: {e}")
        return jsonify({'error': 'Failed to get watchlist'}), 500

# Enhanced Favorites Management
@users_bp.route('/api/user/favorites', methods=['GET'])
@require_auth
def get_favorites(current_user):
    try:
        content_type_filter = request.args.get('type')
        
        favorites_query = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='favorite'
        ).order_by(UserInteraction.timestamp.desc())
        
        favorite_interactions = favorites_query.all()
        content_ids = [interaction.content_id for interaction in favorite_interactions]
        
        if not content_ids:
            return jsonify({'favorites': [], 'count': 0}), 200
        
        contents_query = Content.query.filter(Content.id.in_(content_ids))
        
        if content_type_filter:
            contents_query = contents_query.filter(Content.content_type == content_type_filter)
        
        contents = contents_query.all()
        
        # Group by content type
        grouped_favorites = {
            'movies': [],
            'tv_shows': [],
            'anime': []
        }
        
        all_favorites = []
        
        for content in contents:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            content_data = {
                'id': content.id,
                'slug': content.slug,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'languages': json.loads(content.languages or '[]'),
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:200] + '...' if content.overview and len(content.overview) > 200 else content.overview,
                'youtube_trailer': youtube_url
            }
            
            all_favorites.append(content_data)
            
            # Group by type
            if content.content_type == 'movie':
                grouped_favorites['movies'].append(content_data)
            elif content.content_type == 'tv':
                grouped_favorites['tv_shows'].append(content_data)
            elif content.content_type == 'anime':
                grouped_favorites['anime'].append(content_data)
        
        response = {
            'favorites': all_favorites,
            'grouped_favorites': grouped_favorites,
            'count': len(all_favorites),
            'count_by_type': {
                'movies': len(grouped_favorites['movies']),
                'tv_shows': len(grouped_favorites['tv_shows']),
                'anime': len(grouped_favorites['anime'])
            }
        }
        
        if content_type_filter:
            type_map = {'movie': 'movies', 'tv': 'tv_shows', 'anime': 'anime'}
            filtered_type = type_map.get(content_type_filter)
            if filtered_type:
                response['favorites'] = grouped_favorites[filtered_type]
                response['count'] = len(grouped_favorites[filtered_type])
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Favorites error: {e}")
        return jsonify({'error': 'Failed to get favorites'}), 500

# User Analytics and Insights
@users_bp.route('/api/user/analytics', methods=['GET'])
@require_auth
def get_user_analytics(current_user):
    """Get detailed user analytics and insights"""
    try:
        stats = UserAnalytics.get_user_stats(current_user.id)
        
        # Get viewing patterns
        interactions = UserInteraction.query.filter_by(user_id=current_user.id).all()
        
        # Analyze viewing patterns by time
        hourly_activity = defaultdict(int)
        daily_activity = defaultdict(int)
        
        for interaction in interactions:
            hour = interaction.timestamp.hour
            day = interaction.timestamp.strftime('%A')
            hourly_activity[hour] += 1
            daily_activity[day] += 1
        
        # Get content type distribution over time
        monthly_content_types = defaultdict(lambda: defaultdict(int))
        
        for interaction in interactions:
            content = Content.query.get(interaction.content_id)
            if content:
                month_key = interaction.timestamp.strftime('%Y-%m')
                monthly_content_types[month_key][content.content_type] += 1
        
        # Get personalization insights
        user_pref = UserPreference.query.filter_by(user_id=current_user.id).first()
        personalization_insights = {}
        
        if user_pref:
            try:
                personalization_insights = {
                    'exploration_tendency': user_pref.exploration_tendency,
                    'diversity_preference': user_pref.diversity_preference,
                    'current_mode': user_pref.current_mode,
                    'profile_strength': user_pref.profile_strength,
                    'top_genres': json.loads(user_pref.genre_preferences or '{}'),
                    'top_languages': json.loads(user_pref.language_preferences or '{}'),
                    'content_type_distribution': json.loads(user_pref.content_type_preferences or '{}')
                }
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Get recommendation performance
        recommendation_performance = {}
        if RecommendationFeedback:
            feedback = RecommendationFeedback.query.filter_by(user_id=current_user.id).all()
            
            if feedback:
                total_recommendations = len(feedback)
                successful_recommendations = len([f for f in feedback if f.was_successful])
                
                recommendation_performance = {
                    'total_recommendations': total_recommendations,
                    'successful_recommendations': successful_recommendations,
                    'success_rate': round(successful_recommendations / total_recommendations * 100, 1) if total_recommendations > 0 else 0,
                    'average_engagement': round(
                        sum([f.engagement_score for f in feedback if f.engagement_score]) / 
                        len([f for f in feedback if f.engagement_score]), 2
                    ) if any(f.engagement_score for f in feedback) else 0
                }
        
        analytics_data = {
            'user_stats': stats,
            'viewing_patterns': {
                'hourly_activity': dict(hourly_activity),
                'daily_activity': dict(daily_activity),
                'most_active_hour': max(hourly_activity, key=hourly_activity.get) if hourly_activity else None,
                'most_active_day': max(daily_activity, key=daily_activity.get) if daily_activity else None
            },
            'content_evolution': dict(monthly_content_types),
            'personalization_insights': personalization_insights,
            'recommendation_performance': recommendation_performance,
            'engagement_level': _calculate_engagement_level(stats),
            'profile_completeness': _calculate_profile_completeness(current_user),
            'discovery_suggestions': _get_discovery_suggestions(current_user.id)
        }
        
        return jsonify(analytics_data), 200
        
    except Exception as e:
        logger.error(f"Error getting user analytics: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500

# Watchlist status check
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
            'added_date': interaction.timestamp.isoformat() if interaction else None
        }), 200
        
    except Exception as e:
        logger.error(f"Check watchlist status error: {e}")
        return jsonify({'error': 'Failed to check watchlist status'}), 500

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
            
            # Update profile
            update_user_profile(current_user.id, force_update=False)
            
            # Clear cache
            if cache:
                cache.delete(f"user_profile:{current_user.id}")
            
            return jsonify({'message': 'Removed from watchlist'}), 200
        else:
            return jsonify({'message': 'Content not in watchlist'}), 404
            
    except Exception as e:
        logger.error(f"Remove from watchlist error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to remove from watchlist'}), 500

# Personalization Settings
@users_bp.route('/api/user/personalization/settings', methods=['GET'])
@require_auth
def get_personalization_settings(current_user):
    """Get user's personalization settings"""
    try:
        user_pref = UserPreference.query.filter_by(user_id=current_user.id).first()
        
        if not user_pref:
            # Create default preferences
            user_pref = UserPreference(
                user_id=current_user.id,
                exploration_tendency=0.5,
                diversity_preference=0.5,
                recency_bias=0.5,
                current_mode='discovery_mode'
            )
            db.session.add(user_pref)
            db.session.commit()
        
        settings = {
            'exploration_tendency': user_pref.exploration_tendency,
            'diversity_preference': user_pref.diversity_preference,
            'recency_bias': user_pref.recency_bias,
            'current_mode': user_pref.current_mode,
            'profile_strength': user_pref.profile_strength,
            'confidence_score': user_pref.confidence_score,
            'last_updated': user_pref.updated_at.isoformat() if user_pref.updated_at else None
        }
        
        return jsonify(settings), 200
        
    except Exception as e:
        logger.error(f"Error getting personalization settings: {e}")
        return jsonify({'error': 'Failed to get personalization settings'}), 500

@users_bp.route('/api/user/personalization/settings', methods=['PUT'])
@require_auth
def update_personalization_settings(current_user):
    """Update user's personalization settings"""
    try:
        data = request.get_json()
        
        user_pref = UserPreference.query.filter_by(user_id=current_user.id).first()
        
        if not user_pref:
            user_pref = UserPreference(user_id=current_user.id)
            db.session.add(user_pref)
        
        # Update settings
        if 'exploration_tendency' in data:
            user_pref.exploration_tendency = max(0, min(1, float(data['exploration_tendency'])))
        
        if 'diversity_preference' in data:
            user_pref.diversity_preference = max(0, min(1, float(data['diversity_preference'])))
        
        if 'recency_bias' in data:
            user_pref.recency_bias = max(0, min(1, float(data['recency_bias'])))
        
        if 'current_mode' in data and data['current_mode'] in ['discovery_mode', 'comfort_mode', 'binge_mode', 'selective_mode']:
            user_pref.current_mode = data['current_mode']
        
        user_pref.updated_at = datetime.utcnow()
        db.session.commit()
        
        # Force profile update
        update_user_profile(current_user.id, force_update=True)
        
        return jsonify({'message': 'Personalization settings updated successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error updating personalization settings: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update personalization settings'}), 500

# Recommendation Feedback
@users_bp.route('/api/user/recommendations/feedback', methods=['POST'])
@require_auth
def submit_recommendation_feedback(current_user):
    """Submit feedback for a recommendation"""
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'feedback_type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Record feedback
        success = record_recommendation_feedback(
            current_user.id,
            data['content_id'],
            data['feedback_type'],
            data.get('rating')
        )
        
        if success:
            # Update profile to learn from feedback
            update_user_profile(current_user.id, force_update=False)
            
            return jsonify({'message': 'Feedback recorded successfully'}), 200
        else:
            return jsonify({'error': 'Failed to record feedback'}), 400
            
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        return jsonify({'error': 'Failed to record feedback'}), 500

# Refresh Personalization Profile
@users_bp.route('/api/user/personalization/refresh', methods=['POST'])
@require_auth
def refresh_personalization_profile(current_user):
    """Force refresh of user's personalization profile"""
    try:
        # Force update profile
        profile = update_user_profile(current_user.id, force_update=True)
        
        # Clear all caches
        if cache:
            cache.delete(f"user_profile:{current_user.id}")
            cache.delete(f"advanced_profile:{current_user.id}")
            cache.delete(f"recommendations:{current_user.id}")
        
        return jsonify({
            'message': 'Personalization profile refreshed successfully',
            'profile_strength': profile.get('profile_strength', 0),
            'confidence_score': profile.get('confidence_score', 0)
        }), 200
        
    except Exception as e:
        logger.error(f"Error refreshing personalization profile: {e}")
        return jsonify({'error': 'Failed to refresh personalization profile'}), 500

# Helper functions
def _calculate_engagement_level(stats):
    """Calculate user engagement level"""
    score = 0
    
    # Base activity score
    if stats['total_interactions'] > 100:
        score += 40
    elif stats['total_interactions'] > 50:
        score += 30
    elif stats['total_interactions'] > 20:
        score += 20
    elif stats['total_interactions'] > 5:
        score += 10
    
    # Diversity score
    if stats['discovery_score'] > 0.7:
        score += 20
    elif stats['discovery_score'] > 0.5:
        score += 15
    elif stats['discovery_score'] > 0.3:
        score += 10
    
    # Rating activity
    if stats['ratings_given'] > 20:
        score += 20
    elif stats['ratings_given'] > 10:
        score += 15
    elif stats['ratings_given'] > 5:
        score += 10
    
    # Consistency (viewing streak)
    if stats['viewing_streak'] > 30:
        score += 20
    elif stats['viewing_streak'] > 14:
        score += 15
    elif stats['viewing_streak'] > 7:
        score += 10
    
    return min(score, 100)

def _calculate_profile_completeness(user):
    """Calculate how complete user's profile is"""
    score = 0
    
    if user.preferred_languages and json.loads(user.preferred_languages or '[]'):
        score += 25
    
    if user.preferred_genres and json.loads(user.preferred_genres or '[]'):
        score += 25
    
    if user.location:
        score += 20
    
    if user.avatar_url:
        score += 10
    
    # Check if user has interactions
    interaction_count = UserInteraction.query.filter_by(user_id=user.id).count()
    if interaction_count > 10:
        score += 20
    elif interaction_count > 5:
        score += 10
    
    return score

def _get_discovery_suggestions(user_id):
    """Get suggestions to help user discover new content"""
    suggestions = []
    
    # Get user's interaction patterns
    interactions = UserInteraction.query.filter_by(user_id=user_id).all()
    
    if not interactions:
        suggestions.append("Start by rating some movies and shows you've watched!")
        suggestions.append("Add content to your watchlist to get better recommendations")
        suggestions.append("Explore different genres to help us understand your preferences")
        return suggestions
    
    # Analyze what user hasn't explored
    content_ids = [i.content_id for i in interactions]
    contents = Content.query.filter(Content.id.in_(content_ids)).all()
    
    # Check content types
    content_types = set([c.content_type for c in contents])
    
    if 'anime' not in content_types:
        suggestions.append("Try exploring anime - you might discover something new!")
    
    if 'tv' not in content_types:
        suggestions.append("Consider watching some TV series for longer storytelling")
    
    # Check language diversity
    all_languages = set()
    for content in contents:
        try:
            languages = json.loads(content.languages or '[]')
            all_languages.update(languages)
        except (json.JSONDecodeError, TypeError):
            pass
    
    if 'hindi' not in [lang.lower() for lang in all_languages]:
        suggestions.append("Explore Bollywood movies for great storytelling")
    
    if 'telugu' not in [lang.lower() for lang in all_languages]:
        suggestions.append("Check out Telugu cinema for amazing regional content")
    
    if 'korean' not in [lang.lower() for lang in all_languages]:
        suggestions.append("Try Korean dramas for unique storytelling styles")
    
    # Check if user needs to rate more
    rating_count = len([i for i in interactions if i.rating])
    if rating_count < 10:
        suggestions.append("Rate more content to improve recommendation accuracy")
    
    return suggestions[:4]  # Return top 4 suggestions