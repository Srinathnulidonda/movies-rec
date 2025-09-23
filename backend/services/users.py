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
http_session = None
cache = None
app = None

def init_users(flask_app, database, models, services):
    """Initialize users module with app context and models"""
    global db, User, Content, UserInteraction, AnonymousInteraction, Review
    global http_session, cache, app
    
    app = flask_app
    db = database
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    AnonymousInteraction = models.get('AnonymousInteraction')
    Review = models.get('Review')
    
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
            db.session.commit()
            
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

class UserAnalytics:
    """Enhanced user analytics and insights"""
    
    @staticmethod
    def get_user_stats(user_id: int) -> dict:
        """Get comprehensive user statistics"""
        try:
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            
            if not interactions:
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
                    'discovery_score': 0.0
                }
            
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
            
            # Genre analysis
            genre_counts = defaultdict(int)
            content_type_counts = defaultdict(int)
            
            for interaction in interactions:
                content = content_map.get(interaction.content_id)
                if content:
                    # Count content types
                    content_type_counts[content.content_type] += 1
                    
                    # Count genres
                    try:
                        genres = json.loads(content.genres or '[]')
                        for genre in genres:
                            genre_counts[genre.lower()] += 1
                    except (json.JSONDecodeError, TypeError):
                        pass
            
            stats['most_watched_genre'] = max(genre_counts, key=genre_counts.get) if genre_counts else None
            stats['preferred_content_type'] = max(content_type_counts, key=content_type_counts.get) if content_type_counts else None
            
            # Viewing streak (consecutive days with interactions)
            stats['viewing_streak'] = UserAnalytics._calculate_viewing_streak(interactions)
            
            # Discovery score (how much new content user explores)
            stats['discovery_score'] = UserAnalytics._calculate_discovery_score(interactions, contents)
            
            # Monthly activity
            stats['monthly_activity'] = UserAnalytics._get_monthly_activity(interactions)
            
            # Content quality preference
            quality_ratings = [content_map[i.content_id].rating for i in interactions 
                             if i.content_id in content_map and content_map[i.content_id].rating]
            stats['preferred_content_quality'] = round(sum(quality_ratings) / len(quality_ratings), 1) if quality_ratings else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {}
    
    @staticmethod
    def _calculate_viewing_streak(interactions):
        """Calculate consecutive days with viewing activity"""
        if not interactions:
            return 0
        
        # Group interactions by date
        dates = set()
        for interaction in interactions:
            dates.add(interaction.timestamp.date())
        
        if not dates:
            return 0
        
        # Sort dates and find streak
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
        
        # Calculate genre diversity
        all_genres = set()
        for content in contents:
            try:
                genres = json.loads(content.genres or '[]')
                all_genres.update([g.lower() for g in genres])
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Calculate popularity distribution
        popularities = [c.popularity for c in contents if c.popularity]
        avg_popularity = sum(popularities) / len(popularities) if popularities else 100
        
        # Discovery score based on genre diversity and exploring less popular content
        genre_diversity = len(all_genres) / 20.0 if all_genres else 0  # Normalize by expected max genres
        popularity_exploration = max(0, (200 - avg_popularity) / 200) if avg_popularity else 0.5
        
        return min((genre_diversity + popularity_exploration) / 2, 1.0)
    
    @staticmethod
    def _get_monthly_activity(interactions):
        """Get monthly activity breakdown"""
        monthly_counts = defaultdict(int)
        
        for interaction in interactions:
            month_key = interaction.timestamp.strftime('%Y-%m')
            monthly_counts[month_key] += 1
        
        # Get last 6 months
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

class SmartRecommendationTracker:
    """Track and learn from user interactions with recommendations"""
    
    @staticmethod
    def record_recommendation_interaction(user_id: int, content_id: int, 
                                        interaction_type: str, recommendation_context: dict = None):
        """Record how user interacts with recommended content"""
        try:
            # Create interaction with recommendation context
            interaction = UserInteraction(
                user_id=user_id,
                content_id=content_id,
                interaction_type=interaction_type,
                interaction_metadata=json.dumps(recommendation_context or {})
            )
            
            db.session.add(interaction)
            db.session.commit()
            
            # Clear user's recommendation cache if it exists
            if cache:
                cache.delete(f"user_profile:{user_id}")
                cache.delete(f"recommendations:{user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording recommendation interaction: {e}")
            db.session.rollback()
            return False
    
    @staticmethod
    def get_recommendation_effectiveness(user_id: int):
        """Analyze how effective recommendations have been for user"""
        try:
            # Get interactions that came from recommendations
            recommendation_interactions = UserInteraction.query.filter(
                UserInteraction.user_id == user_id,
                UserInteraction.interaction_metadata.isnot(None)
            ).all()
            
            if not recommendation_interactions:
                return {
                    'total_recommended': 0,
                    'clicked_recommendations': 0,
                    'click_through_rate': 0.0,
                    'favorite_from_recommendations': 0,
                    'rating_satisfaction': 0.0
                }
            
            clicked_count = 0
            favorite_count = 0
            ratings = []
            
            for interaction in recommendation_interactions:
                try:
                    metadata = json.loads(interaction.interaction_metadata or '{}')
                    
                    if metadata.get('from_recommendation'):
                        if interaction.interaction_type in ['view', 'like', 'favorite', 'watchlist']:
                            clicked_count += 1
                        
                        if interaction.interaction_type == 'favorite':
                            favorite_count += 1
                        
                        if interaction.interaction_type == 'rating' and interaction.rating:
                            ratings.append(interaction.rating)
                
                except (json.JSONDecodeError, TypeError):
                    continue
            
            total_recommended = len(recommendation_interactions)
            click_through_rate = (clicked_count / total_recommended * 100) if total_recommended > 0 else 0
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            
            return {
                'total_recommended': total_recommended,
                'clicked_recommendations': clicked_count,
                'click_through_rate': round(click_through_rate, 1),
                'favorite_from_recommendations': favorite_count,
                'rating_satisfaction': round(avg_rating, 1),
                'engagement_score': round((click_through_rate + avg_rating * 10) / 2, 1)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing recommendation effectiveness: {e}")
            return {}

# Enhanced User Profile Management
@users_bp.route('/api/user/profile', methods=['GET'])
@require_auth
def get_user_profile(current_user):
    """Get comprehensive user profile"""
    try:
        stats = UserAnalytics.get_user_stats(current_user.id)
        rec_effectiveness = SmartRecommendationTracker.get_recommendation_effectiveness(current_user.id)
        
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
            'recommendation_effectiveness': rec_effectiveness,
            'recent_activity': recent_activity
        }
        
        return jsonify(profile_data), 200
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return jsonify({'error': 'Failed to get user profile'}), 500

@users_bp.route('/api/user/profile', methods=['PUT'])
@require_auth
def update_user_profile(current_user):
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
        
        # Clear user caches
        if cache:
            cache.delete(f"user_profile:{current_user.id}")
        
        return jsonify({'message': 'Profile updated successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update profile'}), 500

# Enhanced Interaction Routes
@users_bp.route('/api/interactions', methods=['POST'])
@require_auth
def record_interaction(current_user):
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'interaction_type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Handle remove_watchlist specially
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
        
        # For adding to watchlist, check if already exists
        if data['interaction_type'] == 'watchlist':
            existing = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=data['content_id'],
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
            content_id=data['content_id'],
            interaction_type=data['interaction_type'],
            rating=data.get('rating'),
            interaction_metadata=json.dumps(interaction_metadata)
        )
        
        db.session.add(interaction)
        db.session.commit()
        
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
        # Get watchlist with sorting options
        sort_by = request.args.get('sort_by', 'added_date')  # added_date, title, rating, release_date
        order = request.args.get('order', 'desc')  # asc, desc
        
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
        
        # Apply additional sorting if needed
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
        
        # Create interaction mapping for added_date info
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
        
        return jsonify({
            'watchlist': result,
            'count': len(result),
            'sort_by': sort_by,
            'order': order
        }), 200
        
    except Exception as e:
        logger.error(f"Watchlist error: {e}")
        return jsonify({'error': 'Failed to get watchlist'}), 500

# Enhanced Favorites Management
@users_bp.route('/api/user/favorites', methods=['GET'])
@require_auth
def get_favorites(current_user):
    try:
        content_type_filter = request.args.get('type')  # movie, tv, anime
        
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
        rec_effectiveness = SmartRecommendationTracker.get_recommendation_effectiveness(current_user.id)
        
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
        
        analytics_data = {
            'user_stats': stats,
            'recommendation_effectiveness': rec_effectiveness,
            'viewing_patterns': {
                'hourly_activity': dict(hourly_activity),
                'daily_activity': dict(daily_activity),
                'most_active_hour': max(hourly_activity, key=hourly_activity.get) if hourly_activity else None,
                'most_active_day': max(daily_activity, key=daily_activity.get) if daily_activity else None
            },
            'content_evolution': dict(monthly_content_types),
            'engagement_level': UserAnalytics._calculate_engagement_level(stats),
            'recommendations': {
                'profile_completeness': UserAnalytics._calculate_profile_completeness(current_user),
                'discovery_suggestions': UserAnalytics._get_discovery_suggestions(current_user.id)
            }
        }
        
        return jsonify(analytics_data), 200
        
    except Exception as e:
        logger.error(f"Error getting user analytics: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500

# Watchlist status check (optimized)
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

# Enhanced helper methods for UserAnalytics
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
    
    return suggestions[:3]  # Return top 3 suggestions

# Add these methods to UserAnalytics class
UserAnalytics._calculate_engagement_level = staticmethod(_calculate_engagement_level)
UserAnalytics._calculate_profile_completeness = staticmethod(_calculate_profile_completeness)
UserAnalytics._get_discovery_suggestions = staticmethod(_get_discovery_suggestions)