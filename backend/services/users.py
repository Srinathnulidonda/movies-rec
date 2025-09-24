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
import hashlib
import time

# Import enhanced personalized recommendation functions
from services.personalized import (
    get_personalized_recommendations_for_user,
    update_user_profile,
    record_recommendation_feedback,
    StoryAnalyzer,  # Import for story analysis
    DeepUserProfiler  # Import for deep profiling
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
StoryProfile = None
UserStoryPreference = None
ContentSimilarity = None
http_session = None
cache = None
app = None

def init_users(flask_app, database, models, services):
    """Initialize users module with app context and models"""
    global db, User, Content, UserInteraction, AnonymousInteraction, Review
    global UserPreference, RecommendationFeedback, UserSession
    global StoryProfile, UserStoryPreference, ContentSimilarity
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
    StoryProfile = models.get('StoryProfile')
    UserStoryPreference = models.get('UserStoryPreference')
    ContentSimilarity = models.get('ContentSimilarity')
    
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
    """Enhanced session tracking with mood detection"""
    try:
        # Generate or get session ID
        session_id = request_obj.headers.get('X-Session-Id')
        if not session_id:
            # Generate session ID from user agent and timestamp
            user_agent = request_obj.headers.get('User-Agent', '')
            session_id = hashlib.md5(f"{user_id}{user_agent}{time.time()}".encode()).hexdigest()
        
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
                user_agent=request_obj.headers.get('User-Agent', ''),
                user_mode=_detect_user_mode(user_id),
                dominant_mood=_detect_user_mood(user_id)
            )
            db.session.add(session)
        
        # Update session activity
        session.interactions_count += 1
        session.end_time = datetime.utcnow()
        
        if session.start_time:
            session.duration = (session.end_time - session.start_time).seconds
        
        # Track search queries in session
        if request_obj.path == '/api/search':
            query = request_obj.args.get('query', '')
            if query:
                search_queries = json.loads(session.search_queries or '[]')
                search_queries.append(query)
                session.search_queries = json.dumps(search_queries[-10:])  # Keep last 10
        
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

def _detect_user_mode(user_id: int) -> str:
    """Detect user's current mode based on recent activity"""
    try:
        # Get recent interactions
        recent_interactions = UserInteraction.query.filter_by(
            user_id=user_id
        ).order_by(UserInteraction.timestamp.desc()).limit(10).all()
        
        if not recent_interactions:
            return 'discovery_mode'
        
        # Check for binge pattern (multiple views in short time)
        if len(recent_interactions) >= 3:
            time_gaps = []
            for i in range(1, min(len(recent_interactions), 4)):
                gap = (recent_interactions[i-1].timestamp - recent_interactions[i].timestamp).seconds / 3600
                time_gaps.append(gap)
            
            if time_gaps and max(time_gaps) < 3:  # All within 3 hours
                return 'binge_mode'
        
        # Check for selective pattern (high ratings, few interactions)
        ratings = [i.rating for i in recent_interactions if i.rating]
        if ratings and len(ratings) < 5 and sum(ratings) / len(ratings) > 4:
            return 'selective_mode'
        
        # Check exploration (diverse content types/genres)
        content_ids = [i.content_id for i in recent_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        
        genres = set()
        for content in contents:
            try:
                content_genres = json.loads(content.genres or '[]')
                genres.update(content_genres)
            except:
                pass
        
        if len(genres) > 7:
            return 'discovery_mode'
        
        return 'comfort_mode'
        
    except Exception as e:
        logger.error(f"Error detecting user mode: {e}")
        return 'discovery_mode'

def _detect_user_mood(user_id: int) -> str:
    """Detect user's current mood based on time and recent content"""
    try:
        hour = datetime.utcnow().hour
        
        # Time-based default moods
        if 5 <= hour < 9:
            base_mood = 'energetic'
        elif 9 <= hour < 12:
            base_mood = 'focused'
        elif 12 <= hour < 17:
            base_mood = 'productive'
        elif 17 <= hour < 21:
            base_mood = 'relaxed'
        elif 21 <= hour < 24:
            base_mood = 'contemplative'
        else:
            base_mood = 'sleepy'
        
        # Adjust based on recent content emotional tones
        recent_interaction = UserInteraction.query.filter_by(
            user_id=user_id
        ).order_by(UserInteraction.timestamp.desc()).first()
        
        if recent_interaction and StoryProfile:
            story_profile = StoryProfile.query.filter_by(
                content_id=recent_interaction.content_id
            ).first()
            
            if story_profile:
                emotional_tone = story_profile.emotional_tone
                
                # Map emotional tone to mood
                tone_mood_map = {
                    'uplifting': 'happy',
                    'dark': 'serious',
                    'emotional': 'sentimental',
                    'neutral': 'relaxed',
                    'intense': 'excited'
                }
                
                if emotional_tone in tone_mood_map:
                    return tone_mood_map[emotional_tone]
        
        return base_mood
        
    except Exception as e:
        logger.error(f"Error detecting user mood: {e}")
        return 'neutral'

class EnhancedUserAnalytics:
    """Enhanced user analytics with story preference tracking"""
    
    @staticmethod
    def get_comprehensive_stats(user_id: int) -> dict:
        """Get comprehensive user statistics including story preferences"""
        try:
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            
            if not interactions:
                return EnhancedUserAnalytics._get_empty_stats()
            
            # Basic stats
            stats = {
                'total_interactions': len(interactions),
                'content_watched': len([i for i in interactions if i.interaction_type == 'view']),
                'favorites': len([i for i in interactions if i.interaction_type == 'favorite']),
                'watchlist_items': len([i for i in interactions if i.interaction_type == 'watchlist']),
                'ratings_given': len([i for i in interactions if i.interaction_type == 'rating']),
                'rewatched': len([i for i in interactions if i.interaction_type == 'rewatch'])
            }
            
            # Get content for analysis
            content_ids = list(set([i.content_id for i in interactions]))
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            content_map = {c.id: c for c in contents}
            
            # Genre and theme analysis
            genre_counts = defaultdict(int)
            theme_counts = defaultdict(int)
            narrative_styles = defaultdict(int)
            emotional_tones = defaultdict(int)
            
            for interaction in interactions:
                content = content_map.get(interaction.content_id)
                if content:
                    # Genres
                    try:
                        genres = json.loads(content.genres or '[]')
                        for genre in genres:
                            genre_counts[genre.lower()] += 1
                    except:
                        pass
                    
                    # Story analysis
                    if StoryProfile:
                        story_profile = StoryProfile.query.filter_by(content_id=content.id).first()
                        if story_profile:
                            try:
                                themes = json.loads(story_profile.themes or '[]')
                                for theme in themes:
                                    theme_counts[theme] += 1
                            except:
                                pass
                            
                            if story_profile.narrative_style:
                                narrative_styles[story_profile.narrative_style] += 1
                            
                            if story_profile.emotional_tone:
                                emotional_tones[story_profile.emotional_tone] += 1
            
            # Add story preferences to stats
            stats['most_watched_genre'] = max(genre_counts, key=genre_counts.get) if genre_counts else None
            stats['top_themes'] = [theme for theme, _ in Counter(theme_counts).most_common(5)]
            stats['preferred_narrative'] = max(narrative_styles, key=narrative_styles.get) if narrative_styles else 'linear'
            stats['emotional_preference'] = max(emotional_tones, key=emotional_tones.get) if emotional_tones else 'neutral'
            
            # Calculate advanced metrics
            stats['genre_diversity'] = len(genre_counts) / max(len(interactions), 1)
            stats['theme_diversity'] = len(theme_counts) / max(len(interactions), 1)
            stats['viewing_streak'] = EnhancedUserAnalytics._calculate_viewing_streak(interactions)
            stats['discovery_score'] = EnhancedUserAnalytics._calculate_discovery_score(interactions, contents)
            stats['binge_tendency'] = EnhancedUserAnalytics._calculate_binge_tendency(interactions)
            stats['quality_threshold'] = EnhancedUserAnalytics._calculate_quality_threshold(interactions, content_map)
            
            # Personalization readiness
            stats['personalization_ready'] = stats['total_interactions'] >= 10
            stats['deep_profiling_ready'] = stats['total_interactions'] >= 20
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting comprehensive stats: {e}")
            return EnhancedUserAnalytics._get_empty_stats()
    
    @staticmethod
    def _calculate_binge_tendency(interactions):
        """Calculate user's binge-watching tendency"""
        if len(interactions) < 3:
            return 0.0
        
        # Sort by timestamp
        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
        
        binge_sessions = 0
        current_session = [sorted_interactions[0]]
        
        for i in range(1, len(sorted_interactions)):
            time_gap = (sorted_interactions[i].timestamp - sorted_interactions[i-1].timestamp).seconds / 3600
            
            if time_gap <= 3:  # Within 3 hours
                current_session.append(sorted_interactions[i])
            else:
                if len(current_session) >= 3:
                    binge_sessions += 1
                current_session = [sorted_interactions[i]]
        
        if len(current_session) >= 3:
            binge_sessions += 1
        
        return min(binge_sessions / max(len(interactions) / 10, 1), 1.0)
    
    @staticmethod
    def _calculate_quality_threshold(interactions, content_map):
        """Calculate user's quality threshold"""
        ratings = []
        for interaction in interactions:
            content = content_map.get(interaction.content_id)
            if content and content.rating:
                ratings.append(content.rating)
        
        if ratings:
            return sum(ratings) / len(ratings)
        return 7.0
    
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
        all_themes = set()
        
        for content in contents:
            try:
                genres = json.loads(content.genres or '[]')
                all_genres.update([g.lower() for g in genres])
            except:
                pass
            
            if StoryProfile:
                story_profile = StoryProfile.query.filter_by(content_id=content.id).first()
                if story_profile:
                    try:
                        themes = json.loads(story_profile.themes or '[]')
                        all_themes.update(themes)
                    except:
                        pass
        
        genre_diversity = len(all_genres) / 20.0 if all_genres else 0
        theme_diversity = len(all_themes) / 30.0 if all_themes else 0
        
        return min((genre_diversity + theme_diversity) / 2, 1.0)
    
    @staticmethod
    def _get_empty_stats() -> dict:
        """Return empty stats structure"""
        return {
            'total_interactions': 0,
            'content_watched': 0,
            'favorites': 0,
            'watchlist_items': 0,
            'ratings_given': 0,
            'rewatched': 0,
            'most_watched_genre': None,
            'top_themes': [],
            'preferred_narrative': 'linear',
            'emotional_preference': 'neutral',
            'genre_diversity': 0.0,
            'theme_diversity': 0.0,
            'viewing_streak': 0,
            'discovery_score': 0.0,
            'binge_tendency': 0.0,
            'quality_threshold': 7.0,
            'personalization_ready': False,
            'deep_profiling_ready': False
        }

# Enhanced User Profile Management
@users_bp.route('/api/user/profile', methods=['GET'])
@require_auth
def get_user_profile(current_user):
    """Get comprehensive user profile with deep insights"""
    try:
        # Get comprehensive stats
        stats = EnhancedUserAnalytics.get_comprehensive_stats(current_user.id)
        
        # Get deep profile if ready
        deep_profile = None
        if stats['deep_profiling_ready']:
            deep_profile = update_user_profile(current_user.id, force_update=False)
        
        # Get recent activity with story insights
        recent_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id
        ).order_by(UserInteraction.timestamp.desc()).limit(10).all()
        
        recent_activity = []
        for interaction in recent_interactions:
            content = Content.query.get(interaction.content_id)
            if content:
                activity_item = {
                    'content_id': content.id,
                    'content_title': content.title,
                    'content_type': content.content_type,
                    'interaction_type': interaction.interaction_type,
                    'timestamp': interaction.timestamp.isoformat(),
                    'rating': interaction.rating
                }
                
                # Add story insights
                if StoryProfile:
                    story_profile = StoryProfile.query.filter_by(content_id=content.id).first()
                    if story_profile:
                        activity_item['story_insights'] = {
                            'emotional_tone': story_profile.emotional_tone,
                            'narrative_style': story_profile.narrative_style,
                            'themes': json.loads(story_profile.themes or '[]')[:3]
                        }
                
                recent_activity.append(activity_item)
        
        # Get personalization status
        user_pref = UserPreference.query.filter_by(user_id=current_user.id).first()
        user_story_pref = UserStoryPreference.query.filter_by(user_id=current_user.id).first() if UserStoryPreference else None
        
        personalization_status = {
            'enabled': user_pref is not None,
            'profile_strength': user_pref.profile_strength if user_pref else 0,
            'confidence_score': user_pref.confidence_score if user_pref else 0,
            'current_mode': user_pref.current_mode if user_pref else _detect_user_mode(current_user.id),
            'current_mood': user_pref.current_mood if user_pref else _detect_user_mood(current_user.id),
            'personalization_level': user_pref.personalization_level if user_pref else 1,
            'story_profiling_enabled': user_story_pref is not None,
            'last_updated': user_pref.updated_at.isoformat() if user_pref and user_pref.updated_at else None
        }
        
        # Get story preferences if available
        story_preferences = {}
        if user_story_pref:
            story_preferences = {
                'preferred_themes': json.loads(user_story_pref.preferred_themes or '{}'),
                'preferred_narratives': json.loads(user_story_pref.preferred_narratives or '{}'),
                'preferred_endings': json.loads(user_story_pref.preferred_endings or '{}'),
                'ideal_complexity': {
                    'plot': user_story_pref.ideal_plot_complexity,
                    'character': user_story_pref.ideal_character_depth,
                    'narrative': user_story_pref.ideal_narrative_depth
                },
                'emotional_intensity': user_story_pref.emotional_intensity_preference
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
            'personalization_status': personalization_status,
            'story_preferences': story_preferences,
            'deep_insights': deep_profile.get('psychological_profile', {}) if deep_profile else {},
            'interest_evolution': deep_profile.get('interest_evolution', {}) if deep_profile else {}
        }
        
        return jsonify(profile_data), 200
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return jsonify({'error': 'Failed to get user profile'}), 500

# Ultra-Personalized Recommendations
@users_bp.route('/api/user/recommendations', methods=['GET'])
@require_auth
def get_user_recommendations(current_user):
    """Get ultra-personalized recommendations with story matching"""
    try:
        content_type = request.args.get('type')  # movie, tv, anime
        limit = min(int(request.args.get('limit', 20)), 50)
        mood = request.args.get('mood')  # User can specify mood
        
        # Get personalized recommendations with story matching
        recommendations = get_personalized_recommendations_for_user(
            current_user.id, content_type, limit, mood
        )
        
        # Track recommendation request
        _track_recommendation_request(current_user.id, content_type, mood)
        
        return jsonify(recommendations), 200
        
    except Exception as e:
        logger.error(f"Error getting personalized recommendations: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@users_bp.route('/api/user/recommendations/story-based', methods=['GET'])
@require_auth
def get_story_based_recommendations(current_user):
    """Get recommendations based purely on story preferences"""
    try:
        content_id = request.args.get('content_id')  # Base content for story matching
        limit = min(int(request.args.get('limit', 10)), 20)
        
        # Get story-focused recommendations
        recommendations = get_personalized_recommendations_for_user(
            current_user.id, 
            content_type=None, 
            limit=limit,
            mood='story_focused'  # Special mood for story-based
        )
        
        # Filter to only story-matched items
        story_matched = [
            rec for rec in recommendations.get('recommendations', [])
            if 'story_matching' in rec.get('methods_used', [])
        ]
        
        return jsonify({
            'story_recommendations': story_matched,
            'metadata': recommendations.get('recommendation_metadata', {})
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting story-based recommendations: {e}")
        return jsonify({'error': 'Failed to get story recommendations'}), 500

@users_bp.route('/api/user/recommendations/mood-based', methods=['GET'])
@require_auth
def get_mood_based_recommendations(current_user):
    """Get recommendations based on current mood"""
    try:
        mood = request.args.get('mood', _detect_user_mood(current_user.id))
        limit = min(int(request.args.get('limit', 15)), 30)
        
        # Map mood to content preferences
        mood_mapping = {
            'happy': {'type': None, 'mood': 'happy'},
            'sad': {'type': None, 'mood': 'sad'},
            'excited': {'type': 'movie', 'mood': 'excited'},
            'relaxed': {'type': 'tv', 'mood': 'relaxed'},
            'contemplative': {'type': None, 'mood': 'contemplative'},
            'adventurous': {'type': None, 'mood': 'adventurous'},
            'romantic': {'type': None, 'mood': 'romantic'},
            'thrilled': {'type': 'movie', 'mood': 'thrilled'}
        }
        
        mood_config = mood_mapping.get(mood, {'type': None, 'mood': mood})
        
        recommendations = get_personalized_recommendations_for_user(
            current_user.id,
            content_type=mood_config['type'],
            limit=limit,
            mood=mood_config['mood']
        )
        
        return jsonify({
            'mood': mood,
            'recommendations': recommendations.get('recommendations', []),
            'mood_insights': {
                'detected_mood': mood,
                'content_alignment': 'optimized',
                'emotional_tone_preference': recommendations.get('user_insights', {})
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting mood-based recommendations: {e}")
        return jsonify({'error': 'Failed to get mood recommendations'}), 500

# Enhanced Interaction Recording
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
        
        # Handle special interaction types
        if interaction_type == 'rewatch':
            # Mark as rewatched
            existing = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=content_id,
                interaction_type='view'
            ).first()
            
            if existing:
                # Create rewatch interaction
                interaction_type = 'rewatch'
        
        # Enhanced interaction metadata
        interaction_metadata = {
            'from_recommendation': data.get('from_recommendation', False),
            'recommendation_score': data.get('recommendation_score'),
            'recommendation_method': data.get('recommendation_method'),
            'story_match_score': data.get('story_match_score'),
            'user_mood': data.get('mood', _detect_user_mood(current_user.id)),
            'user_mode': _detect_user_mode(current_user.id),
            'device_type': _detect_device_type(request.headers.get('User-Agent', '')),
            'time_of_day': datetime.utcnow().hour,
            'day_of_week': datetime.utcnow().weekday(),
            'search_query': data.get('search_query'),  # If from search
            'user_agent': request.headers.get('User-Agent', ''),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Check for duplicate interactions (except view/rewatch)
        if interaction_type not in ['view', 'rewatch', 'search']:
            existing = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=content_id,
                interaction_type=interaction_type
            ).first()
            
            if existing and interaction_type != 'remove_watchlist':
                return jsonify({'message': f'Already marked as {interaction_type}'}), 200
        
        # Create interaction
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
        
        # Trigger story analysis for new content
        _trigger_story_analysis(content_id)
        
        # Update user profile (async if possible)
        update_user_profile(current_user.id, force_update=False)
        
        # Clear user caches
        if cache:
            cache.delete(f"user_profile:{current_user.id}")
            cache.delete(f"deep_profile:{current_user.id}")
        
        return jsonify({
            'message': 'Interaction recorded successfully',
            'interaction_id': interaction.id,
            'profile_update_triggered': True
        }), 201
        
    except Exception as e:
        logger.error(f"Interaction recording error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record interaction'}), 500

# Story Preferences Management
@users_bp.route('/api/user/story-preferences', methods=['GET'])
@require_auth
def get_story_preferences(current_user):
    """Get user's story preferences"""
    try:
        story_pref = UserStoryPreference.query.filter_by(user_id=current_user.id).first()
        
        if not story_pref:
            return jsonify({
                'message': 'No story preferences found. Keep watching to build your profile!',
                'preferences': {}
            }), 200
        
        preferences = {
            'preferred_themes': json.loads(story_pref.preferred_themes or '{}'),
            'avoided_themes': json.loads(story_pref.avoided_themes or '{}'),
            'preferred_narratives': json.loads(story_pref.preferred_narratives or '{}'),
            'preferred_endings': json.loads(story_pref.preferred_endings or '{}'),
            'complexity_preferences': {
                'plot': story_pref.ideal_plot_complexity,
                'character': story_pref.ideal_character_depth,
                'narrative': story_pref.ideal_narrative_depth
            },
            'emotional_preferences': {
                'preferred_tones': json.loads(story_pref.preferred_emotional_tones or '{}'),
                'intensity': story_pref.emotional_intensity_preference
            },
            'story_types': json.loads(story_pref.story_type_weights or '{}'),
            'confidence_score': story_pref.confidence_score,
            'last_updated': story_pref.last_updated.isoformat() if story_pref.last_updated else None
        }
        
        return jsonify(preferences), 200
        
    except Exception as e:
        logger.error(f"Error getting story preferences: {e}")
        return jsonify({'error': 'Failed to get story preferences'}), 500

@users_bp.route('/api/user/story-preferences', methods=['PUT'])
@require_auth
def update_story_preferences(current_user):
    """Manually update story preferences"""
    try:
        data = request.get_json()
        
        story_pref = UserStoryPreference.query.filter_by(user_id=current_user.id).first()
        
        if not story_pref:
            story_pref = UserStoryPreference(user_id=current_user.id)
            db.session.add(story_pref)
        
        # Update preferences
        if 'avoided_themes' in data:
            story_pref.avoided_themes = json.dumps(data['avoided_themes'])
        
        if 'complexity_preferences' in data:
            complexity = data['complexity_preferences']
            story_pref.ideal_plot_complexity = complexity.get('plot', 0.5)
            story_pref.ideal_character_depth = complexity.get('character', 0.5)
            story_pref.ideal_narrative_depth = complexity.get('narrative', 0.5)
        
        if 'emotional_intensity' in data:
            story_pref.emotional_intensity_preference = data['emotional_intensity']
        
        story_pref.last_updated = datetime.utcnow()
        db.session.commit()
        
        # Force profile update
        update_user_profile(current_user.id, force_update=True)
        
        return jsonify({'message': 'Story preferences updated successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error updating story preferences: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update story preferences'}), 500

# Enhanced Analytics
@users_bp.route('/api/user/analytics/deep', methods=['GET'])
@require_auth
def get_deep_analytics(current_user):
    """Get deep analytics with story and mood insights"""
    try:
        # Get comprehensive stats
        stats = EnhancedUserAnalytics.get_comprehensive_stats(current_user.id)
        
        # Get deep profile
        deep_profile = update_user_profile(current_user.id, force_update=False)
        
        # Get recommendation performance
        recommendation_performance = _get_recommendation_performance(current_user.id)
        
        # Get mood patterns
        mood_patterns = deep_profile.get('mood_patterns', {}) if deep_profile else {}
        
        # Get story journey
        story_journey = _get_story_journey(current_user.id)
        
        analytics_data = {
            'user_stats': stats,
            'deep_insights': {
                'psychological_profile': deep_profile.get('psychological_profile', {}) if deep_profile else {},
                'discovery_profile': deep_profile.get('discovery_profile', {}) if deep_profile else {},
                'interest_evolution': deep_profile.get('interest_evolution', {}) if deep_profile else {},
                'behavioral_patterns': deep_profile.get('behavioral_patterns', {}) if deep_profile else {}
            },
            'story_insights': {
                'top_themes': stats.get('top_themes', []),
                'narrative_preference': stats.get('preferred_narrative', 'linear'),
                'emotional_preference': stats.get('emotional_preference', 'neutral'),
                'story_journey': story_journey
            },
            'mood_patterns': mood_patterns,
            'recommendation_performance': recommendation_performance,
            'personalization_metrics': {
                'profile_strength': deep_profile.get('profile_strength', 0) if deep_profile else 0,
                'confidence_score': deep_profile.get('confidence_score', 0) if deep_profile else 0,
                'interaction_count': deep_profile.get('interaction_count', 0) if deep_profile else 0
            }
        }
        
        return jsonify(analytics_data), 200
        
    except Exception as e:
        logger.error(f"Error getting deep analytics: {e}")
        return jsonify({'error': 'Failed to get deep analytics'}), 500

# Helper functions
def _trigger_story_analysis(content_id: int):
    """Trigger story analysis for content if not exists"""
    try:
        if StoryProfile:
            existing = StoryProfile.query.filter_by(content_id=content_id).first()
            if not existing:
                content = Content.query.get(content_id)
                if content:
                    # Initialize story analyzer
                    analyzer = StoryAnalyzer()
                    analyzer.analyze_story(content)
    except Exception as e:
        logger.error(f"Error triggering story analysis: {e}")

def _track_recommendation_request(user_id: int, content_type: Optional[str], mood: Optional[str]):
    """Track recommendation request for analytics"""
    try:
        # This could be stored in a separate analytics table
        # For now, we'll just log it
        logger.info(f"Recommendation request: user={user_id}, type={content_type}, mood={mood}")
    except Exception as e:
        logger.error(f"Error tracking recommendation request: {e}")

def _get_recommendation_performance(user_id: int) -> Dict:
    """Get recommendation performance metrics"""
    try:
        if not RecommendationFeedback:
            return {}
        
        feedbacks = RecommendationFeedback.query.filter_by(user_id=user_id).all()
        
        if not feedbacks:
            return {
                'total_recommendations': 0,
                'success_rate': 0,
                'engagement_rate': 0,
                'satisfaction_score': 0
            }
        
        total = len(feedbacks)
        successful = len([f for f in feedbacks if f.was_successful])
        engaged = len([f for f in feedbacks if f.engagement_score and f.engagement_score > 0.5])
        satisfaction_scores = [f.satisfaction_score for f in feedbacks if f.satisfaction_score]
        
        return {
            'total_recommendations': total,
            'success_rate': round(successful / total * 100, 1) if total > 0 else 0,
            'engagement_rate': round(engaged / total * 100, 1) if total > 0 else 0,
            'satisfaction_score': round(sum(satisfaction_scores) / len(satisfaction_scores), 2) if satisfaction_scores else 0,
            'story_match_accuracy': _calculate_story_match_accuracy(feedbacks)
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendation performance: {e}")
        return {}

def _calculate_story_match_accuracy(feedbacks: List) -> float:
    """Calculate how well story matching is working"""
    story_matched = [f for f in feedbacks if f.story_match_score and f.story_match_score > 0]
    if not story_matched:
        return 0.0
    
    successful_matches = [f for f in story_matched if f.was_successful]
    return round(len(successful_matches) / len(story_matched) * 100, 1)

def _get_story_journey(user_id: int) -> List[Dict]:
    """Get user's story preference journey over time"""
    try:
        # Get interactions over last 3 months
        three_months_ago = datetime.utcnow() - timedelta(days=90)
        interactions = UserInteraction.query.filter(
            UserInteraction.user_id == user_id,
            UserInteraction.timestamp >= three_months_ago
        ).order_by(UserInteraction.timestamp).all()
        
        if not interactions:
            return []
        
        # Group by month
        monthly_themes = defaultdict(list)
        monthly_tones = defaultdict(list)
        
        for interaction in interactions:
            month_key = interaction.timestamp.strftime('%Y-%m')
            
            if StoryProfile:
                story_profile = StoryProfile.query.filter_by(
                    content_id=interaction.content_id
                ).first()
                
                if story_profile:
                    try:
                        themes = json.loads(story_profile.themes or '[]')
                        monthly_themes[month_key].extend(themes)
                    except:
                        pass
                    
                    if story_profile.emotional_tone:
                        monthly_tones[month_key].append(story_profile.emotional_tone)
        
        # Build journey
        journey = []
        for month in sorted(monthly_themes.keys()):
            journey.append({
                'month': month,
                'top_themes': Counter(monthly_themes[month]).most_common(3),
                'dominant_tone': Counter(monthly_tones[month]).most_common(1)[0][0] if monthly_tones[month] else 'neutral'
            })
        
        return journey
        
    except Exception as e:
        logger.error(f"Error getting story journey: {e}")
        return []