# user/utils.py
from flask import request, jsonify
from datetime import datetime, timedelta
import json
import logging
import jwt
from functools import wraps
from collections import defaultdict, Counter
import numpy as np
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Global variables (will be set by init function)
db = None
User = None
Content = None
UserInteraction = None
Review = None
app = None
recommendation_engine = None
cache = None
content_service = None

def init_user_module(flask_app, database, models, services):
    """Initialize the user module with dependencies"""
    global db, User, Content, UserInteraction, Review, app, recommendation_engine, cache, content_service
    
    app = flask_app
    db = database
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    Review = models.get('Review')
    cache = services.get('cache')
    content_service = services.get('ContentService')
    
    # Enhanced recommendation engine detection
    try:
        # First try the new advanced personalization system
        from personalized import get_personalization_system
        personalization_system = get_personalization_system()
        
        if personalization_system and personalization_system.is_ready():
            if hasattr(personalization_system, 'recommendation_engine'):
                recommendation_engine = personalization_system.recommendation_engine
                logger.info("✅ CineBrain Advanced Personalization System connected to user module")
            else:
                # Try legacy personalized service
                from services.personalized import get_recommendation_engine
                recommendation_engine = get_recommendation_engine()
                logger.info("✅ CineBrain legacy personalized recommendation engine connected to user module")
        else:
            # Fallback to legacy
            from services.personalized import get_recommendation_engine
            recommendation_engine = get_recommendation_engine()
            logger.info("⚠️ Using CineBrain legacy recommendation engine in user module")
            
    except Exception as e:
        logger.warning(f"Could not connect to CineBrain recommendation engines: {e}")
        recommendation_engine = None

def require_auth(f):
    """Authentication decorator for user routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return '', 200
            
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No CineBrain token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': 'Invalid CineBrain token'}), 401
            
            current_user.last_active = datetime.utcnow()
            try:
                db.session.commit()
            except Exception as e:
                logger.warning(f"Failed to update CineBrain user last_active: {e}")
                db.session.rollback()
                
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'CineBrain token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid CineBrain token'}), 401
        except Exception as e:
            logger.error(f"CineBrain authentication error: {e}")
            return jsonify({'error': 'CineBrain authentication failed'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

def get_advanced_user_recommendations(user_id: int, recommendation_type: str = 'for_you', 
                                    limit: int = 20) -> Optional[Dict[str, Any]]:
    """Get recommendations using the advanced personalization system"""
    try:
        from personalized import get_personalization_system
        personalization_system = get_personalization_system()
        
        if not personalization_system or not personalization_system.is_ready():
            return None
        
        if hasattr(personalization_system, 'recommendation_engine'):
            advanced_engine = personalization_system.recommendation_engine
            result = advanced_engine.generate_personalized_recommendations(
                user_id=user_id,
                recommendation_type=recommendation_type,
                limit=limit
            )
            return result
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting advanced recommendations: {e}")
        return None

def get_user_personalization_insights(user_id: int) -> Dict[str, Any]:
    """Get detailed personalization insights for a user"""
    try:
        from personalized import get_personalization_system
        personalization_system = get_personalization_system()
        
        if not personalization_system or not personalization_system.is_ready():
            return {'available': False, 'message': 'Advanced personalization not available'}
        
        if hasattr(personalization_system, 'profile_analyzer'):
            profile_analyzer = personalization_system.profile_analyzer
            user_profile = profile_analyzer.build_comprehensive_user_profile(user_id)
            
            if user_profile:
                insights = {
                    'available': True,
                    'profile_version': user_profile.get('profile_version', '3.0'),
                    'personalization_strength': user_profile.get('personalization_strength', 0.0),
                    'telugu_cinema_affinity': user_profile.get('telugu_cinema_affinity', 0.0),
                    'recommendation_confidence': user_profile.get('recommendation_confidence', 0.0),
                    'cultural_profile': user_profile.get('cultural_profile', {}),
                    'cinematic_dna': {
                        'sophistication': user_profile.get('cinematic_dna', {}).get('cinematic_maturity_score', 0.0),
                        'discovery_openness': user_profile.get('cinematic_dna', {}).get('discovery_openness', 0.0),
                        'telugu_connection': user_profile.get('cinematic_dna', {}).get('telugu_cinema_connection', {})
                    },
                    'behavioral_intelligence': {
                        'engagement_level': user_profile.get('behavioral_intelligence', {}).get('engagement_intelligence', {}).get('engagement_level', 'unknown'),
                        'prediction_confidence': user_profile.get('behavioral_intelligence', {}).get('prediction_confidence', 0.0),
                        'behavioral_maturity': user_profile.get('behavioral_intelligence', {}).get('behavioral_maturity', 0.0)
                    },
                    'recommendations_ready': True,
                    'neural_embeddings': 'enabled' if user_profile.get('user_embedding') else 'disabled',
                    'last_analysis': user_profile.get('analysis_metadata', {}).get('generated_at')
                }
                
                return insights
        
        return {'available': False, 'message': 'Profile analyzer not ready'}
        
    except Exception as e:
        logger.error(f"Error getting personalization insights: {e}")
        return {'available': False, 'error': str(e)}

def trigger_personalization_refresh(user_id: int) -> bool:
    """Trigger a refresh of user's personalization profile"""
    try:
        # Clear relevant caches
        if cache:
            cache_keys = [
                f"cinebrain:advanced_profile:{user_id}",
                f"cinebrain:advanced_recs:{user_id}:*",
                f"cinebrain:behavioral_intelligence:{user_id}"
            ]
            
            for key in cache_keys:
                try:
                    cache.delete(key)
                except:
                    pass
        
        # Trigger background refresh if advanced system is available
        from personalized import get_personalization_system
        personalization_system = get_personalization_system()
        
        if personalization_system and hasattr(personalization_system, 'profile_analyzer'):
            # This would trigger a background refresh in a production system
            # For now, just log the refresh trigger
            logger.info(f"Triggered personalization refresh for user {user_id}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error triggering personalization refresh: {e}")
        return False

def get_enhanced_user_stats(user_id):
    """Get comprehensive user statistics with advanced personalization insights"""
    try:
        # Get basic stats first
        try:
            from services.auth import EnhancedUserAnalytics
            basic_stats = EnhancedUserAnalytics.get_comprehensive_user_stats(user_id)
        except ImportError:
            logger.warning("CineBrain enhanced analytics not available, using basic stats")
            basic_stats = get_basic_user_stats(user_id)
        
        # Add advanced personalization insights
        personalization_insights = get_user_personalization_insights(user_id)
        
        if personalization_insights.get('available', False):
            # Merge advanced insights with basic stats
            enhanced_stats = basic_stats.copy()
            enhanced_stats.update({
                'advanced_personalization': personalization_insights,
                'neural_features': {
                    'collaborative_filtering': 'enabled',
                    'cultural_awareness': 'active',
                    'adaptive_learning': 'real_time',
                    'telugu_prioritization': 'maximum'
                },
                'recommendation_quality': {
                    'personalization_strength': personalization_insights.get('personalization_strength', 0.0),
                    'confidence_level': personalization_insights.get('recommendation_confidence', 0.0),
                    'cultural_match': personalization_insights.get('telugu_cinema_affinity', 0.0),
                    'system_version': 'advanced_neural_cultural'
                },
                'cinematic_profile': personalization_insights.get('cinematic_dna', {}),
                'behavioral_profile': personalization_insights.get('behavioral_intelligence', {})
            })
            
            return enhanced_stats
        else:
            # Return basic stats with note about advanced features
            basic_stats['advanced_personalization'] = {
                'available': False,
                'message': 'Advanced personalization not available',
                'fallback': 'using_basic_stats'
            }
            return basic_stats
            
    except Exception as e:
        logger.error(f"Error getting enhanced user stats: {e}")
        return get_basic_user_stats(user_id)

def get_basic_user_stats(user_id):
    """Get basic user statistics"""
    try:
        if not UserInteraction:
            return {
                'total_interactions': 0,
                'content_watched': 0,
                'favorites': 0,
                'watchlist_items': 0,
                'ratings_given': 0
            }
        
        interactions = UserInteraction.query.filter_by(user_id=user_id).all()
        
        stats = {
            'total_interactions': len(interactions),
            'content_watched': len([i for i in interactions if i.interaction_type == 'view']),
            'favorites': len([i for i in interactions if i.interaction_type == 'favorite']),
            'watchlist_items': len([i for i in interactions if i.interaction_type == 'watchlist']),
            'ratings_given': len([i for i in interactions if i.interaction_type == 'rating']),
            'likes_given': len([i for i in interactions if i.interaction_type == 'like']),
            'searches_made': len([i for i in interactions if i.interaction_type == 'search'])
        }
        
        ratings = [i.rating for i in interactions if i.rating is not None]
        stats['average_rating'] = round(sum(ratings) / len(ratings), 1) if ratings else 0
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating CineBrain basic stats: {e}")
        return {}

def update_user_preferences_realtime(user_id: int, interaction_data: Dict[str, Any]) -> bool:
    """Update user preferences in real-time with advanced system integration"""
    try:
        # Clear relevant caches
        if cache:
            cache_keys = [
                f"cinebrain_recs:{user_id}:*",
                f"user_profile:{user_id}",
                f"cinebrain:advanced_profile:{user_id}",
                f"cinebrain:advanced_recs:{user_id}:*"
            ]
            for key in cache_keys:
                try:
                    cache.delete(key)
                except:
                    pass
        
        # Update with legacy system if available
        success = False
        if recommendation_engine and hasattr(recommendation_engine, 'update_user_preferences_realtime'):
            try:
                success = recommendation_engine.update_user_preferences_realtime(user_id, interaction_data)
            except Exception as e:
                logger.warning(f"Legacy system update failed: {e}")
        
        # Update with advanced system if available
        try:
            from personalized import get_personalization_system
            personalization_system = get_personalization_system()
            
            if (personalization_system and 
                hasattr(personalization_system, 'recommendation_engine') and
                hasattr(personalization_system.recommendation_engine, 'adaptive_engine')):
                
                adaptive_engine = personalization_system.recommendation_engine.adaptive_engine
                
                # Format feedback for adaptive engine
                feedback_record = {
                    'user_id': user_id,
                    'content_id': interaction_data.get('content_id'),
                    'feedback_type': interaction_data.get('interaction_type'),
                    'rating': interaction_data.get('rating'),
                    'metadata': interaction_data.get('metadata', {}),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Store in adaptive engine
                adaptive_engine.recommendation_feedback[user_id].append(feedback_record)
                
                logger.info(f"Updated advanced personalization system for user {user_id}")
                success = True
                
        except Exception as e:
            logger.warning(f"Advanced system update failed: {e}")
        
        # Record interaction in database
        try:
            interaction = UserInteraction(
                user_id=user_id,
                content_id=interaction_data.get('content_id'),
                interaction_type=interaction_data.get('interaction_type'),
                rating=interaction_data.get('rating'),
                interaction_metadata=json.dumps(interaction_data.get('metadata', {}))
            )
            
            db.session.add(interaction)
            db.session.commit()
            
            logger.info(f"Recorded interaction for user {user_id}: {interaction_data.get('interaction_type')}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording interaction: {e}")
            db.session.rollback()
            return success
            
    except Exception as e:
        logger.error(f"Error in real-time preference update: {e}")
        return False

def create_minimal_content_record(content_id, content_info):
    """Create minimal content record if content doesn't exist"""
    try:
        content_type = str(content_info.get('content_type', 'movie')).strip().lower()
        content_type = ' '.join(content_type.split())
        if content_type not in ['movie', 'tv', 'anime']:
            content_type = 'movie'
        
        title = str(content_info.get('title', 'Unknown Title')).strip()[:255]
        if not title:
            title = 'Unknown Title'
        
        timestamp = int(datetime.utcnow().timestamp())
        slug = f"content-{content_id}-{timestamp}"
        if len(slug) > 150:
            slug = slug[:150]
        
        overview = str(content_info.get('overview', '')).strip()[:1000]
        
        poster_path = content_info.get('poster_path')
        if poster_path and len(str(poster_path)) > 255:
            poster_path = str(poster_path)[:255]
        
        content_record = Content(
            id=content_id,
            title=title,
            content_type=content_type,
            poster_path=poster_path,
            rating=float(content_info.get('rating', 0)) if content_info.get('rating') else 0,
            overview=overview,
            release_date=None,
            tmdb_id=content_info.get('tmdb_id'),
            genres='[]',
            languages='[]',
            slug=slug
        )
        
        if content_info.get('release_date'):
            try:
                release_date_str = str(content_info['release_date'])[:10]
                content_record.release_date = datetime.strptime(
                    release_date_str, '%Y-%m-%d'
                ).date()
            except (ValueError, TypeError):
                logger.warning(f"Invalid release date format: {content_info.get('release_date')}")
        
        db.session.add(content_record)
        db.session.commit()
        logger.info(f"CineBrain: Created minimal content record for ID {content_id}")
        return content_record
        
    except Exception as e:
        logger.error(f"Failed to create minimal content record for ID {content_id}: {e}")
        db.session.rollback()
        return None

def format_content_for_response(content, interaction=None):
    """Format content object for API response"""
    youtube_url = None
    if content.youtube_trailer_id:
        youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
    
    formatted = {
        'id': content.id,
        'slug': content.slug,
        'title': content.title,
        'content_type': content.content_type,
        'genres': json.loads(content.genres or '[]'),
        'rating': content.rating,
        'release_date': content.release_date.isoformat() if content.release_date else None,
        'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
        'youtube_trailer': youtube_url,
        'is_new_release': content.is_new_release,
        'is_trending': content.is_trending
    }
    
    if interaction:
        formatted['added_at'] = interaction.timestamp.isoformat()
        if hasattr(interaction, 'rating') and interaction.rating:
            formatted['user_rating'] = interaction.rating
    
    return formatted

def add_cors_headers(response):
    """Add CORS headers to response"""
    origin = request.headers.get('Origin')
    allowed_origins = [
        'https://cinebrain.vercel.app',
        'http://127.0.0.1:5500',
        'http://127.0.0.1:5501',
        'http://localhost:3000',
        'http://localhost:5173'
    ]
    
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response