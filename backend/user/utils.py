# user/utils.py
from flask import request, jsonify
from datetime import datetime, timedelta
import json
import logging
import jwt
from functools import wraps
from collections import defaultdict, Counter
import numpy as np

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
    
    try:
        from services.personalized import get_recommendation_engine
        recommendation_engine = get_recommendation_engine()
        logger.info("✅ CineBrain personalized recommendation engine connected to user module")
    except Exception as e:
        logger.warning(f"Could not connect to CineBrain recommendation engine: {e}")

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

def get_enhanced_user_stats(user_id):
    """Get comprehensive user statistics"""
    try:
        try:
            from services.auth import EnhancedUserAnalytics
            return EnhancedUserAnalytics.get_comprehensive_user_stats(user_id)
        except ImportError:
            logger.warning("CineBrain enhanced analytics not available, using basic stats")
            return get_basic_user_stats(user_id)
    except Exception as e:
        logger.error(f"Error getting CineBrain user stats: {e}")
        return {}

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