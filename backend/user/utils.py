# backend/user/utils.py

from flask import request, jsonify, current_app
from datetime import datetime, timedelta
import json
import logging
import jwt
from functools import wraps
from collections import defaultdict, Counter
import numpy as np
import hashlib
import time

logger = logging.getLogger(__name__)

db = None
User = None
Content = None
UserInteraction = None
Review = None
UserSettings = None
UserDevice = None
UserDeviceActivity = None
app = None
recommendation_engine = None
profile_analyzer = None
cache = None
content_service = None

def init_user_module(flask_app, database, models, services):
    global db, User, Content, UserInteraction, Review, UserSettings, UserDevice, UserDeviceActivity, app, recommendation_engine, cache, content_service, profile_analyzer
    
    app = flask_app
    db = database
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    Review = models.get('Review')
    UserSettings = models.get('UserSettings')
    UserDevice = models.get('UserDevice')
    UserDeviceActivity = models.get('UserDeviceActivity')
    cache = services.get('cache')
    content_service = services.get('ContentService')
    
    try:
        # NEW: Get from the personalized module directly
        from personalized import get_profile_analyzer, get_embedding_manager
        from personalized.recommendation_engine import HybridRecommendationEngine
        
        # Check if already initialized in services
        if 'profile_analyzer' in services:
            profile_analyzer = services['profile_analyzer']
            logger.info("✅ Using provided profile analyzer")
        else:
            profile_analyzer = get_profile_analyzer()
            logger.info("✅ CineBrain profile analyzer connected to user module")
            
        if 'recommendation_engine' in services:
            recommendation_engine = services['recommendation_engine']
            logger.info("✅ Using provided recommendation engine")
        else:
            # Initialize a new one if needed
            recommendation_engine = HybridRecommendationEngine(db=db, models=models, cache_manager=cache)
            logger.info("✅ Created new recommendation engine for user module")
            
    except Exception as e:
        logger.warning(f"Could not connect to CineBrain personalization engine: {e}")
        recommendation_engine = None
        profile_analyzer = None

def require_auth(f):
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
                track_user_activity(current_user.id, request)
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
    try:
        if profile_analyzer:
            profile = profile_analyzer.build_user_profile(user_id)
            return {
                'total_interactions': profile.get('implicit_preferences', {}).get('total_interactions', 0),
                'content_watched': len([i for i in get_user_interactions(user_id) if i.interaction_type == 'view']),
                'favorites': len([i for i in get_user_interactions(user_id) if i.interaction_type == 'favorite']),
                'watchlist_items': len([i for i in get_user_interactions(user_id) if i.interaction_type == 'watchlist']),
                'ratings_given': len([i for i in get_user_interactions(user_id) if i.interaction_type == 'rating']),
                'engagement_metrics': profile.get('engagement_metrics', {}),
                'content_diversity': profile.get('diversity_score', 0),
                'profile_confidence': profile.get('confidence_score', 0),
                'user_segment': profile.get('user_segment', 'new_user'),
                'cinematic_dna': profile.get('cinematic_dna', {}),
                'behavioral_patterns': profile.get('behavioral_patterns', {}),
                'quality_preferences': profile.get('recommendation_context', {})
            }
        else:
            return get_basic_user_stats(user_id)
    except Exception as e:
        logger.error(f"Error getting CineBrain user stats: {e}")
        return get_basic_user_stats(user_id)

def get_basic_user_stats(user_id):
    try:
        if not UserInteraction:
            return {'total_interactions': 0, 'content_watched': 0, 'favorites': 0, 'watchlist_items': 0, 'ratings_given': 0}
        
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

def get_user_interactions(user_id, limit=None):
    if not UserInteraction:
        return []
    
    query = UserInteraction.query.filter_by(user_id=user_id).order_by(UserInteraction.timestamp.desc())
    if limit:
        query = query.limit(limit)
    
    return query.all()

def create_minimal_content_record(content_id, content_info):
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
                content_record.release_date = datetime.strptime(release_date_str, '%Y-%m-%d').date()
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

def track_user_activity(user_id, request_obj):
    try:
        if not UserDeviceActivity:
            return
        
        device_info = extract_device_info(request_obj)
        
        activity = UserDeviceActivity(
            user_id=user_id,
            device_type=device_info['device_type'],
            browser=device_info['browser'],
            os=device_info['os'],
            ip_address=request_obj.remote_addr,
            timestamp=datetime.utcnow()
        )
        
        db.session.add(activity)
        db.session.commit()
        
    except Exception as e:
        logger.warning(f"Failed to track user activity: {e}")
        db.session.rollback()

def extract_device_info(request_obj):
    user_agent = request_obj.headers.get('User-Agent', '').lower()
    
    device_type = 'desktop'
    if any(mobile in user_agent for mobile in ['mobile', 'android', 'iphone']):
        device_type = 'mobile'
    elif any(tablet in user_agent for tablet in ['tablet', 'ipad']):
        device_type = 'tablet'
    
    browser = 'unknown'
    if 'chrome' in user_agent:
        browser = 'chrome'
    elif 'firefox' in user_agent:
        browser = 'firefox'
    elif 'safari' in user_agent:
        browser = 'safari'
    elif 'edge' in user_agent:
        browser = 'edge'
    
    os = 'unknown'
    if 'windows' in user_agent:
        os = 'windows'
    elif 'mac' in user_agent:
        os = 'macos'
    elif 'linux' in user_agent:
        os = 'linux'
    elif 'android' in user_agent:
        os = 'android'
    elif 'ios' in user_agent:
        os = 'ios'
    
    return {
        'device_type': device_type,
        'browser': browser,
        'os': os
    }

def get_cache_key(prefix, *args):
    key_parts = [str(arg) for arg in args if arg is not None]
    return f"cinebrain:{prefix}:{':'.join(key_parts)}"

def cache_get(key, default=None):
    if cache:
        try:
            return cache.get(key) or default
        except:
            return default
    return default

def cache_set(key, value, timeout=300):
    if cache:
        try:
            cache.set(key, value, timeout=timeout)
        except:
            pass

def cache_delete(key):
    if cache:
        try:
            cache.delete(key)
        except:
            pass

def get_personalized_recommendations(user_id, limit=20, categories=None):
    try:
        if recommendation_engine:
            return recommendation_engine.generate_recommendations(
                user_id=user_id,
                limit=limit,
                context={'categories': categories}
            )
        return {'recommendations': [], 'metadata': {}}
    except Exception as e:
        logger.error(f"Error getting personalized recommendations: {e}")
        return {'recommendations': [], 'metadata': {}}

def get_user_profile_insights(user_id):
    try:
        if profile_analyzer:
            return profile_analyzer.build_user_profile(user_id)
        return {}
    except Exception as e:
        logger.error(f"Error getting user profile insights: {e}")
        return {}

def calculate_watch_time(user_id):
    try:
        interactions = get_user_interactions(user_id)
        total_minutes = 0
        
        for interaction in interactions:
            if interaction.interaction_type in ['view', 'favorite']:
                content = Content.query.get(interaction.content_id)
                if content and content.runtime:
                    total_minutes += content.runtime
        
        return {
            'total_minutes': total_minutes,
            'total_hours': round(total_minutes / 60, 1),
            'total_days': round(total_minutes / (60 * 24), 2)
        }
    except Exception as e:
        logger.error(f"Error calculating watch time: {e}")
        return {'total_minutes': 0, 'total_hours': 0, 'total_days': 0}

def get_content_by_ids(content_ids):
    try:
        if not content_ids:
            return []
        return Content.query.filter(Content.id.in_(content_ids)).all()
    except Exception as e:
        logger.error(f"Error getting content by IDs: {e}")
        return []