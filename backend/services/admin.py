from flask import Blueprint, request, jsonify
from flask_caching import Cache
from datetime import datetime, timedelta
import json
import logging
import time
import telebot
from sqlalchemy import func, desc
from collections import defaultdict
from functools import wraps
import jwt
import os

admin_bp = Blueprint('admin', __name__)

logger = logging.getLogger(__name__)

cache = Cache()

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '7974343726:AAFUCW444L6jbj1tVLRyf8V7Isz2Ua1SxSk')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '-1002850793757')

if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != 'your_telegram_bot_token':
    try:
        bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
    except:
        bot = None
        logging.warning("Failed to initialize Telegram bot")
else:
    bot = None

db = None
User = None
Content = None
UserInteraction = None
AdminRecommendation = None
TMDBService = None
JikanService = None
ContentService = None
http_session = None
cache = None
app = None

def init_admin(flask_app, database, models, services):
    global db, User, Content, UserInteraction, AdminRecommendation
    global TMDBService, JikanService, ContentService
    global http_session, cache, app
    
    app = flask_app
    db = database
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    AdminRecommendation = models['AdminRecommendation']
    
    TMDBService = services['TMDBService']
    JikanService = services['JikanService']
    ContentService = services['ContentService']
    http_session = services['http_session']
    cache = services['cache']

def require_admin(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user or not current_user.is_admin:
                return jsonify({'error': 'Admin access required'}), 403
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

class TelegramService:
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram bot or channel ID not configured")
                return False
            
            genres_list = []
            if content.genres:
                try:
                    genres_list = json.loads(content.genres)
                except:
                    genres_list = []
            
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

#AdminChoice #MovieRecommendation #CineScope"""
            
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='Markdown'
                    )
                except Exception as photo_error:
                    logger.error(f"Failed to send photo, sending text only: {photo_error}")
                    bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
            else:
                bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
            
            return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

@admin_bp.route('/api/admin/search', methods=['GET'])
@require_admin
def admin_search(current_user):
    try:
        query = request.args.get('query', '')
        source = request.args.get('source', 'tmdb')
        page = int(request.args.get('page', 1))
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        results = []
        
        if source == 'tmdb':
            tmdb_results = TMDBService.search_content(query, page=page)
            if tmdb_results:
                for item in tmdb_results.get('results', []):
                    results.append({
                        'id': item['id'],
                        'title': item.get('title') or item.get('name'),
                        'content_type': 'movie' if 'title' in item else 'tv',
                        'release_date': item.get('release_date') or item.get('first_air_date'),
                        'poster_path': f"https://image.tmdb.org/t/p/w300{item['poster_path']}" if item.get('poster_path') else None,
                        'overview': item.get('overview'),
                        'rating': item.get('vote_average'),
                        'source': 'tmdb'
                    })
        
        elif source == 'anime':
            anime_results = JikanService.search_anime(query, page=page)
            if anime_results:
                for anime in anime_results.get('data', []):
                    results.append({
                        'id': anime['mal_id'],
                        'title': anime.get('title'),
                        'content_type': 'anime',
                        'release_date': anime.get('aired', {}).get('from'),
                        'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                        'overview': anime.get('synopsis'),
                        'rating': anime.get('score'),
                        'source': 'anime'
                    })
        
        return jsonify({'results': results}), 200
        
    except Exception as e:
        logger.error(f"Admin search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

@admin_bp.route('/api/admin/content', methods=['POST'])
@require_admin
def save_external_content(current_user):
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No content data provided'}), 400
        
        existing_content = None
        if data.get('source') == 'anime' and data.get('id'):
            existing_content = Content.query.filter_by(mal_id=data['id']).first()
        elif data.get('id'):
            existing_content = Content.query.filter_by(tmdb_id=data['id']).first()
        
        if existing_content:
            return jsonify({
                'message': 'Content already exists',
                'content_id': existing_content.id
            }), 200
        
        try:
            release_date = None
            if data.get('release_date'):
                try:
                    release_date = datetime.strptime(data['release_date'][:10], '%Y-%m-%d').date()
                except:
                    release_date = None
            
            youtube_trailer_id = ContentService.get_youtube_trailer(data.get('title'), data.get('content_type')) if ContentService else None
            
            if data.get('source') == 'anime':
                content = Content(
                    mal_id=data.get('id'),
                    title=data.get('title'),
                    original_title=data.get('original_title'),
                    content_type='anime',
                    genres=json.dumps(data.get('genres', [])),
                    anime_genres=json.dumps([]),
                    languages=json.dumps(['japanese']),
                    release_date=release_date,
                    rating=data.get('rating'),
                    overview=data.get('overview'),
                    poster_path=data.get('poster_path'),
                    youtube_trailer_id=youtube_trailer_id
                )
            else:
                content = Content(
                    tmdb_id=data.get('id'),
                    title=data.get('title'),
                    original_title=data.get('original_title'),
                    content_type=data.get('content_type', 'movie'),
                    genres=json.dumps(data.get('genres', [])),
                    languages=json.dumps(data.get('languages', ['en'])),
                    release_date=release_date,
                    runtime=data.get('runtime'),
                    rating=data.get('rating'),
                    vote_count=data.get('vote_count'),
                    popularity=data.get('popularity'),
                    overview=data.get('overview'),
                    poster_path=data.get('poster_path'),
                    backdrop_path=data.get('backdrop_path'),
                    youtube_trailer_id=youtube_trailer_id
                )
            
            content.ensure_slug()
            
            db.session.add(content)
            db.session.commit()
            
            return jsonify({
                'message': 'Content saved successfully',
                'content_id': content.id
            }), 201
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving content: {e}")
            return jsonify({'error': 'Failed to save content to database'}), 500
        
    except Exception as e:
        logger.error(f"Save content error: {e}")
        return jsonify({'error': 'Failed to process content'}), 500

@admin_bp.route('/api/admin/recommendations', methods=['POST'])
@require_admin
def create_admin_recommendation(current_user):
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'recommendation_type', 'description']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        content = Content.query.get(data['content_id'])
        if not content:
            content = Content.query.filter_by(tmdb_id=data['content_id']).first()
        
        if not content:
            return jsonify({'error': 'Content not found. Please save content first.'}), 404
        
        admin_rec = AdminRecommendation(
            content_id=content.id,
            admin_id=current_user.id,
            recommendation_type=data['recommendation_type'],
            description=data['description']
        )
        
        db.session.add(admin_rec)
        db.session.commit()
        
        telegram_success = TelegramService.send_admin_recommendation(content, current_user.username, data['description'])
        
        return jsonify({
            'message': 'Admin recommendation created successfully',
            'telegram_sent': telegram_success
        }), 201
        
    except Exception as e:
        logger.error(f"Admin recommendation error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create recommendation'}), 500

@admin_bp.route('/api/admin/recommendations', methods=['GET'])
@require_admin
def get_admin_recommendations(current_user):
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        admin_recs = AdminRecommendation.query.filter_by(is_active=True)\
            .order_by(AdminRecommendation.created_at.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        result = []
        for rec in admin_recs.items:
            content = Content.query.get(rec.content_id)
            admin = User.query.get(rec.admin_id)
            
            result.append({
                'id': rec.id,
                'recommendation_type': rec.recommendation_type,
                'description': rec.description,
                'created_at': rec.created_at.isoformat(),
                'admin_name': admin.username if admin else 'Unknown',
                'content': {
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path
                }
            })
        
        return jsonify({
            'recommendations': result,
            'total': admin_recs.total,
            'pages': admin_recs.pages,
            'current_page': page
        }), 200
        
    except Exception as e:
        logger.error(f"Get admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@admin_bp.route('/api/admin/analytics', methods=['GET'])
@require_admin
def get_analytics(current_user):
    try:
        total_users = User.query.count()
        total_content = Content.query.count()
        total_interactions = UserInteraction.query.count()
        active_users_last_week = User.query.filter(
            User.last_active >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        popular_content = db.session.query(
            Content.id, Content.title, func.count(UserInteraction.id).label('interaction_count')
        ).join(UserInteraction).group_by(Content.id, Content.title)\
         .order_by(desc('interaction_count')).limit(10).all()
        
        all_interactions = UserInteraction.query.join(Content).all()
        genre_counts = defaultdict(int)
        for interaction in all_interactions:
            content = Content.query.get(interaction.content_id)
            if content and content.genres:
                try:
                    genres = json.loads(content.genres)
                    for genre in genres:
                        genre_counts[genre] += 1
                except:
                    pass
        
        popular_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        user_interactions_by_type = db.session.query(
            UserInteraction.interaction_type,
            func.count(UserInteraction.id).label('count')
        ).group_by(UserInteraction.interaction_type).all()
        
        content_by_type = db.session.query(
            Content.content_type,
            func.count(Content.id).label('count')
        ).group_by(Content.content_type).all()
        
        recent_users = User.query.filter(
            User.created_at >= datetime.utcnow() - timedelta(days=30)
        ).count()
        
        return jsonify({
            'total_users': total_users,
            'total_content': total_content,
            'total_interactions': total_interactions,
            'active_users_last_week': active_users_last_week,
            'new_users_last_month': recent_users,
            'popular_content': [
                {'title': item.title, 'interactions': item.interaction_count}
                for item in popular_content
            ],
            'popular_genres': [
                {'genre': genre, 'count': count}
                for genre, count in popular_genres
            ],
            'interactions_by_type': [
                {'type': item.interaction_type, 'count': item.count}
                for item in user_interactions_by_type
            ],
            'content_by_type': [
                {'type': item.content_type, 'count': item.count}
                for item in content_by_type
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500

@admin_bp.route('/api/admin/users', methods=['GET'])
@require_admin
def get_users_management(current_user):
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        search = request.args.get('search', '')
        
        query = User.query
        
        if search:
            query = query.filter(
                User.username.contains(search) | 
                User.email.contains(search)
            )
        
        users = query.order_by(User.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        result = []
        for user in users.items:
            user_interactions = UserInteraction.query.filter_by(user_id=user.id).count()
            
            result.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'created_at': user.created_at.isoformat(),
                'last_active': user.last_active.isoformat() if user.last_active else None,
                'total_interactions': user_interactions,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]')
            })
        
        return jsonify({
            'users': result,
            'total': users.total,
            'pages': users.pages,
            'current_page': page
        }), 200
        
    except Exception as e:
        logger.error(f"Users management error: {e}")
        return jsonify({'error': 'Failed to get users'}), 500

@admin_bp.route('/api/admin/content/manage', methods=['GET'])
@require_admin
def get_content_management(current_user):
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        content_type = request.args.get('type', 'all')
        search = request.args.get('search', '')
        
        query = Content.query
        
        if content_type != 'all':
            query = query.filter_by(content_type=content_type)
        
        if search:
            query = query.filter(Content.title.contains(search))
        
        contents = query.order_by(Content.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        result = []
        for content in contents.items:
            interaction_count = UserInteraction.query.filter_by(content_id=content.id).count()
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'created_at': content.created_at.isoformat(),
                'interaction_count': interaction_count,
                'genres': json.loads(content.genres or '[]'),
                'tmdb_id': content.tmdb_id,
                'mal_id': content.mal_id,
                'poster_path': content.poster_path
            })
        
        return jsonify({
            'content': result,
            'total': contents.total,
            'pages': contents.pages,
            'current_page': page
        }), 200
        
    except Exception as e:
        logger.error(f"Content management error: {e}")
        return jsonify({'error': 'Failed to get content'}), 500

@admin_bp.route('/api/admin/cache/clear', methods=['POST'])
@require_admin
def clear_cache(current_user):
    try:
        cache_type = request.args.get('type', 'all')
        
        if cache_type == 'all':
            cache.clear()
            message = 'All cache cleared'
        elif cache_type == 'search':
            cache.delete_memoized(TMDBService.search_content)
            cache.delete_memoized(JikanService.search_anime)
            message = 'Search cache cleared'
        elif cache_type == 'recommendations':
            message = 'Recommendations cache cleared'
        else:
            return jsonify({'error': 'Invalid cache type'}), 400
        
        return jsonify({
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        return jsonify({'error': 'Failed to clear cache'}), 500

@admin_bp.route('/api/admin/cache/stats', methods=['GET'])
@require_admin
def get_cache_stats(current_user):
    try:
        cache_info = {
            'type': app.config.get('CACHE_TYPE', 'unknown'),
            'default_timeout': app.config.get('CACHE_DEFAULT_TIMEOUT', 0),
        }
        
        if app.config.get('CACHE_TYPE') == 'redis':
            try:
                import redis
                REDIS_URL = app.config.get('CACHE_REDIS_URL')
                if REDIS_URL:
                    r = redis.from_url(REDIS_URL)
                    redis_info = r.info()
                    cache_info['redis'] = {
                        'used_memory': redis_info.get('used_memory_human', 'N/A'),
                        'connected_clients': redis_info.get('connected_clients', 0),
                        'total_commands_processed': redis_info.get('total_commands_processed', 0),
                        'uptime_in_seconds': redis_info.get('uptime_in_seconds', 0)
                    }
            except:
                cache_info['redis'] = {'status': 'Unable to connect'}
        
        return jsonify(cache_info), 200
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return jsonify({'error': 'Failed to get cache stats'}), 500

@admin_bp.route('/api/admin/system-health', methods=['GET'])
@require_admin
def get_system_health(current_user):
    try:
        health_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'healthy',
            'components': {}
        }
        
        health_data['components']['database'] = {
            'status': 'healthy',
            'total_users': User.query.count(),
            'total_content': Content.query.count(),
            'total_interactions': UserInteraction.query.count()
        }
        
        try:
            cache.set('health_check', 'ok', timeout=10)
            if cache.get('health_check') == 'ok':
                health_data['components']['cache'] = {'status': 'healthy'}
            else:
                health_data['components']['cache'] = {'status': 'degraded'}
                health_data['status'] = 'degraded'
        except:
            health_data['components']['cache'] = {'status': 'unhealthy'}
            health_data['status'] = 'degraded'
        
        health_data['components']['external_apis'] = {
            'tmdb': 'configured' if TMDBService else 'not_configured',
            'jikan': 'configured' if JikanService else 'not_configured',
            'telegram': 'configured' if bot else 'not_configured'
        }
        
        return jsonify(health_data), 200
        
    except Exception as e:
        logger.error(f"System health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500