#backend/admin.py
from flask import Blueprint, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from functools import wraps
import logging
import json
import time
import jwt
import telebot
from sqlalchemy import func, desc
from collections import defaultdict
import requests
from concurrent.futures import ThreadPoolExecutor

# Create admin blueprint
admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')

# Initialize logger
logger = logging.getLogger(__name__)

# Global variables (will be initialized from main app)
db = None
cache = None
app_instance = None
bot = None
TELEGRAM_BOT_TOKEN = None
TELEGRAM_CHANNEL_ID = None
ML_SERVICE_URL = None
http_session = None
trending_service = None

def init_admin(app, database, cache_instance):
    """Initialize admin module with app instance"""
    global db, cache, app_instance, bot, TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID
    global ML_SERVICE_URL, http_session, trending_service
    
    db = database
    cache = cache_instance
    app_instance = app
    
    # Get configuration from app
    TELEGRAM_BOT_TOKEN = app.config.get('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHANNEL_ID = app.config.get('TELEGRAM_CHANNEL_ID')
    ML_SERVICE_URL = app.config.get('ML_SERVICE_URL')
    
    # Initialize Telegram bot
    if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != 'your_telegram_bot_token':
        try:
            bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
            logger.info("Telegram bot initialized successfully")
        except Exception as e:
            bot = None
            logger.warning(f"Failed to initialize Telegram bot: {e}")
    else:
        bot = None
        logger.warning("Telegram bot token not configured")
    
    # Create HTTP session
    http_session = create_http_session()
    
    logger.info("Admin module initialized")

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

def require_admin(f):
    """Decorator to require admin authentication"""
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
            
            if not current_user or not current_user.is_admin:
                return jsonify({'error': 'Admin access required'}), 403
        except Exception as e:
            logger.error(f"Admin auth error: {e}")
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

class TelegramService:
    """Service for Telegram notifications and channel management"""
    
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        """Send admin recommendation to Telegram channel"""
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
                    logger.info(f"Sent Telegram photo message for {content.title}")
                except Exception as photo_error:
                    logger.error(f"Failed to send photo, sending text only: {photo_error}")
                    bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
            else:
                bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
                logger.info(f"Sent Telegram text message for {content.title}")
            
            return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    @staticmethod
    def send_trending_update(trending_data, language="English"):
        """Send trending update to Telegram channel"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = f"""📊 **Trending Update - {language}**

🔥 **Top Trending Movies:**
"""
            for i, movie in enumerate(trending_data[:5], 1):
                message += f"{i}. {movie.get('title', 'Unknown')} ⭐ {movie.get('rating', 'N/A')}\n"
            
            message += "\n#Trending #DailyUpdate #CineScope"
            
            bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
            return True
        except Exception as e:
            logger.error(f"Telegram trending update error: {e}")
            return False
    
    @staticmethod
    def send_analytics_report(analytics_data):
        """Send analytics report to Telegram channel"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = f"""📈 **Weekly Analytics Report**

👥 Total Users: {analytics_data.get('total_users', 0)}
🎬 Total Content: {analytics_data.get('total_content', 0)}
🔄 Total Interactions: {analytics_data.get('total_interactions', 0)}
✅ Active Users (Last Week): {analytics_data.get('active_users_last_week', 0)}

🏆 **Top Content:**
"""
            for i, content in enumerate(analytics_data.get('popular_content', [])[:5], 1):
                message += f"{i}. {content['title']} - {content['interactions']} interactions\n"
            
            message += "\n#Analytics #WeeklyReport #CineScope"
            
            bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
            return True
        except Exception as e:
            logger.error(f"Telegram analytics report error: {e}")
            return False

# Admin Routes

@admin_bp.route('/search', methods=['GET'])
@require_admin
def admin_search(current_user):
    """Admin search for content from various sources"""
    try:
        from app import TMDBService, JikanService
        
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

@admin_bp.route('/content', methods=['POST'])
@require_admin
def save_external_content(current_user):
    """Save content from external sources"""
    try:
        from app import Content, ContentService
        
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
            
            youtube_trailer_id = ContentService.get_youtube_trailer(data.get('title'), data.get('content_type'))
            
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

@admin_bp.route('/recommendations', methods=['POST'])
@require_admin
def create_admin_recommendation(current_user):
    """Create admin recommendation and send to Telegram"""
    try:
        from app import Content, AdminRecommendation
        
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
        
        # Send to Telegram
        telegram_success = TelegramService.send_admin_recommendation(
            content, current_user.username, data['description']
        )
        
        # Clear cache
        cache.delete_memoized('get_public_admin_recommendations')
        
        return jsonify({
            'message': 'Admin recommendation created successfully',
            'telegram_sent': telegram_success
        }), 201
        
    except Exception as e:
        logger.error(f"Admin recommendation error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create recommendation'}), 500

@admin_bp.route('/recommendations', methods=['GET'])
@require_admin
def get_admin_recommendations(current_user):
    """Get all admin recommendations"""
    try:
        from app import Content, AdminRecommendation, User
        
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

@admin_bp.route('/analytics', methods=['GET'])
@require_admin
def get_analytics(current_user):
    """Get analytics data and optionally send to Telegram"""
    try:
        from app import User, Content, UserInteraction
        
        send_telegram = request.args.get('send_telegram', 'false').lower() == 'true'
        
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
                genres = json.loads(content.genres)
                for genre in genres:
                    genre_counts[genre] += 1
        
        popular_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        analytics_data = {
            'total_users': total_users,
            'total_content': total_content,
            'total_interactions': total_interactions,
            'active_users_last_week': active_users_last_week,
            'popular_content': [
                {'title': item.title, 'interactions': item.interaction_count}
                for item in popular_content
            ],
            'popular_genres': [
                {'genre': genre, 'count': count}
                for genre, count in popular_genres
            ]
        }
        
        # Send to Telegram if requested
        if send_telegram:
            TelegramService.send_analytics_report(analytics_data)
        
        return jsonify(analytics_data), 200
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500

@admin_bp.route('/ml-service-check', methods=['GET'])
@require_admin
def ml_service_comprehensive_check(current_user):
    """Check ML service health and status"""
    try:
        from app import User, Content, UserInteraction
        
        ml_url = ML_SERVICE_URL
        if not ml_url:
            return jsonify({
                'status': 'error',
                'message': 'ML_SERVICE_URL not configured',
                'checks': {}
            }), 500

        checks = {}
        overall_status = 'healthy'
        
        # Connectivity check
        try:
            start_time = time.time()
            health_resp = http_session.get(f"{ml_url}/api/health", timeout=10)
            health_time = time.time() - start_time
            
            if health_resp.status_code == 200:
                health_data = health_resp.json()
                checks['connectivity'] = {
                    'status': 'pass',
                    'response_time': f"{health_time:.2f}s",
                    'models_initialized': health_data.get('models_initialized', False),
                    'data_status': health_data.get('data_status', {})
                }
            else:
                checks['connectivity'] = {'status': 'fail', 'error': f'HTTP {health_resp.status_code}'}
                overall_status = 'unhealthy'
        except Exception as e:
            checks['connectivity'] = {'status': 'fail', 'error': str(e)}
            overall_status = 'unhealthy'

        # Recommendations check
        if checks['connectivity']['status'] == 'pass':
            try:
                start_time = time.time()
                test_request = {
                    'user_id': 1,
                    'preferred_languages': ['english'],
                    'preferred_genres': ['Action'],
                    'interactions': [{
                        'content_id': 1,
                        'interaction_type': 'view',
                        'timestamp': datetime.utcnow().isoformat()
                    }]
                }
                
                rec_resp = http_session.post(f"{ml_url}/api/recommendations", json=test_request, timeout=20)
                rec_time = time.time() - start_time
                
                if rec_resp.status_code == 200:
                    rec_data = rec_resp.json()
                    checks['recommendations'] = {
                        'status': 'pass',
                        'response_time': f"{rec_time:.2f}s",
                        'count': len(rec_data.get('recommendations', [])),
                        'strategy': rec_data.get('strategy', 'unknown'),
                        'cached': rec_data.get('cached', False)
                    }
                else:
                    checks['recommendations'] = {'status': 'fail', 'error': f'HTTP {rec_resp.status_code}'}
                    overall_status = 'partial'
            except Exception as e:
                checks['recommendations'] = {'status': 'fail', 'error': str(e)}
                overall_status = 'partial'

        # Statistics check
        if checks['connectivity']['status'] == 'pass':
            try:
                start_time = time.time()
                stats_resp = http_session.get(f"{ml_url}/api/stats", timeout=10)
                stats_time = time.time() - start_time
                
                if stats_resp.status_code == 200:
                    stats_data = stats_resp.json()
                    checks['statistics'] = {
                        'status': 'pass',
                        'response_time': f"{stats_time:.2f}s",
                        'data_count': stats_data.get('data_statistics', {}).get('total_content', 0),
                        'user_count': stats_data.get('data_statistics', {}).get('unique_users', 0)
                    }
                else:
                    checks['statistics'] = {'status': 'fail', 'error': f'HTTP {stats_resp.status_code}'}
            except Exception as e:
                checks['statistics'] = {'status': 'fail', 'error': str(e)}

        # Database integration check
        try:
            total_users = User.query.count()
            total_content = Content.query.count() 
            total_interactions = UserInteraction.query.count()
            
            checks['database_integration'] = {
                'status': 'pass',
                'backend_users': total_users,
                'backend_content': total_content,
                'backend_interactions': total_interactions,
                'data_ready': total_content > 0 and total_interactions > 0
            }
        except Exception as e:
            checks['database_integration'] = {'status': 'fail', 'error': str(e)}

        # Calculate overall status
        failed_checks = sum(1 for check in checks.values() 
                           if isinstance(check, dict) and check.get('status') == 'fail')
        total_checks = len([check for check in checks.values() if isinstance(check, dict)])
        
        if failed_checks == 0:
            overall_status = 'healthy'
        elif failed_checks < total_checks:
            overall_status = 'partial'
        else:
            overall_status = 'unhealthy'

        return jsonify({
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'ml_service_url': ml_url,
            'summary': {
                'total_checks': total_checks,
                'passed': total_checks - failed_checks,
                'failed': failed_checks
            },
            'checks': checks,
            'quick_actions': {
                'force_update_available': checks.get('connectivity', {}).get('status') == 'pass',
                'recommendations_working': checks.get('recommendations', {}).get('status') == 'pass'
            }
        }), 200

    except Exception as e:
        logger.error(f"ML service check error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@admin_bp.route('/ml-service-update', methods=['POST'])
@require_admin  
def ml_service_force_update(current_user):
    """Force update ML service models"""
    try:
        if not ML_SERVICE_URL:
            return jsonify({'success': False, 'message': 'ML service not configured'}), 400
            
        response = http_session.post(f"{ML_SERVICE_URL}/api/update-models", timeout=30)
        
        if response.status_code == 200:
            return jsonify({'success': True, 'message': 'Model update initiated'})
        else:
            return jsonify({'success': False, 'message': f'Update failed: {response.status_code}'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@admin_bp.route('/ml-stats', methods=['GET'])
@require_admin
def get_ml_service_stats(current_user):
    """Get ML service statistics"""
    try:
        from app import User, Content, UserInteraction, MLServiceClient
        
        if not ML_SERVICE_URL:
            return jsonify({'error': 'ML service not configured'}), 400
        
        ml_stats = MLServiceClient.call_ml_service('/api/stats')
        
        if ml_stats:
            backend_stats = {
                'total_users': User.query.count(),
                'total_content': Content.query.count(),
                'total_interactions': UserInteraction.query.count(),
                'active_users_last_week': User.query.filter(
                    User.last_active >= datetime.utcnow() - timedelta(days=7)
                ).count()
            }
            
            return jsonify({
                'ml_service_stats': ml_stats,
                'backend_stats': backend_stats,
                'data_sync_status': {
                    'content_match': backend_stats['total_content'] == ml_stats.get('data_statistics', {}).get('total_content', 0),
                    'user_match': backend_stats['total_users'] == ml_stats.get('data_statistics', {}).get('unique_users', 0)
                }
            }), 200
        else:
            return jsonify({'error': 'Failed to get ML service stats'}), 500
            
    except Exception as e:
        logger.error(f"ML stats error: {e}")
        return jsonify({'error': 'Failed to get ML statistics'}), 500

@admin_bp.route('/cache/clear', methods=['POST'])
@require_admin
def clear_cache(current_user):
    """Clear application cache"""
    try:
        from app import TMDBService, JikanService, RecommendationEngine
        
        cache_type = request.args.get('type', 'all')
        
        if cache_type == 'all':
            cache.clear()
            message = 'All cache cleared'
        elif cache_type == 'search':
            cache.delete_memoized(TMDBService.search_content)
            cache.delete_memoized(JikanService.search_anime)
            message = 'Search cache cleared'
        elif cache_type == 'recommendations':
            cache.delete_memoized(RecommendationEngine.get_trending_recommendations)
            cache.delete_memoized(RecommendationEngine.get_new_releases)
            cache.delete_memoized(RecommendationEngine.get_critics_choice)
            cache.delete_memoized(RecommendationEngine.get_genre_recommendations)
            cache.delete_memoized(RecommendationEngine.get_regional_recommendations)
            cache.delete_memoized(RecommendationEngine.get_anime_recommendations)
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

@admin_bp.route('/cache/stats', methods=['GET'])
@require_admin
def get_cache_stats(current_user):
    """Get cache statistics"""
    try:
        cache_info = {
            'type': app_instance.config.get('CACHE_TYPE', 'unknown'),
            'default_timeout': app_instance.config.get('CACHE_DEFAULT_TIMEOUT', 0),
        }
        
        REDIS_URL = app_instance.config.get('REDIS_URL')
        
        if app_instance.config.get('CACHE_TYPE') == 'redis' and REDIS_URL:
            try:
                import redis
                r = redis.from_url(REDIS_URL)
                redis_info = r.info()
                cache_info['redis'] = {
                    'used_memory': redis_info.get('used_memory_human', 'N/A'),
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'total_commands_processed': redis_info.get('total_commands_processed', 0),
                    'uptime_in_seconds': redis_info.get('uptime_in_seconds', 0)
                }
            except Exception as e:
                cache_info['redis'] = {'status': f'Unable to connect: {str(e)}'}
        
        return jsonify(cache_info), 200
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return jsonify({'error': 'Failed to get cache stats'}), 500

@admin_bp.route('/trending/unified-score/<int:content_id>', methods=['GET'])
@require_admin
def get_unified_trending_score(current_user, content_id):
    """Get detailed unified trending score for a specific content"""
    try:
        from trending import get_trending_service
        
        language = request.args.get('language', 'english')
        
        trending_service = get_trending_service()
        if not trending_service:
            return jsonify({'error': 'Trending service not initialized'}), 500
        
        # Calculate unified score
        trending_content = trending_service.unified_engine.calculate_unified_score(
            content_id, language
        )
        
        response = {
            'content_id': content_id,
            'language': language,
            'unified_score': trending_content.unified_score,
            'confidence': trending_content.confidence,
            'category': trending_content.category.value,
            'trending_reasons': trending_content.trending_reasons,
            'metrics': {
                'tmdb_score': trending_content.metrics.tmdb_score,
                'box_office_score': trending_content.metrics.box_office_score,
                'ott_score': trending_content.metrics.ott_score,
                'social_score': trending_content.metrics.social_score,
                'search_score': trending_content.metrics.search_score,
                'velocity': trending_content.metrics.velocity,
                'momentum': trending_content.metrics.momentum,
                'viral_score': trending_content.metrics.viral_score,
                'geographic_score': trending_content.metrics.geographic_score
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Unified score error: {e}")
        return jsonify({'error': 'Failed to calculate unified score'}), 500

@admin_bp.route('/trending/refresh', methods=['POST'])
@require_admin
def refresh_trending_data(current_user):
    """Force refresh trending data and send update to Telegram"""
    try:
        from trending import get_trending_service
        
        languages = request.json.get('languages', ['telugu', 'english', 'hindi', 'tamil', 'malayalam'])
        send_telegram = request.json.get('send_telegram', False)
        
        trending_service = get_trending_service()
        if not trending_service:
            return jsonify({'error': 'Trending service not initialized'}), 500
        
        # Clear cache for each language
        for language in languages:
            cache_key = f"trending:{language}:latest"
            cache.delete(cache_key)
            cache_key = f"trending:{language}:movies"
            cache.delete(cache_key)
        
        # Trigger update
        updated_data = {}
        for language in languages:
            trending_service._update_language_trending(language)
            
            # Get updated data for Telegram
            if send_telegram:
                trending_data = trending_service._get_trending_movies([language], 5)
                updated_data[language] = trending_data
        
        # Send to Telegram if requested
        if send_telegram:
            for language, data in updated_data.items():
                TelegramService.send_trending_update(data, language.capitalize())
        
        return jsonify({
            'success': True,
            'message': f'Refreshed trending data for {", ".join(languages)}',
            'telegram_sent': send_telegram,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Refresh trending error: {e}")
        return jsonify({'error': 'Failed to refresh trending data'}), 500

@admin_bp.route('/telegram/test', methods=['POST'])
@require_admin
def test_telegram_message(current_user):
    """Test Telegram integration"""
    try:
        message_type = request.json.get('type', 'test')
        
        if message_type == 'test':
            test_message = f"""🔧 **Test Message from CineScope Admin**

Admin: {current_user.username}
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

✅ Telegram integration is working!

#Test #AdminPanel #CineScope"""
            
            if bot and TELEGRAM_CHANNEL_ID:
                bot.send_message(TELEGRAM_CHANNEL_ID, test_message, parse_mode='Markdown')
                return jsonify({'success': True, 'message': 'Test message sent'}), 200
            else:
                return jsonify({'success': False, 'message': 'Telegram bot not configured'}), 400
        
        return jsonify({'success': False, 'message': 'Invalid message type'}), 400
        
    except Exception as e:
        logger.error(f"Telegram test error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@admin_bp.route('/system/info', methods=['GET'])
@require_admin
def get_system_info(current_user):
    """Get system information and status"""
    try:
        import platform
        import sys
        import os
        
        system_info = {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': sys.version,
                'python_implementation': platform.python_implementation()
            },
            'environment': {
                'flask_env': os.environ.get('FLASK_ENV', 'production'),
                'debug_mode': app_instance.debug,
                'testing_mode': app_instance.testing
            },
            'services': {
                'database': bool(db),
                'cache': bool(cache),
                'telegram': bool(bot),
                'ml_service': bool(ML_SERVICE_URL)
            },
            'admin': {
                'username': current_user.username,
                'email': current_user.email,
                'last_active': current_user.last_active.isoformat() if current_user.last_active else None
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(system_info), 200
        
    except Exception as e:
        logger.error(f"System info error: {e}")
        return jsonify({'error': 'Failed to get system information'}), 500