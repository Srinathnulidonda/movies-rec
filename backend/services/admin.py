# backend/admin.py
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

# Create admin blueprint
admin_bp = Blueprint('admin', __name__)

# Configure logging
logger = logging.getLogger(__name__)

cache = Cache()

# Initialize Telegram bot
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

# Will be initialized by main app
db = None
User = None
Content = None
UserInteraction = None
AdminRecommendation = None
TMDBService = None
JikanService = None
ContentService = None
MLServiceClient = None
http_session = None
ML_SERVICE_URL = None
cache = None
app = None

def init_admin(flask_app, database, models, services):
    """Initialize admin module with app context and models"""
    global db, User, Content, UserInteraction, AdminRecommendation
    global TMDBService, JikanService, ContentService, MLServiceClient
    global http_session, ML_SERVICE_URL, cache, app
    
    app = flask_app
    db = database
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    AdminRecommendation = models['AdminRecommendation']
    
    TMDBService = services['TMDBService']
    JikanService = services['JikanService']
    ContentService = services['ContentService']
    MLServiceClient = services['MLServiceClient']
    http_session = services['http_session']
    ML_SERVICE_URL = services['ML_SERVICE_URL']
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

# Telegram Service
class TelegramService:
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram bot or channel ID not configured")
                return False
            
            # Format genre list
            genres_list = []
            if content.genres:
                try:
                    genres_list = json.loads(content.genres)
                except:
                    genres_list = []
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Create message
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

#AdminChoice #MovieRecommendation #CineScope"""
            
            # Send message with photo if available
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

# Admin Routes
@admin_bp.route('/api/admin/search', methods=['GET'])
@require_admin
def admin_search(current_user):
    try:
        query = request.args.get('query', '')
        source = request.args.get('source', 'tmdb')  # tmdb, omdb, anime
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
        
        # Check if content already exists by external ID
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
        
        # Create new content from external data
        try:
            # Handle release date
            release_date = None
            if data.get('release_date'):
                try:
                    release_date = datetime.strptime(data['release_date'][:10], '%Y-%m-%d').date()
                except:
                    release_date = None
            
            # Get YouTube trailer
            youtube_trailer_id = ContentService.get_youtube_trailer(data.get('title'), data.get('content_type'))
            
            # Create content object
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

@admin_bp.route('/api/admin/recommendations', methods=['POST'])
@require_admin
def create_admin_recommendation(current_user):
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'recommendation_type', 'description']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Get content - handle both internal ID and external ID
        content = Content.query.get(data['content_id'])
        if not content:
            # Try to find by TMDB ID if direct ID lookup fails
            content = Content.query.filter_by(tmdb_id=data['content_id']).first()
        
        if not content:
            return jsonify({'error': 'Content not found. Please save content first.'}), 404
        
        # Create admin recommendation
        admin_rec = AdminRecommendation(
            content_id=content.id,
            admin_id=current_user.id,
            recommendation_type=data['recommendation_type'],
            description=data['description']
        )
        
        db.session.add(admin_rec)
        db.session.commit()
        
        # Send to Telegram channel
        telegram_success = TelegramService.send_admin_recommendation(content, current_user.username, data['description'])
        
        # Invalidate admin recommendations cache
        # cache.delete_memoized(get_public_admin_recommendations)
        
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
        # Get basic analytics
        total_users = User.query.count()
        total_content = Content.query.count()
        total_interactions = UserInteraction.query.count()
        active_users_last_week = User.query.filter(
            User.last_active >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        # Popular content
        popular_content = db.session.query(
            Content.id, Content.title, func.count(UserInteraction.id).label('interaction_count')
        ).join(UserInteraction).group_by(Content.id, Content.title)\
         .order_by(desc('interaction_count')).limit(10).all()
        
        # Popular genres
        all_interactions = UserInteraction.query.join(Content).all()
        genre_counts = defaultdict(int)
        for interaction in all_interactions:
            content = Content.query.get(interaction.content_id)
            if content and content.genres:
                genres = json.loads(content.genres)
                for genre in genres:
                    genre_counts[genre] += 1
        
        popular_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return jsonify({
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
        }), 200
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500

# ML Service Admin Routes
@admin_bp.route('/api/admin/ml-service-check', methods=['GET'])
@require_admin
def ml_service_comprehensive_check(current_user):
    """Simple comprehensive ML service check"""
    try:
        ml_url = ML_SERVICE_URL
        if not ml_url:
            return jsonify({
                'status': 'error',
                'message': 'ML_SERVICE_URL not configured',
                'checks': {}
            }), 500

        checks = {}
        overall_status = 'healthy'
        
        # 1. Basic Health Check
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

        # 2. Recommendation Test (only if connectivity passes)
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

        # 3. Statistics Check
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

        # 4. Quick Performance Test
        endpoints = [
            {'name': 'trending', 'url': '/api/trending?limit=3'},
        ]
        
        performance = {}
        for endpoint in endpoints:
            try:
                start_time = time.time()
                resp = http_session.get(f"{ml_url}{endpoint['url']}", timeout=10)
                response_time = time.time() - start_time
                
                performance[endpoint['name']] = {
                    'status': 'pass' if resp.status_code == 200 else 'fail',
                    'response_time': f"{response_time:.2f}s"
                }
            except Exception as e:
                performance[endpoint['name']] = {'status': 'fail', 'error': str(e)}

        checks['performance'] = performance

        # 5. Database Integration Check
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

        # Summary
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

@admin_bp.route('/api/admin/ml-service-update', methods=['POST'])
@require_admin  
def ml_service_force_update(current_user):
    """Force ML service model update"""
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

@admin_bp.route('/api/admin/ml-stats', methods=['GET'])
@require_admin
def get_ml_service_stats(current_user):
    """Get ML service statistics and performance metrics"""
    try:
        if not ML_SERVICE_URL:
            return jsonify({'error': 'ML service not configured'}), 400
        
        # Get ML service stats
        ml_stats = MLServiceClient.call_ml_service('/api/stats')
        
        if ml_stats:
            # Add backend stats for comparison
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

# Cache management endpoints
@admin_bp.route('/api/admin/cache/clear', methods=['POST'])
@require_admin
def clear_cache(current_user):
    """Clear all or specific cache entries"""
    try:
        cache_type = request.args.get('type', 'all')
        
        if cache_type == 'all':
            cache.clear()
            message = 'All cache cleared'
        elif cache_type == 'search':
            # Clear search-related cache
            cache.delete_memoized(TMDBService.search_content)
            cache.delete_memoized(JikanService.search_anime)
            message = 'Search cache cleared'
        elif cache_type == 'recommendations':
            # Clear recommendation-related cache
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
    """Get cache statistics"""
    try:
        # Get cache type and basic info
        cache_info = {
            'type': app.config.get('CACHE_TYPE', 'unknown'),
            'default_timeout': app.config.get('CACHE_DEFAULT_TIMEOUT', 0),
        }
        
        # Add Redis-specific stats if using Redis
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