# recommendation/router.py
from flask import Blueprint
import logging
from .new_releases import init_new_releases_routes
from .upcoming import init_upcoming_routes
from .critics_choice import init_critics_choice_routes
from .trending import init_trending_routes
from .similar import init_similar_routes
from .genre import init_genre_routes

logger = logging.getLogger(__name__)

# Create the main blueprint
recommendation_bp = Blueprint('recommendations', __name__, url_prefix='/api/recommendations')

def init_recommendation_routes(app, db, models, services, cache):
    """Initialize all recommendation routes with exact same endpoints as app.py"""
    
    try:
        # Initialize route handlers
        new_releases_handlers = init_new_releases_routes(app, db, models, services, cache)
        upcoming_handlers = init_upcoming_routes(app, db, models, services, cache)
        critics_choice_handlers = init_critics_choice_routes(app, db, models, services, cache)
        trending_handlers = init_trending_routes(app, db, models, services, cache)
        similar_handlers = init_similar_routes(app, db, models, services, cache)
        genre_handlers = init_genre_routes(app, db, models, services, cache)
        
        # === EXACT SAME ENDPOINTS AS ORIGINAL APP.PY ===
        
        # 1. /api/recommendations/trending
        @recommendation_bp.route('/trending', methods=['GET'])
        def get_trending():
            return trending_handlers['get_trending']()
        
        # 2. /api/recommendations/new-releases  
        @recommendation_bp.route('/new-releases', methods=['GET'])
        def get_new_releases():
            return new_releases_handlers['get_new_releases']()
        
        # 3. /api/recommendations/genre/<genre>
        @recommendation_bp.route('/genre/<genre>', methods=['GET'])
        def get_genre_recommendations(genre):
            return genre_handlers['get_genre_recommendations'](genre)
        
        # 4. /api/recommendations/regional/<language>
        @recommendation_bp.route('/regional/<language>', methods=['GET'])
        def get_regional(language):
            return genre_handlers['get_regional_recommendations'](language)
        
        # 5. /api/recommendations/anime
        @recommendation_bp.route('/anime', methods=['GET'])
        def get_anime():
            return genre_handlers['get_anime_recommendations']()
        
        # 6. /api/recommendations/similar/<int:content_id>
        @recommendation_bp.route('/similar/<int:content_id>', methods=['GET'])
        def get_similar_recommendations(content_id):
            return similar_handlers['get_similar_recommendations'](content_id)
        
        # 7. /api/recommendations/anonymous
        @recommendation_bp.route('/anonymous', methods=['GET'])
        def get_anonymous_recommendations():
            return genre_handlers['get_anonymous_recommendations']()
        
        # 8. /api/recommendations/admin-choice
        @recommendation_bp.route('/admin-choice', methods=['GET'])
        def get_public_admin_recommendations():
            return genre_handlers['get_admin_choice_recommendations']()
        
        # 9. /api/recommendations/critics-choice (from existing critics service)
        @recommendation_bp.route('/critics-choice', methods=['GET'])
        def get_enhanced_critics_choice():
            return critics_choice_handlers['get_enhanced_critics_choice']()
        
        # === ADMIN ENDPOINTS (outside recommendation blueprint) ===
        
        # 10. /api/admin/cinebrain/new-releases/stats
        @app.route('/api/admin/cinebrain/new-releases/stats', methods=['GET'])
        def get_cinebrain_new_releases_stats():
            return new_releases_handlers['get_new_releases_stats']()
        
        # 11. /api/admin/cinebrain/new-releases/refresh
        @app.route('/api/admin/cinebrain/new-releases/refresh', methods=['POST'])
        def trigger_cinebrain_new_releases_refresh():
            return new_releases_handlers['trigger_new_releases_refresh']()
        
        # 12. /api/admin/cinebrain/new-releases/config
        @app.route('/api/admin/cinebrain/new-releases/config', methods=['PUT'])
        def update_cinebrain_new_releases_config():
            return new_releases_handlers['update_new_releases_config']()
        
        # 13. /api/admin/critics-choice/refresh
        @app.route('/api/admin/critics-choice/refresh', methods=['POST'])
        def trigger_critics_refresh():
            return critics_choice_handlers['trigger_critics_refresh']()
        
        # 14. /api/admin/critics-choice/status
        @app.route('/api/admin/critics-choice/status', methods=['GET'])
        def get_critics_status():
            return critics_choice_handlers['get_critics_status']()
        
        # === UPCOMING ENDPOINTS (outside recommendation blueprint) ===
        
        # 15. /api/upcoming (async)
        @app.route('/api/upcoming', methods=['GET'])
        async def get_upcoming_releases():
            return await upcoming_handlers['get_upcoming_releases_async']()
        
        # 16. /api/upcoming-sync
        @app.route('/api/upcoming-sync', methods=['GET'])
        def get_upcoming_releases_sync():
            return upcoming_handlers['get_upcoming_releases']()
        
        logger.info("CineBrain recommendation routes initialized successfully with exact same endpoints")
        
    except Exception as e:
        logger.error(f"Error initializing CineBrain recommendation routes: {e}")
        raise

# Export the blueprint
__all__ = ['recommendation_bp', 'init_recommendation_routes']