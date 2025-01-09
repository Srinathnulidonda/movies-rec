# reviews/__init__.py
"""
CineBrain Reviews Module
Comprehensive review and rating management system
"""

from .review_rating import ReviewService
from .routes import reviews_bp
from .moderation import ReviewModerationService
from .analysis import ReviewAnalyticsService

def init_reviews_service(app, db, models, cache=None):
    """Initialize the complete reviews system"""
    try:
        # Initialize core review service
        review_service = ReviewService(db, models, cache)
        
        # Initialize moderation service
        moderation_service = ReviewModerationService(db, models, cache)
        
        # Initialize analytics service
        analytics_service = ReviewAnalyticsService(db, models, cache)
        
        # Store services in app context
        app.review_service = review_service
        app.review_moderation_service = moderation_service
        app.review_analytics_service = analytics_service
        
        # Register blueprint
        app.register_blueprint(reviews_bp, url_prefix='/api')
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info("✅ CineBrain Reviews system initialized successfully")
        
        return {
            'review_service': review_service,
            'moderation_service': moderation_service,
            'analytics_service': analytics_service
        }
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Failed to initialize CineBrain Reviews system: {e}")
        raise e

__all__ = ['init_reviews_service', 'reviews_bp', 'ReviewService', 'ReviewModerationService', 'ReviewAnalyticsService']
__version__ = '1.0.0'