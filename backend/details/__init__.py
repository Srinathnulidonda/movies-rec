"""
CineBrain Details Module
Comprehensive content details management system
"""

from .routes import details_bp
from .core import DetailsService
from .errors import DetailsError, APIError, ValidationError
from .validator import DetailsValidator
from .cache_manager import CacheManager

__version__ = "2.0.0"
__author__ = "CineBrain Team"

def init_details_module(app, db, models, cache=None):
    """Initialize the details module with Flask app"""
    try:
        # Initialize cache manager
        cache_manager = CacheManager(cache)
        
        # Initialize core service
        details_service = DetailsService(db, models, cache_manager)
        
        # Register blueprint
        app.register_blueprint(details_bp, url_prefix='/api/details')
        
        # Store service in app context
        app.details_service = details_service
        
        return details_service
        
    except Exception as e:
        app.logger.error(f"Failed to initialize CineBrain details module: {e}")
        raise e

__all__ = [
    'details_bp',
    'DetailsService', 
    'DetailsError',
    'APIError',
    'ValidationError',
    'DetailsValidator',
    'CacheManager',
    'init_details_module'
]