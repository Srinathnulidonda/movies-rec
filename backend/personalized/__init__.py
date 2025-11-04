# backend/personalized/__init__.py

"""
CineBrain Personalized Recommendation System
===========================================

A modern, continuously learning recommendation engine that adapts to user preferences
in real-time, similar to TikTok, YouTube, or Instagram feeds.

Key Features:
- Real-time user preference learning
- Dynamic user embeddings that evolve with interactions
- Hybrid recommendation algorithms (content-based, collaborative, graph-based)
- Telugu-first language prioritization
- Cinematic DNA analysis for sophisticated content matching
- Production-grade caching and performance optimization

Architecture:
- profile_analyzer.py: Continuous user preference learning
- recommendation_engine.py: Hybrid recommendation generation
- utils.py: Core utilities, embeddings, similarity calculations
- routes.py: Flask API endpoints
"""

__version__ = "2.0.0"
__author__ = "CineBrain Team"

# Core imports for external use
from .profile_analyzer import (
    UserProfileAnalyzer,
    CinematicDNAAnalyzer,
    RealTimeProfileUpdater
)

from .recommendation_engine import (
    HybridRecommendationEngine,
    ModernPersonalizationEngine,
    RecommendationOrchestrator
)

from .utils import (
    EmbeddingManager,
    SimilarityCalculator,
    CacheManager,
    PerformanceOptimizer
)

from .routes import personalized_bp

# Package-level configuration
PACKAGE_CONFIG = {
    'version': __version__,
    'telugu_priority': True,
    'real_time_learning': True,
    'cache_enabled': True,
    'performance_mode': 'production',
    'supported_algorithms': [
        'content_based',
        'collaborative_filtering', 
        'graph_based',
        'clustering',
        'cinematic_dna',
        'hybrid_scoring'
    ]
}

def initialize_personalization_system(app, db, models, services, cache=None):
    """
    Initialize the complete CineBrain personalization system
    
    Args:
        app: Flask application instance
        db: SQLAlchemy database instance
        models: Dictionary of database models
        services: Dictionary of external services
        cache: Cache backend (Redis/Simple)
    
    Returns:
        dict: Initialized personalization components
    """
    try:
        # Initialize core utilities
        embedding_manager = EmbeddingManager(cache=cache)
        similarity_calculator = SimilarityCalculator()
        cache_manager = CacheManager(cache=cache)
        
        # Initialize profile analyzer
        profile_analyzer = UserProfileAnalyzer(
            db=db,
            models=models,
            embedding_manager=embedding_manager,
            cache_manager=cache_manager
        )
        
        # Initialize recommendation engine
        recommendation_engine = ModernPersonalizationEngine(
            db=db,
            models=models,
            profile_analyzer=profile_analyzer,
            similarity_calculator=similarity_calculator,
            cache_manager=cache_manager
        )
        
        # Register Flask blueprint
        app.register_blueprint(personalized_bp, url_prefix='/api/personalized')
        
        # Store components in app context for access by routes
        app.personalization_system = {
            'profile_analyzer': profile_analyzer,
            'recommendation_engine': recommendation_engine,
            'embedding_manager': embedding_manager,
            'similarity_calculator': similarity_calculator,
            'cache_manager': cache_manager
        }
        
        return app.personalization_system
        
    except Exception as e:
        app.logger.error(f"Failed to initialize CineBrain personalization system: {e}")
        return None

# Export all public components
__all__ = [
    'UserProfileAnalyzer',
    'CinematicDNAAnalyzer', 
    'RealTimeProfileUpdater',
    'HybridRecommendationEngine',
    'ModernPersonalizationEngine',
    'RecommendationOrchestrator',
    'EmbeddingManager',
    'SimilarityCalculator',
    'CacheManager',
    'PerformanceOptimizer',
    'personalized_bp',
    'initialize_personalization_system',
    'PACKAGE_CONFIG'
]