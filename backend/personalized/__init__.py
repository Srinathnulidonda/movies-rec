# backend/personalized/__init__.py

__version__ = "3.0.0"
__author__ = "CineBrain AI Team"

from .profile_analyzer import UserProfileAnalyzer
from .utils import (
    EmbeddingManager,
    SimilarityEngine, 
    CacheManager,
    TeluguPriorityManager
)

_profile_analyzer = None
_embedding_manager = None
_similarity_engine = None
_cache_manager = None
_recommendation_engine = None

def get_profile_analyzer():
    global _profile_analyzer
    if _profile_analyzer is None:
        _profile_analyzer = UserProfileAnalyzer()
    return _profile_analyzer

def get_embedding_manager():
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager

def get_similarity_engine():
    global _similarity_engine
    if _similarity_engine is None:
        _similarity_engine = SimilarityEngine()
    return _similarity_engine

def get_cache_manager():
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def get_recommendation_engine():
    global _recommendation_engine
    if _recommendation_engine is None:
        from .recommendation_engine import HybridRecommendationEngine
        _recommendation_engine = HybridRecommendationEngine()
    return _recommendation_engine

def initialize_personalization_system(app=None, db=None, models=None, cache=None):
    global _profile_analyzer, _embedding_manager, _similarity_engine, _cache_manager, _recommendation_engine
    
    try:
        _cache_manager = CacheManager(cache_backend=cache)
        
        _embedding_manager = EmbeddingManager(cache_manager=_cache_manager)
        
        _similarity_engine = SimilarityEngine(
            embedding_manager=_embedding_manager,
            cache_manager=_cache_manager
        )
        
        _profile_analyzer = UserProfileAnalyzer(
            db=db,
            models=models,
            embedding_manager=_embedding_manager,
            similarity_engine=_similarity_engine,
            cache_manager=_cache_manager
        )
        
        from .recommendation_engine import HybridRecommendationEngine
        _recommendation_engine = HybridRecommendationEngine(
            db=db,
            models=models,
            cache_manager=_cache_manager
        )
        
        if models and db and cache:
            from .routes import init_personalized_routes
            init_personalized_routes(models, db, cache)
        
        if app:
            app.logger.info("🧠 CineBrain Personalization System v3.0 initialized successfully")
        
        return {
            'profile_analyzer': _profile_analyzer,
            'embedding_manager': _embedding_manager,
            'similarity_engine': _similarity_engine,
            'cache_manager': _cache_manager,
            'recommendation_engine': _recommendation_engine,
            'status': 'initialized'
        }
        
    except Exception as e:
        if app:
            app.logger.error(f"Failed to initialize CineBrain personalization system: {e}")
        return {'status': 'failed', 'error': str(e)}

__all__ = [
    'UserProfileAnalyzer',
    'EmbeddingManager', 
    'SimilarityEngine',
    'CacheManager',
    'TeluguPriorityManager',
    'get_profile_analyzer',
    'get_embedding_manager', 
    'get_similarity_engine',
    'get_cache_manager',
    'get_recommendation_engine',
    'initialize_personalization_system'
]