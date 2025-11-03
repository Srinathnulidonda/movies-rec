# backend/personalized/__init__.py

"""
CineBrain Advanced Personalization Engine
Next-generation recommendation system with cinematic DNA analysis and Telugu-first prioritization
"""

from .recommendation_engine import CineBrainRecommendationEngine
from .profile_analyzer import UserProfileAnalyzer, CinematicDNAAnalyzer
from .utils import VectorOperations, CacheManager, ContentEmbedding
from .metrics import PerformanceTracker, RecommendationMetrics
from .feedback import FeedbackProcessor, OnlineLearner

__version__ = "2.0.0"
__author__ = "CineBrain AI Team"

# Global engine instance
_recommendation_engine = None

def init_personalization_engine(app, db, models, cache=None):
    """Initialize the CineBrain personalization engine"""
    global _recommendation_engine
    
    try:
        _recommendation_engine = CineBrainRecommendationEngine(
            app=app,
            db=db, 
            models=models,
            cache=cache
        )
        
        app.logger.info("🧠 CineBrain Advanced Personalization Engine v2.0 initialized successfully")
        return _recommendation_engine
        
    except Exception as e:
        app.logger.error(f"Failed to initialize CineBrain personalization engine: {e}")
        return None

def get_recommendation_engine():
    """Get the global recommendation engine instance"""
    return _recommendation_engine

# Export key components
__all__ = [
    'CineBrainRecommendationEngine',
    'UserProfileAnalyzer', 
    'CinematicDNAAnalyzer',
    'VectorOperations',
    'CacheManager',
    'PerformanceTracker',
    'FeedbackProcessor',
    'init_personalization_engine',
    'get_recommendation_engine'
]