# backend/personalized/__init__.py

"""
CineBrain Personalized Recommendation System
Advanced AI-Powered Entertainment Personalization Engine

A production-ready recommendation system that understands user preferences
at a cinematic DNA level and delivers highly personalized content recommendations.
"""

import logging
from typing import Optional, Dict, Any
from .profile_analyzer import (
    AdvancedProfileAnalyzer,
    CinematicDNAEngine,
    BehavioralIntelligenceSystem,
    UserEmbeddingGenerator
)

logger = logging.getLogger(__name__)

class CineBrainPersonalizationSystem:
    """
    Main orchestrator for CineBrain's personalization system
    """
    
    def __init__(self, db, models, cache=None, config=None):
        self.db = db
        self.models = models
        self.cache = cache
        self.config = config or {}
        
        # Initialize core components
        self.profile_analyzer = None
        self.recommendation_engine = None
        self._initialized = False
        
        # Performance monitoring
        self.performance_metrics = {
            'recommendations_served': 0,
            'avg_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'user_satisfaction_score': 0.0
        }
    
    def initialize(self):
        """Initialize all personalization components"""
        try:
            # Initialize profile analyzer
            self.profile_analyzer = AdvancedProfileAnalyzer(
                db=self.db,
                models=self.models,
                cache=self.cache
            )
            
            logger.info("✅ CineBrain Advanced Profile Analyzer initialized")
            
            # Recommendation engine will be initialized in Phase 2
            self._initialized = True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize CineBrain Personalization System: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if the system is ready for recommendations"""
        return self._initialized and self.profile_analyzer is not None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system health and performance status"""
        return {
            'initialized': self._initialized,
            'profile_analyzer_ready': self.profile_analyzer is not None,
            'recommendation_engine_ready': self.recommendation_engine is not None,
            'performance_metrics': self.performance_metrics,
            'cache_status': 'enabled' if self.cache else 'disabled'
        }

# Global instance (will be initialized by init_personalized_system)
_personalization_system: Optional[CineBrainPersonalizationSystem] = None

def init_personalized_system(app, db, models, services, cache=None):
    """
    Initialize the CineBrain Personalized Recommendation System
    
    Args:
        app: Flask application instance
        db: Database instance
        models: Dictionary of database models
        services: Dictionary of external services
        cache: Cache instance
    
    Returns:
        CineBrainPersonalizationSystem: Initialized system instance
    """
    global _personalization_system
    
    try:
        config = {
            'telugu_priority_weight': 1.0,
            'cultural_preference_boost': 0.3,
            'real_time_learning_rate': 0.1,
            'embedding_dimensions': 256,
            'max_recommendations': 100,
            'cache_ttl': 3600,
            'performance_tracking': True
        }
        
        _personalization_system = CineBrainPersonalizationSystem(
            db=db,
            models=models,
            cache=cache,
            config=config
        )
        
        if _personalization_system.initialize():
            logger.info("🚀 CineBrain Personalization System ready for production")
            return _personalization_system
        else:
            logger.error("❌ CineBrain Personalization System initialization failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ Critical error initializing CineBrain Personalization: {e}")
        return None

def get_personalization_system() -> Optional[CineBrainPersonalizationSystem]:
    """Get the global personalization system instance"""
    return _personalization_system

# Export main components
__all__ = [
    'CineBrainPersonalizationSystem',
    'AdvancedProfileAnalyzer',
    'CinematicDNAEngine',
    'BehavioralIntelligenceSystem',
    'UserEmbeddingGenerator',
    'init_personalized_system',
    'get_personalization_system'
]

__version__ = '3.0.0'
__author__ = 'SN'