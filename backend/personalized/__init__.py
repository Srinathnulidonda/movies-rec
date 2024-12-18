#backend/personalized/__init__.py

"""
CineBrain Personalized Recommendation System
Advanced AI-powered recommendation engine with Cinematic DNA Analysis

This module provides:
- Advanced user behavior analysis and preference modeling
- Cinematic DNA extraction and matching
- Multi-algorithm hybrid recommendation engine
- Telugu-first regional prioritization
- Real-time preference adaptation

Components:
- ProfileAnalyzer: Deep user behavior understanding
- RecommendationEngine: Hybrid AI recommendation generation
- Routes: API endpoints for personalized content delivery
"""

from .profile_analyzer import (
    CinematicDNAAnalyzer,
    UserBehaviorAnalyzer, 
    PreferenceEmbeddingEngine,
    ProfileAnalyzer
)

from .recommendation_engine import (
    PersonalizedRecommendationEngine,
    AdaptiveAlgorithmMixer,
    ContentPersonalizer
)

from .routes import personalized_bp

__version__ = '3.0.0'
__author__ = 'CineBrain AI Team'

# Global instances (will be initialized by app)
profile_analyzer = None
recommendation_engine = None

def init_personalized_system(app, db, models, services, cache=None):
    """
    Initialize the complete CineBrain personalized recommendation system
    
    Args:
        app: Flask application instance
        db: SQLAlchemy database instance
        models: Dictionary of database models
        services: Dictionary of external services
        cache: Redis/Cache instance for performance optimization
    
    Returns:
        Tuple[ProfileAnalyzer, PersonalizedRecommendationEngine]: Initialized components
    """
    global profile_analyzer, recommendation_engine
    
    try:
        # Initialize Profile Analyzer with advanced Cinematic DNA
        profile_analyzer = ProfileAnalyzer(db, models, services, cache)
        
        # Initialize Recommendation Engine with adaptive algorithms
        recommendation_engine = PersonalizedRecommendationEngine(
            db, models, services, profile_analyzer, cache
        )
        
        # Configure Telugu-first priority and regional preferences
        recommendation_engine.configure_regional_priorities({
            'primary_languages': ['telugu', 'english', 'hindi'],
            'secondary_languages': ['malayalam', 'kannada', 'tamil'],
            'cultural_weights': {
                'indian_content': 1.2,
                'regional_content': 1.5,
                'telugu_content': 2.0
            }
        })
        
        # Initialize real-time learning capabilities
        recommendation_engine.enable_real_time_learning()
        
        print("✅ CineBrain Personalized Recommendation System initialized successfully")
        print(f"   📊 Profile Analyzer: {profile_analyzer.__class__.__name__}")
        print(f"   🤖 Recommendation Engine: {recommendation_engine.__class__.__name__}")
        print(f"   🎯 Telugu-first priority: Enabled")
        print(f"   ⚡ Real-time learning: Active")
        
        return profile_analyzer, recommendation_engine
        
    except Exception as e:
        print(f"❌ Failed to initialize CineBrain Personalized System: {e}")
        return None, None

def get_profile_analyzer():
    """Get the global profile analyzer instance"""
    return profile_analyzer

def get_recommendation_engine():
    """Get the global recommendation engine instance"""
    return recommendation_engine

# Export main components
__all__ = [
    'CinematicDNAAnalyzer',
    'UserBehaviorAnalyzer', 
    'PreferenceEmbeddingEngine',
    'ProfileAnalyzer',
    'PersonalizedRecommendationEngine',
    'AdaptiveAlgorithmMixer',
    'ContentPersonalizer',
    'personalized_bp',
    'init_personalized_system',
    'get_profile_analyzer',
    'get_recommendation_engine'
]