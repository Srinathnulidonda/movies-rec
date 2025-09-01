#backend/trending.py
from __future__ import annotations
import logging
import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
import threading
import re
from bs4 import BeautifulSoup
import random
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler
from flask import current_app

logger = logging.getLogger(__name__)

# ================== Configuration & Constants ==================

# PRIORITY LANGUAGES - EXACT ORDER MATTERS!
PRIORITY_LANGUAGES = ['telugu', 'english', 'hindi', 'tamil', 'malayalam', 'kannada']

# Priority multipliers for guaranteed visibility
LANGUAGE_PRIORITY_MULTIPLIERS = {
    'telugu': 3.0,      # Highest priority
    'english': 2.5,     
    'hindi': 2.3,       
    'tamil': 2.2,       
    'malayalam': 2.0,   
    'kannada': 1.8      
}

class TrendingCategory(Enum):
    """Enhanced trending content categories"""
    BLOCKBUSTER_TRENDING = "blockbuster_trending"
    VIRAL_TRENDING = "viral_trending"
    STRONG_TRENDING = "strong_trending"
    MODERATE_TRENDING = "moderate_trending"
    RISING_FAST = "rising_fast"
    POPULAR_REGIONAL = "popular_regional"
    CROSS_LANGUAGE_TRENDING = "cross_language_trending"
    FESTIVAL_TRENDING = "festival_trending"
    CRITICS_TRENDING = "critics_trending"
    PRIORITY_LANGUAGE_TRENDING = "priority_language_trending"
    VECTOR_BASED_TRENDING = "vector_based_trending"

@dataclass
class VectorMetrics:
    """Vector-based metrics for advanced trending calculation"""
    content_vector: np.ndarray
    language_vector: np.ndarray
    temporal_vector: np.ndarray
    social_vector: np.ndarray
    similarity_score: float
    vector_magnitude: float
    vector_direction: np.ndarray
    cluster_id: int = -1
    cluster_confidence: float = 0.0

@dataclass
class LanguageConfig:
    """Enhanced language-specific configuration with vector weights"""
    code: str
    tmdb_code: str
    weight_matrix: Dict[str, float]
    vector_weights: np.ndarray
    trending_threshold: float
    momentum_threshold: float
    viral_z_score: float
    min_absolute_score: int
    market_adjustment: float
    priority_boost: float
    primary_regions: List[str]
    secondary_regions: List[str]
    festivals: List[Tuple[str, float]]
    cultural_vectors: np.ndarray

@dataclass
class ContentMetrics:
    """Enhanced metrics with vector support"""
    tmdb_id: int
    title: str
    language: str
    tmdb_score: float = 0
    box_office_score: float = 0
    ott_score: float = 0
    social_score: float = 0
    search_score: float = 0
    velocity: float = 0
    acceleration: float = 0
    momentum: float = 0
    viral_score: float = 0
    geographic_score: float = 0
    cross_language_score: float = 0
    festival_boost: float = 1.0
    priority_language_boost: float = 1.0
    vector_metrics: Optional[VectorMetrics] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def calculate_unified_score(self, config: LanguageConfig) -> float:
        """Enhanced unified score with priority language boost"""
        base_score = (
            config.weight_matrix['tmdb'] * self.tmdb_score +
            config.weight_matrix['box_office'] * self.box_office_score +
            config.weight_matrix['ott'] * self.ott_score +
            config.weight_matrix['social'] * self.social_score +
            config.weight_matrix['search'] * self.search_score
        )
        
        if self.vector_metrics and 'vector' in config.weight_matrix:
            vector_score = self.vector_metrics.similarity_score * 100
            base_score += config.weight_matrix['vector'] * vector_score
        
        momentum_boost = 1 + (self.momentum / 100) if self.momentum > 0 else 1
        viral_boost = 1 + (self.viral_score / 10) if self.viral_score > 0 else 1
        
        priority_boost = config.priority_boost if self.language in PRIORITY_LANGUAGES else 1.0
        
        final_score = (base_score * momentum_boost * viral_boost * 
                      self.festival_boost * (1 + self.geographic_score / 100) * 
                      priority_boost * self.priority_language_boost)
        
        return min(100, final_score)

@dataclass
class TrendingContent:
    """Trending content with all metadata"""
    content_id: int
    metrics: ContentMetrics
    category: TrendingCategory
    unified_score: float
    confidence: float
    predicted_peak_time: Optional[datetime] = None
    trending_reasons: List[str] = field(default_factory=list)

# Enhanced Language Configurations with Vector Support
LANGUAGE_CONFIGS = {
    'telugu': LanguageConfig(
        code='telugu',
        tmdb_code='te',
        weight_matrix={
            'tmdb': 0.3, 'box_office': 0.35, 'ott': 0.2, 
            'social': 0.15, 'search': 0.1, 'vector': 0.25
        },
        vector_weights=np.array([0.9, 0.8, 0.7, 0.85, 0.6, 0.95, 0.88, 0.92]),
        trending_threshold=75,
        momentum_threshold=12.0,
        viral_z_score=2.0,
        min_absolute_score=300,
        market_adjustment=3.5,
        priority_boost=3.0,
        primary_regions=['Hyderabad', 'Vijayawada', 'Visakhapatnam', 'Guntur', 'Warangal'],
        secondary_regions=['Bangalore', 'Chennai', 'Mumbai'],
        festivals=[
            ('Sankranti', 2.2), ('Ugadi', 1.8), 
            ('Vinayaka Chavithi', 1.6), ('Dussehra', 1.9),
            ('Diwali', 1.7), ('Dasara', 1.8)
        ],
        cultural_vectors=np.array([0.95, 0.85, 0.9, 0.8, 0.88, 0.92, 0.87, 0.9])
    ),
    'english': LanguageConfig(
        code='english',
        tmdb_code='en',
        weight_matrix={
            'tmdb': 0.25, 'box_office': 0.3, 'ott': 0.25,
            'social': 0.2, 'search': 0.1, 'vector': 0.2
        },
        vector_weights=np.array([0.85, 0.9, 0.95, 0.8, 0.88, 0.75, 0.82, 0.9]),
        trending_threshold=80,
        momentum_threshold=10.0,
        viral_z_score=1.8,
        min_absolute_score=3000,
        market_adjustment=1.0,
        priority_boost=2.5,
        primary_regions=['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'],
        secondary_regions=['All metros'],
        festivals=[
            ('Christmas', 1.5), ('New Year', 1.3),
            ('Halloween', 1.2), ('Valentine', 1.3),
            ('Summer Blockbuster', 1.4)
        ],
        cultural_vectors=np.array([0.8, 0.95, 0.85, 0.9, 0.78, 0.82, 0.88, 0.85])
    ),
    'hindi': LanguageConfig(
        code='hindi',
        tmdb_code='hi',
        weight_matrix={
            'tmdb': 0.25, 'box_office': 0.35, 'ott': 0.2,
            'social': 0.15, 'search': 0.1, 'vector': 0.22
        },
        vector_weights=np.array([0.88, 0.82, 0.85, 0.9, 0.75, 0.8, 0.86, 0.84]),
        trending_threshold=78,
        momentum_threshold=11.0,
        viral_z_score=1.9,
        min_absolute_score=1500,
        market_adjustment=1.5,
        priority_boost=2.3,
        primary_regions=['Mumbai', 'Delhi', 'Kolkata', 'Lucknow', 'Jaipur'],
        secondary_regions=['All metros', 'Tier 2 cities'],
        festivals=[
            ('Diwali', 1.8), ('Holi', 1.6),
            ('Karva Chauth', 1.3), ('Eid', 1.5),
            ('Raksha Bandhan', 1.4), ('Navratri', 1.5)
        ],
        cultural_vectors=np.array([0.85, 0.8, 0.88, 0.82, 0.9, 0.85, 0.83, 0.87])
    ),
    'tamil': LanguageConfig(
        code='tamil',
        tmdb_code='ta',
        weight_matrix={
            'tmdb': 0.3, 'box_office': 0.3, 'ott': 0.2,
            'social': 0.15, 'search': 0.1, 'vector': 0.23
        },
        vector_weights=np.array([0.87, 0.83, 0.88, 0.85, 0.78, 0.9, 0.84, 0.86]),
        trending_threshold=76,
        momentum_threshold=14.0,
        viral_z_score=2.2,
        min_absolute_score=500,
        market_adjustment=3.0,
        priority_boost=2.2,
        primary_regions=['Chennai', 'Coimbatore', 'Madurai', 'Trichy', 'Salem'],
        secondary_regions=['Bangalore', 'Mumbai', 'Kerala'],
        festivals=[
            ('Pongal', 2.1), ('Tamil New Year', 1.7),
            ('Navaratri', 1.5), ('Deepavali', 1.8),
            ('Karthigai Deepam', 1.4)
        ],
        cultural_vectors=np.array([0.9, 0.82, 0.87, 0.85, 0.88, 0.91, 0.86, 0.89])
    ),
    'malayalam': LanguageConfig(
        code='malayalam',
        tmdb_code='ml',
        weight_matrix={
            'tmdb': 0.35, 'box_office': 0.25, 'ott': 0.2,
            'social': 0.15, 'search': 0.1, 'vector': 0.25
        },
        vector_weights=np.array([0.85, 0.88, 0.82, 0.87, 0.8, 0.85, 0.83, 0.88]),
        trending_threshold=74,
        momentum_threshold=16.0,
        viral_z_score=2.5,
        min_absolute_score=200,
        market_adjustment=4.5,
        priority_boost=2.0,
        primary_regions=['Kochi', 'Trivandrum', 'Kozhikode', 'Thrissur', 'Kannur'],
        secondary_regions=['Bangalore', 'Chennai', 'Mumbai', 'Gulf countries'],
        festivals=[
            ('Onam', 2.3), ('Vishu', 1.7),
            ('Thrissur Pooram', 1.5), ('Eid', 1.4),
            ('Christmas', 1.6)
        ],
        cultural_vectors=np.array([0.88, 0.85, 0.83, 0.9, 0.82, 0.87, 0.85, 0.86])
    ),
    'kannada': LanguageConfig(
        code='kannada',
        tmdb_code='kn',
        weight_matrix={
            'tmdb': 0.32, 'box_office': 0.28, 'ott': 0.2,
            'social': 0.15, 'search': 0.1, 'vector': 0.22
        },
        vector_weights=np.array([0.83, 0.8, 0.85, 0.82, 0.78, 0.84, 0.81, 0.83]),
        trending_threshold=73,
        momentum_threshold=17.0,
        viral_z_score=2.6,
        min_absolute_score=150,
        market_adjustment=5.0,
        priority_boost=1.8,
        primary_regions=['Bangalore', 'Mysore', 'Hubli', 'Mangalore', 'Belgaum'],
        secondary_regions=['Chennai', 'Mumbai', 'Hyderabad'],
        festivals=[
            ('Ugadi', 1.8), ('Dasara', 2.0),
            ('Makara Sankranti', 1.6), ('Ganesha Chaturthi', 1.5),
            ('Deepavali', 1.7)
        ],
        cultural_vectors=np.array([0.82, 0.78, 0.84, 0.8, 0.85, 0.83, 0.79, 0.82])
    )
}

# Enhanced Cross-Language Influence Matrix
CROSS_LANGUAGE_INFLUENCE = {
    ('telugu', 'tamil'): 0.7,
    ('tamil', 'telugu'): 0.65,
    ('telugu', 'kannada'): 0.6,
    ('kannada', 'telugu'): 0.55,
    ('hindi', 'telugu'): 0.5,
    ('telugu', 'hindi'): 0.45,
    ('hindi', 'tamil'): 0.45,
    ('tamil', 'hindi'): 0.4,
    ('hindi', 'malayalam'): 0.4,
    ('malayalam', 'hindi'): 0.35,
    ('english', 'hindi'): 0.8,
    ('hindi', 'english'): 0.7,
    ('english', 'telugu'): 0.6,
    ('telugu', 'english'): 0.55,
    ('malayalam', 'tamil'): 0.4,
    ('tamil', 'malayalam'): 0.35,
    ('kannada', 'tamil'): 0.5,
    ('tamil', 'kannada'): 0.45
}

# ================== Advanced Algorithms Implementation ==================

class MultiSourceTrendingAlgorithm:
    """Multi-Source Real-Time Trending Score Algorithm (MRTSA)"""
    
    def __init__(self, session):
        self.session = session
        self.data_cache = {}
        self.cache_ttl = 300
        
    def calculate_recency_factor(self, timestamp: datetime, lambda_val: float = 0.05) -> float:
        time_diff = (datetime.utcnow() - timestamp).total_seconds() / 3600
        return math.exp(-lambda_val * time_diff)
    
    def normalize_score(self, score: float, max_score: float = 100) -> float:
        return min(100, (score / max_score) * 100)
    
    def aggregate_scores(self, content_id: int, language: str) -> ContentMetrics:
        config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['english'])
        metrics = ContentMetrics(tmdb_id=content_id, title='', language=language)
        
        metrics.tmdb_score = self._fetch_tmdb_score(content_id)
        metrics.box_office_score = self._fetch_box_office_score(content_id, language)
        metrics.ott_score = self._fetch_ott_score(content_id, language)
        metrics.social_score = self._fetch_social_score(content_id)
        metrics.search_score = self._fetch_search_score(content_id)
        
        for source in ['tmdb', 'box_office', 'ott', 'social', 'search']:
            recency = self.calculate_recency_factor(metrics.timestamp)
            score_attr = f"{source}_score"
            current_score = getattr(metrics, score_attr)
            setattr(metrics, score_attr, current_score * recency)
        
        return metrics
    
    def _fetch_tmdb_score(self, content_id: int) -> float:
        return random.uniform(60, 95)
    
    def _fetch_box_office_score(self, content_id: int, language: str) -> float:
        config = LANGUAGE_CONFIGS[language]
        base_score = random.uniform(40, 90)
        return base_score * config.market_adjustment
    
    def _fetch_ott_score(self, content_id: int, language: str) -> float:
        return random.uniform(50, 85)
    
    def _fetch_social_score(self, content_id: int) -> float:
        return random.uniform(45, 88)
    
    def _fetch_search_score(self, content_id: int) -> float:
        return random.uniform(35, 75)

class VelocityBasedTrendingDetection:
    """Velocity-Based Trending Detection (VBTD)"""
    
    def __init__(self):
        self.history_buffer = defaultdict(lambda: deque(maxlen=168))
        
    def calculate_velocity(self, content_id: int, current_score: float) -> Tuple[float, float]:
        history = self.history_buffer[content_id]
        
        if len(history) < 2:
            history.append(current_score)
            return 0.0, 0.0
        
        velocity = current_score - history[-1]
        
        acceleration = 0.0
        if len(history) >= 3:
            prev_velocity = history[-1] - history[-2]
            acceleration = velocity - prev_velocity
        
        history.append(current_score)
        return velocity, acceleration
    
    def calculate_momentum(self, velocity: float, acceleration: float, 
                          consistency_factor: float) -> float:
        alpha, beta, gamma = 0.5, 0.3, 0.2
        momentum = alpha * velocity + beta * acceleration + gamma * consistency_factor
        return momentum
    
    def calculate_consistency(self, content_id: int) -> float:
        history = list(self.history_buffer[content_id])
        if len(history) < 7:
            return 0.5
        
        velocities = [history[i] - history[i-1] for i in range(1, len(history))]
        if not velocities:
            return 0.5
            
        mean_velocity = np.mean(velocities)
        std_velocity = np.std(velocities)
        
        if mean_velocity == 0:
            return 0.5
            
        consistency = 1 - (std_velocity / abs(mean_velocity))
        return max(0, min(1, consistency))
    
    def is_trending_by_velocity(self, metrics: ContentMetrics, config: LanguageConfig) -> bool:
        consistency = self.calculate_consistency(metrics.tmdb_id)
        metrics.momentum = self.calculate_momentum(
            metrics.velocity, metrics.acceleration, consistency
        )
        
        return (metrics.momentum > config.momentum_threshold and 
                metrics.tmdb_score > config.min_absolute_score)

class GeographicCascadeModel:
    """Geographic Cascade Trending Model (GCTM)"""
    
    def __init__(self):
        self.city_influence_graph = self._build_influence_graph()
        
    def _build_influence_graph(self) -> Dict[Tuple[str, str], float]:
        return {
            ('Hyderabad', 'Vijayawada'): 0.8,
            ('Hyderabad', 'Bangalore'): 0.6,
            ('Chennai', 'Coimbatore'): 0.7,
            ('Chennai', 'Bangalore'): 0.5,
            ('Mumbai', 'Delhi'): 0.7,
            ('Mumbai', 'Pune'): 0.9,
            ('Kochi', 'Trivandrum'): 0.8,
            ('Kochi', 'Bangalore'): 0.4,
        }
    
    def calculate_cascade_probability(self, source_city: str, target_city: str,
                                     language_overlap: float = 0.5) -> float:
        geographic_influence = self.city_influence_graph.get(
            (source_city, target_city), 0.1
        )
        return geographic_influence * language_overlap
    
    def predict_geographic_spread(self, content_id: int, initial_cities: List[str],
                                 language: str) -> float:
        config = LANGUAGE_CONFIGS[language]
        spread_score = 0.0
        
        for primary in config.primary_regions:
            if primary in initial_cities:
                for secondary in config.secondary_regions:
                    cascade_prob = self.calculate_cascade_probability(
                        primary, secondary, 0.7
                    )
                    spread_score += cascade_prob
        
        max_possible_spread = len(config.primary_regions) * len(config.secondary_regions)
        if max_possible_spread > 0:
            spread_score = (spread_score / max_possible_spread) * 100
        
        return spread_score

class BoxOfficePerformancePredictor:
    """Box Office Performance Predictor (BOPP)"""
    
    def __init__(self):
        self.performance_cache = {}
        
    def calculate_performance_score(self, box_office_data: Dict, language: str) -> float:
        config = LANGUAGE_CONFIGS[language]
        
        collection = box_office_data.get('collection', 0)
        budget_estimate = box_office_data.get('budget', 1)
        screen_count = box_office_data.get('screens', 1)
        occupancy_rate = box_office_data.get('occupancy', 0.5)
        
        roi = (collection / budget_estimate) if budget_estimate > 0 else 0
        screen_utilization = occupancy_rate * (screen_count / 1000)
        
        performance_score = roi * screen_utilization * config.market_adjustment
        
        return min(100, performance_score * 10)
    
    def classify_trending_status(self, performance_score: float, 
                                growth_rate: float) -> TrendingCategory:
        if performance_score > 90 and growth_rate > 20:
            return TrendingCategory.BLOCKBUSTER_TRENDING
        elif performance_score > 75 and growth_rate > 15:
            return TrendingCategory.STRONG_TRENDING
        elif performance_score > 60 and growth_rate > 10:
            return TrendingCategory.MODERATE_TRENDING
        else:
            return TrendingCategory.RISING_FAST

class OTTPlatformAggregator:
    """OTT Platform Aggregation Algorithm (OPAA)"""
    
    PLATFORM_WEIGHTS = {
        'netflix': 0.3,
        'amazon_prime': 0.25,
        'disney_hotstar': 0.2,
        'sony_liv': 0.15,
        'aha': 0.1,
        'sun_nxt': 0.1,
        'koode': 0.1
    }
    
    def calculate_engagement_score(self, platform_data: Dict, language: str) -> float:
        total_score = 0.0
        
        for platform, weight in self.PLATFORM_WEIGHTS.items():
            if platform in platform_data:
                metrics = platform_data[platform]
                
                view_velocity = metrics.get('view_velocity', 0)
                completion_rate = metrics.get('completion_rate', 0)
                rating_momentum = metrics.get('rating_momentum', 0)
                watchlist_adds = metrics.get('watchlist_adds', 0)
                
                platform_score = (
                    view_velocity * 0.3 +
                    completion_rate * 0.3 +
                    rating_momentum * 0.2 +
                    watchlist_adds * 0.2
                )
                
                relevance = self._calculate_language_relevance(platform, language)
                total_score += platform_score * weight * relevance
        
        num_platforms = len(platform_data)
        cross_platform_boost = 1 + 0.2 * math.log(max(1, num_platforms))
        
        return min(100, total_score * cross_platform_boost)
    
    def _calculate_language_relevance(self, platform: str, language: str) -> float:
        relevance_map = {
            ('aha', 'telugu'): 1.0,
            ('sun_nxt', 'tamil'): 1.0,
            ('koode', 'malayalam'): 1.0,
            ('disney_hotstar', 'hindi'): 0.9,
            ('netflix', 'english'): 0.9,
            ('amazon_prime', 'english'): 0.8
        }
        return relevance_map.get((platform, language), 0.5)

class TemporalPatternRecognizer:
    """Temporal Pattern Recognition for Trending (TPRT)"""
    
    def __init__(self):
        self.current_festivals = self._get_current_festivals()
        
    def _get_current_festivals(self) -> Dict[str, float]:
        current_date = datetime.now()
        festivals = {}
        
        if current_date.month == 10:
            festivals['Dussehra'] = 1.7
            festivals['Diwali'] = 1.5
        elif current_date.month == 1:
            festivals['Pongal'] = 1.9
            festivals['Sankranti'] = 1.8
        elif current_date.month == 8:
            festivals['Onam'] = 2.0
        
        return festivals
    
    def calculate_temporal_score(self, base_score: float, language: str,
                                release_date: datetime, genre: str) -> float:
        config = LANGUAGE_CONFIGS[language]
        
        festival_multiplier = 1.0
        for festival, boost in config.festivals:
            if festival in self.current_festivals:
                festival_multiplier = max(festival_multiplier, boost)
        
        day_of_week = datetime.now().weekday()
        weekly_weights = {
            4: 1.4, 5: 1.6, 6: 1.5, 0: 0.8,
            1: 0.8, 2: 0.9, 3: 1.0
        }
        weekly_weight = weekly_weights.get(day_of_week, 1.0)
        
        month = datetime.now().month
        seasonal_adjustment = 1.0
        if month in [4, 5, 6]:
            seasonal_adjustment = 1.3
        elif month in [10, 11, 12, 1]:
            seasonal_adjustment = 1.5
        elif month in [3, 11]:
            seasonal_adjustment = 0.7
        
        genre_boosts = {
            'family': festival_multiplier * 1.2,
            'action': 1.1,
            'romance': 1.0,
            'comedy': festival_multiplier * 1.1
        }
        genre_boost = genre_boosts.get(genre.lower(), 1.0)
        
        temporal_score = (base_score * festival_multiplier * weekly_weight * 
                         seasonal_adjustment * genre_boost)
        
        return min(100, temporal_score)

class AnomalyDetector:
    """Anomaly Detection for Viral Content (ADVC)"""
    
    def __init__(self):
        self.baseline_history = defaultdict(lambda: deque(maxlen=30))
        
    def detect_viral_spike(self, content_id: int, current_score: float,
                          language: str) -> Tuple[bool, float]:
        config = LANGUAGE_CONFIGS[language]
        history = self.baseline_history[content_id]
        
        if len(history) < 7:
            history.append(current_score)
            return False, 0.0
        
        moving_avg = np.mean(history)
        std_dev = np.std(history)
        
        if std_dev == 0:
            std_dev = 1
        
        z_score = (current_score - moving_avg) / std_dev
        
        is_viral = (z_score > config.viral_z_score and 
                   current_score > config.min_absolute_score)
        
        viral_boost = min(3.0, z_score / 2) if is_viral else 1.0
        
        history.append(current_score)
        return is_viral, viral_boost

class CrossLanguageDetector:
    """Multi-Language Cross-Pollination Detection (MLCPD)"""
    
    def calculate_cross_language_score(self, content_metrics: Dict[str, ContentMetrics]) -> float:
        total_score = 0.0
        
        for source_lang, source_metrics in content_metrics.items():
            for target_lang in LANGUAGE_CONFIGS.keys():
                if source_lang != target_lang:
                    influence = CROSS_LANGUAGE_INFLUENCE.get(
                        (source_lang, target_lang), 0.1
                    )
                    
                    adaptability = self._calculate_adaptability(
                        source_metrics, source_lang, target_lang
                    )
                    
                    cross_score = (source_metrics.calculate_unified_score(LANGUAGE_CONFIGS[source_lang]) *
                                 influence * adaptability)
                    
                    total_score += cross_score
        
        num_pairs = len(content_metrics) * (len(LANGUAGE_CONFIGS) - 1)
        if num_pairs > 0:
            total_score = total_score / num_pairs
        
        return min(100, total_score)
    
    def _calculate_adaptability(self, metrics: ContentMetrics, 
                               source_lang: str, target_lang: str) -> float:
        return 0.7

# ================== New Vector-Based Algorithms ==================

class VectorSpaceModel:
    """Vector Space Model for Trending Analysis (VSMTA)"""
    
    def __init__(self):
        self.dimension = 8
        self.scaler = MinMaxScaler()
        self.cluster_centers = self._initialize_cluster_centers()
        
    def _initialize_cluster_centers(self) -> Dict[str, np.ndarray]:
        return {
            'blockbuster': np.array([0.95, 0.9, 0.88, 0.92, 0.85, 0.9, 0.93, 0.91]),
            'viral': np.array([0.7, 0.95, 0.98, 0.85, 0.9, 0.88, 0.8, 0.92]),
            'rising': np.array([0.6, 0.7, 0.85, 0.9, 0.95, 0.8, 0.75, 0.88]),
            'regional': np.array([0.85, 0.7, 0.6, 0.95, 0.8, 0.9, 0.88, 0.85]),
            'critical': np.array([0.98, 0.6, 0.5, 0.7, 0.95, 0.85, 0.9, 0.88])
        }
    
    def create_content_vector(self, metrics: ContentMetrics) -> np.ndarray:
        vector = np.array([
            metrics.tmdb_score / 100,
            metrics.box_office_score / 100,
            metrics.ott_score / 100,
            metrics.social_score / 100,
            min(1.0, abs(metrics.velocity) / 50),
            min(1.0, metrics.momentum / 100),
            min(1.0, metrics.viral_score / 10),
            min(1.0, metrics.geographic_score / 100)
        ])
        return vector
    
    def calculate_vector_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return 1 - cosine(v1, v2)
    
    def find_nearest_cluster(self, content_vector: np.ndarray) -> Tuple[str, float]:
        max_similarity = 0
        nearest_cluster = 'regional'
        
        for cluster_name, cluster_center in self.cluster_centers.items():
            similarity = self.calculate_vector_similarity(content_vector, cluster_center)
            if similarity > max_similarity:
                max_similarity = similarity
                nearest_cluster = cluster_name
        
        return nearest_cluster, max_similarity
    
    def calculate_vector_momentum(self, current_vector: np.ndarray, 
                                 previous_vector: np.ndarray,
                                 time_delta: float = 1.0) -> float:
        vector_change = current_vector - previous_vector
        magnitude = np.linalg.norm(vector_change)
        return magnitude / time_delta

class LanguagePriorityEngine:
    """Language Priority Optimization Engine (LPOE)"""
    
    def __init__(self):
        self.priority_queue = defaultdict(list)
        self.language_quotas = self._initialize_quotas()
        
    def _initialize_quotas(self) -> Dict[str, int]:
        base_quotas = {
            'telugu': 25,
            'english': 20,
            'hindi': 18,
            'tamil': 15,
            'malayalam': 12,
            'kannada': 10
        }
        return base_quotas
    
    def enforce_priority_distribution(self, content_list: List[TrendingContent],
                                     target_size: int = 20) -> List[TrendingContent]:
        by_language = defaultdict(list)
        for content in content_list:
            by_language[content.metrics.language].append(content)
        
        for lang in by_language:
            by_language[lang].sort(key=lambda x: x.unified_score, reverse=True)
        
        result = []
        used_quotas = defaultdict(int)
        
        for language in PRIORITY_LANGUAGES:
            if language in by_language:
                quota = min(
                    int(self.language_quotas[language] * target_size / 100),
                    len(by_language[language])
                )
                for i in range(quota):
                    result.append(by_language[language][i])
                    used_quotas[language] += 1
        
        remaining = []
        for language, contents in by_language.items():
            start_idx = used_quotas[language]
            remaining.extend(contents[start_idx:])
        
        remaining.sort(key=lambda x: x.unified_score, reverse=True)
        
        slots_left = target_size - len(result)
        result.extend(remaining[:slots_left])
        
        result.sort(key=lambda x: (
            -LANGUAGE_PRIORITY_MULTIPLIERS.get(x.metrics.language, 1.0),
            -x.unified_score
        ))
        
        return result[:target_size]
    
    def boost_priority_language_content(self, metrics: ContentMetrics) -> ContentMetrics:
        if metrics.language in PRIORITY_LANGUAGES:
            boost = LANGUAGE_PRIORITY_MULTIPLIERS[metrics.language]
            metrics.priority_language_boost = boost
            
            metrics.tmdb_score *= (1 + (boost - 1) * 0.3)
            metrics.social_score *= (1 + (boost - 1) * 0.2)
            metrics.ott_score *= (1 + (boost - 1) * 0.25)
        
        return metrics

class NeuralTrendingPredictor:
    """Neural-Inspired Trending Predictor (NITP)"""
    
    def __init__(self):
        self.weights = self._initialize_weights()
        self.bias = 0.1
        
    def _initialize_weights(self) -> np.ndarray:
        return np.array([
            [0.3, 0.25, 0.2, 0.15, 0.1],
            [0.35, 0.3, 0.2, 0.1, 0.05],
            [0.4, 0.25, 0.15, 0.15, 0.05]
        ])
    
    def sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))
    
    def relu(self, x: float) -> float:
        return max(0, x)
    
    def tanh(self, x: float) -> float:
        return math.tanh(x)
    
    def forward_pass(self, input_vector: np.ndarray) -> float:
        layer1 = np.dot(self.weights[0], input_vector) + self.bias
        layer1_activated = self.sigmoid(layer1)
        
        layer2 = np.dot(self.weights[1], input_vector) + self.bias
        layer2_activated = self.relu(layer2)
        
        layer3 = np.dot(self.weights[2], input_vector) + self.bias
        layer3_activated = self.tanh(layer3)
        
        final_score = (layer1_activated * 0.3 + 
                      layer2_activated * 0.4 + 
                      layer3_activated * 0.3)
        
        return min(1.0, max(0.0, final_score))
    
    def predict_trending_probability(self, metrics: ContentMetrics) -> float:
        input_vector = np.array([
            metrics.tmdb_score / 100,
            metrics.velocity / 50,
            metrics.momentum / 100,
            metrics.viral_score / 10,
            metrics.geographic_score / 100
        ])
        
        probability = self.forward_pass(input_vector)
        
        if metrics.language in PRIORITY_LANGUAGES:
            boost = LANGUAGE_PRIORITY_MULTIPLIERS[metrics.language]
            probability = min(1.0, probability * boost)
        
        return probability

class QuantumInspiredTrending:
    """Quantum-Inspired Superposition Trending (QIST)"""
    
    def __init__(self):
        self.quantum_states = self._initialize_quantum_states()
        
    def _initialize_quantum_states(self) -> Dict[str, complex]:
        return {
            'trending': complex(1, 0),
            'not_trending': complex(0, 1),
            'superposition': complex(0.707, 0.707)
        }
    
    def calculate_amplitude(self, metrics: ContentMetrics) -> complex:
        real = (metrics.tmdb_score + metrics.box_office_score) / 200
        imag = (metrics.velocity + metrics.momentum) / 200
        return complex(real, imag)
    
    def measure_trending_state(self, amplitude: complex) -> Tuple[str, float]:
        probability = abs(amplitude) ** 2
        
        if probability > 0.7:
            state = 'trending'
        elif probability < 0.3:
            state = 'not_trending'
        else:
            state = 'superposition'
        
        return state, probability
    
    def apply_quantum_boost(self, metrics: ContentMetrics) -> float:
        amplitude = self.calculate_amplitude(metrics)
        state, probability = self.measure_trending_state(amplitude)
        
        if metrics.language in PRIORITY_LANGUAGES:
            probability *= LANGUAGE_PRIORITY_MULTIPLIERS[metrics.language]
        
        boost = 1.0
        if state == 'trending':
            boost = 1.5 + probability
        elif state == 'superposition':
            boost = 1.2 + probability * 0.5
        
        return min(3.0, boost)

# ================== Enhanced Main Algorithms ==================

class EnhancedUnifiedTrendingScoreEngine:
    """Enhanced Unified Trending Score Algorithm with Priority Language Support"""
    
    def __init__(self, db, cache, tmdb_api_key):
        self.db = db
        self.cache = cache
        self.tmdb_api_key = tmdb_api_key
        
        session = self._create_http_session()
        self.mrtsa = MultiSourceTrendingAlgorithm(session)
        self.vbtd = VelocityBasedTrendingDetection()
        self.gctm = GeographicCascadeModel()
        self.bopp = BoxOfficePerformancePredictor()
        self.opaa = OTTPlatformAggregator()
        self.tprt = TemporalPatternRecognizer()
        self.advc = AnomalyDetector()
        self.mlcpd = CrossLanguageDetector()
        
        self.vsm = VectorSpaceModel()
        self.lpoe = LanguagePriorityEngine()
        self.nitp = NeuralTrendingPredictor()
        self.qist = QuantumInspiredTrending()
        
        self.component_weights = {
            'realtime': 0.15,
            'velocity': 0.12,
            'geographic': 0.08,
            'box_office': 0.12,
            'ott': 0.12,
            'temporal': 0.08,
            'viral': 0.08,
            'cross_language': 0.05,
            'vector': 0.08,
            'neural': 0.06,
            'quantum': 0.04,
            'priority': 0.02
        }
    
    def _create_http_session(self):
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
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
    
    def calculate_unified_score(self, content_id: int, language: str,
                               additional_data: Dict = None) -> TrendingContent:
        config = LANGUAGE_CONFIGS[language]
        
        metrics = self.mrtsa.aggregate_scores(content_id, language)
        metrics = self.lpoe.boost_priority_language_content(metrics)
        
        metrics.velocity, metrics.acceleration = self.vbtd.calculate_velocity(
            content_id, metrics.tmdb_score
        )
        
        content_vector = self.vsm.create_content_vector(metrics)
        cluster_name, cluster_confidence = self.vsm.find_nearest_cluster(content_vector)
        
        metrics.vector_metrics = VectorMetrics(
            content_vector=content_vector,
            language_vector=config.vector_weights,
            temporal_vector=np.random.rand(8),
            social_vector=np.random.rand(8),
            similarity_score=self.vsm.calculate_vector_similarity(
                content_vector, config.vector_weights
            ),
            vector_magnitude=np.linalg.norm(content_vector),
            vector_direction=content_vector / (np.linalg.norm(content_vector) + 1e-10),
            cluster_id=list(self.vsm.cluster_centers.keys()).index(cluster_name),
            cluster_confidence=cluster_confidence
        )
        
        initial_cities = config.primary_regions[:2]
        metrics.geographic_score = self.gctm.predict_geographic_spread(
            content_id, initial_cities, language
        )
        
        if additional_data and 'box_office' in additional_data:
            metrics.box_office_score = self.bopp.calculate_performance_score(
                additional_data['box_office'], language
            )
        
        if additional_data and 'ott_data' in additional_data:
            metrics.ott_score = self.opaa.calculate_engagement_score(
                additional_data['ott_data'], language
            )
        
        base_unified = metrics.calculate_unified_score(config)
        temporal_score = self.tprt.calculate_temporal_score(
            base_unified, language, 
            datetime.now(),
            'action'
        )
        
        is_viral, viral_boost = self.advc.detect_viral_spike(
            content_id, metrics.tmdb_score, language
        )
        metrics.viral_score = viral_boost * 10 if is_viral else 0
        
        neural_probability = self.nitp.predict_trending_probability(metrics)
        quantum_boost = self.qist.apply_quantum_boost(metrics)
        
        component_scores = {
            'realtime': metrics.calculate_unified_score(config),
            'velocity': min(100, abs(metrics.momentum) * 5),
            'geographic': metrics.geographic_score,
            'box_office': metrics.box_office_score,
            'ott': metrics.ott_score,
            'temporal': temporal_score,
            'viral': metrics.viral_score,
            'cross_language': 0,
            'vector': metrics.vector_metrics.similarity_score * 100,
            'neural': neural_probability * 100,
            'quantum': quantum_boost * 20,
            'priority': metrics.priority_language_boost * 10
        }
        
        final_score = sum(
            self.component_weights[component] * score 
            for component, score in component_scores.items()
        )
        
        if language in PRIORITY_LANGUAGES:
            final_score *= (1 + (config.priority_boost - 1) * 0.5)
        
        category = self._determine_category(final_score, metrics, config, cluster_name)
        confidence = self._calculate_confidence(component_scores)
        
        trending_content = TrendingContent(
            content_id=content_id,
            metrics=metrics,
            category=category,
            unified_score=min(100, final_score),
            confidence=confidence,
            trending_reasons=self._generate_reasons(metrics, component_scores, cluster_name)
        )
        
        return trending_content
    
    def _determine_category(self, score: float, metrics: ContentMetrics,
                           config: LanguageConfig, cluster_name: str) -> TrendingCategory:
        if metrics.language in PRIORITY_LANGUAGES[:3]:
            if score > 70:
                return TrendingCategory.PRIORITY_LANGUAGE_TRENDING
        
        if metrics.vector_metrics and metrics.vector_metrics.cluster_confidence > 0.8:
            return TrendingCategory.VECTOR_BASED_TRENDING
        
        if cluster_name == 'blockbuster' and score > 80:
            return TrendingCategory.BLOCKBUSTER_TRENDING
        elif cluster_name == 'viral' or metrics.viral_score > 20:
            return TrendingCategory.VIRAL_TRENDING
        elif score > config.trending_threshold:
            return TrendingCategory.STRONG_TRENDING
        elif cluster_name == 'rising' or (score > 60 and metrics.velocity > 0):
            return TrendingCategory.RISING_FAST
        elif cluster_name == 'regional' or metrics.geographic_score > 70:
            return TrendingCategory.POPULAR_REGIONAL
        elif cluster_name == 'critical':
            return TrendingCategory.CRITICS_TRENDING
        else:
            return TrendingCategory.MODERATE_TRENDING
    
    def _calculate_confidence(self, component_scores: Dict[str, float]) -> float:
        scores = [s for s in component_scores.values() if s > 0]
        if not scores:
            return 0.5
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if mean_score > 0:
            confidence = 1 - (std_score / mean_score)
        else:
            confidence = 0.5
        
        return max(0.3, min(1.0, confidence))
    
    def _generate_reasons(self, metrics: ContentMetrics, 
                         component_scores: Dict[str, float],
                         cluster_name: str) -> List[str]:
        reasons = []
        
        if metrics.language in PRIORITY_LANGUAGES:
            reasons.append(f"Priority {metrics.language.title()} content")
        
        if metrics.vector_metrics and metrics.vector_metrics.cluster_confidence > 0.7:
            reasons.append(f"High {cluster_name} similarity")
        
        if metrics.viral_score > 0:
            reasons.append("Viral spike detected")
        if metrics.momentum > 15:
            reasons.append("High momentum growth")
        if component_scores.get('neural', 0) > 80:
            reasons.append("AI predicts high trending potential")
        if component_scores.get('quantum', 0) > 60:
            reasons.append("Multi-state trending detected")
        if component_scores['box_office'] > 80:
            reasons.append("Strong box office performance")
        if component_scores['ott'] > 75:
            reasons.append("Popular on streaming platforms")
        if component_scores['temporal'] > 85:
            reasons.append("Festival/seasonal boost")
        if metrics.geographic_score > 70:
            reasons.append("Spreading across regions")
        
        return reasons

# ================== Main Enhanced Trending Service ==================

class AdvancedTrendingService:
    """Enhanced Trending Service with 100% Priority Language Support"""
    
    def __init__(self, db, cache, tmdb_api_key, app=None):
        self.db = db
        self.cache = cache
        self.tmdb_api_key = tmdb_api_key
        self.app = app  # Store Flask app instance
        self.unified_engine = EnhancedUnifiedTrendingScoreEngine(db, cache, tmdb_api_key)
        self.priority_engine = LanguagePriorityEngine()
        self.session = self._create_http_session()
        self.base_url = 'https://api.themoviedb.org/3'
        
        self.update_thread = None
        self.stop_updates = False
        
        self.priority_cache = {}
        self.cache_lock = threading.Lock()
    
    def _create_http_session(self):
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        session = requests.Session()
        retry = Retry(
            total=3,
            read=3,
            connect=3,
            backoff_factor=0.3,
            status_forcelist=(500, 502, 504)
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def _get_app_context(self):
        """Get or create Flask app context"""
        try:
            # Try to get current app
            if current_app:
                return current_app._get_current_object()
        except:
            pass
        
        # Fall back to stored app
        if self.app:
            return self.app
        
        # Last resort - try to import app
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from app import app
            return app
        except:
            logger.error("Could not get Flask app instance")
            return None
    
    def start_background_updates(self):
        if not self.update_thread:
            self.stop_updates = False
            self.update_thread = threading.Thread(target=self._update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            logger.info("Started background trending updates with priority language support")
    
    def stop_background_updates(self):
        self.stop_updates = True
        if self.update_thread:
            self.update_thread.join(timeout=5)
            self.update_thread = None
            logger.info("Stopped background trending updates")
    
    def _update_loop(self):
        while not self.stop_updates:
            try:
                app = self._get_app_context()
                if app:
                    with app.app_context():
                        for language in PRIORITY_LANGUAGES:
                            self._update_language_trending(language, is_priority=True)
                else:
                    logger.warning("No app context available for background updates")
                
                time.sleep(180)
                
            except Exception as e:
                logger.error(f"Error in background update: {e}")
                time.sleep(60)
    
    def _update_language_trending(self, language: str, is_priority: bool = False):
        """Update trending data for specific language"""
        try:
            cache_key = f"trending:{language}:latest"
            
            config = LANGUAGE_CONFIGS[language]
            
            num_pages = 3 if is_priority else 1
            all_movies = []
            
            for page in range(1, num_pages + 1):
                params = {
                    'api_key': self.tmdb_api_key,
                    'with_original_language': config.tmdb_code,
                    'sort_by': 'popularity.desc',
                    'page': page,
                    'primary_release_date.gte': (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                }
                
                if is_priority:
                    params['region'] = 'IN'
                
                response = self.session.get(f"{self.base_url}/discover/movie", 
                                          params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    all_movies.extend(data.get('results', []))
            
            trending_list = []
            
            for movie in all_movies[:30 if is_priority else 20]:
                trending_content = self.unified_engine.calculate_unified_score(
                    movie['id'], language, {'tmdb_data': movie}
                )
                
                threshold = config.trending_threshold * 0.8 if is_priority else config.trending_threshold
                
                if trending_content.unified_score > threshold:
                    trending_list.append(trending_content)
            
            trending_list.sort(key=lambda x: x.unified_score, reverse=True)
            
            cache_ttl = 300 if is_priority else 600
            self.cache.set(cache_key, trending_list, timeout=cache_ttl)
            
            if is_priority:
                with self.cache_lock:
                    self.priority_cache[language] = trending_list
            
            logger.info(f"Updated {language} trending: {len(trending_list)} items (priority={is_priority})")
            
        except Exception as e:
            logger.error(f"Error updating {language} trending: {e}")
    
    def get_trending(self, languages: List[str] = None, 
                    categories: List[str] = None,
                    limit: int = 20) -> Dict[str, List[Dict]]:
        """Get trending content with guaranteed priority language visibility"""
        if not languages:
            languages = PRIORITY_LANGUAGES
        else:
            languages = list(set(PRIORITY_LANGUAGES + languages))
        
        if not categories:
            categories = ['trending_movies', 'trending_tv', 'trending_anime', 
                        'rising_fast', 'popular_regional', 'priority_trending']
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {}
            
            if 'priority_trending' in categories:
                futures['priority_trending'] = executor.submit(
                    self._get_priority_trending, PRIORITY_LANGUAGES, limit
                )
            
            if 'trending_movies' in categories:
                futures['trending_movies'] = executor.submit(
                    self._get_trending_movies, languages, limit
                )
            
            if 'trending_tv' in categories:
                futures['trending_tv'] = executor.submit(
                    self._get_trending_tv, languages, limit
                )
            
            if 'trending_anime' in categories:
                futures['trending_anime'] = executor.submit(
                    self._get_trending_anime, limit
                )
            
            if 'rising_fast' in categories:
                futures['rising_fast'] = executor.submit(
                    self._get_rising_fast, languages, limit
                )
            
            if 'popular_regional' in categories:
                futures['popular_regional'] = executor.submit(
                    self._get_popular_regional, languages, limit
                )
            
            for category, future in futures.items():
                try:
                    category_results = future.result(timeout=15)
                    if category != 'trending_anime':
                        category_results = self._ensure_priority_distribution(
                            category_results, limit
                        )
                    results[category] = category_results
                except Exception as e:
                    logger.error(f"Error getting {category}: {e}")
                    results[category] = []
        
        return results
    
    def _get_priority_trending(self, languages: List[str], limit: int) -> List[Dict]:
        """Get trending content specifically for priority languages"""
        priority_trending = []
        per_language_limit = max(3, limit // len(languages))
        
        for language in languages:
            cache_key = f"trending:{language}:priority"
            cached = self.cache.get(cache_key)
            
            if cached:
                priority_trending.extend(cached[:per_language_limit])
            else:
                config = LANGUAGE_CONFIGS[language]
                params = {
                    'api_key': self.tmdb_api_key,
                    'with_original_language': config.tmdb_code,
                    'sort_by': 'popularity.desc',
                    'page': 1,
                    'region': 'IN'
                }
                
                response = self.session.get(f"{self.base_url}/discover/movie", 
                                          params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for movie in data.get('results', [])[:per_language_limit]:
                        try:
                            trending = self.unified_engine.calculate_unified_score(
                                movie['id'], language, {'tmdb_data': movie}
                            )
                            
                            content = self._save_content(movie, 'movie', language)
                            if content:
                                formatted = self._format_trending_content(
                                    content, trending
                                )
                                formatted['is_priority'] = True
                                priority_trending.append(formatted)
                        except Exception as e:
                            logger.error(f"Error processing priority movie: {e}")
                
                self.cache.set(cache_key, priority_trending[-per_language_limit:], timeout=300)
        
        priority_trending.sort(key=lambda x: (
            -LANGUAGE_PRIORITY_MULTIPLIERS.get(x.get('language', 'other'), 1.0),
            -x.get('unified_score', 0)
        ))
        
        return priority_trending[:limit]
    
    def _save_content(self, tmdb_data: Dict, content_type: str, language: str):
        """Save content to database and return a dictionary representation"""
        try:
            app = self._get_app_context()
            if not app:
                logger.error("No app context available for saving content")
                return None
            
            with app.app_context():
                # Import Content model inside app context
                from app import Content
                
                # Check if exists
                existing = self.db.session.query(Content).filter_by(
                    tmdb_id=tmdb_data['id']
                ).first()
                
                if existing:
                    # Update existing
                    existing.popularity = tmdb_data.get('popularity', existing.popularity)
                    existing.vote_count = tmdb_data.get('vote_count', existing.vote_count)
                    existing.rating = tmdb_data.get('vote_average', existing.rating)
                    existing.is_trending = True
                    existing.updated_at = datetime.utcnow()
                    
                    # Add language if not present
                    languages = json.loads(existing.languages or '[]')
                    if language not in languages:
                        languages.append(language)
                        existing.languages = json.dumps(languages)
                    
                    self.db.session.commit()
                    
                    # Return a dictionary representation instead of the ORM object
                    return {
                        'id': existing.id,
                        'tmdb_id': existing.tmdb_id,
                        'title': existing.title,
                        'content_type': existing.content_type,
                        'genres': existing.genres,
                        'languages': existing.languages,
                        'rating': existing.rating,
                        'poster_path': existing.poster_path,
                        'overview': existing.overview,
                        'popularity': existing.popularity,
                        'vote_count': existing.vote_count,
                        'is_trending': existing.is_trending
                    }
                
                # Create new
                new_content = Content(
                    tmdb_id=tmdb_data['id'],
                    title=tmdb_data.get('title') or tmdb_data.get('name'),
                    original_title=tmdb_data.get('original_title'),
                    content_type=content_type,
                    genres=json.dumps([]),
                    languages=json.dumps([language]),
                    release_date=None,
                    rating=tmdb_data.get('vote_average'),
                    vote_count=tmdb_data.get('vote_count'),
                    popularity=tmdb_data.get('popularity'),
                    overview=tmdb_data.get('overview'),
                    poster_path=tmdb_data.get('poster_path'),
                    backdrop_path=tmdb_data.get('backdrop_path'),
                    is_trending=True
                )
                
                self.db.session.add(new_content)
                self.db.session.commit()
                
                # Return a dictionary representation
                return {
                    'id': new_content.id,
                    'tmdb_id': new_content.tmdb_id,
                    'title': new_content.title,
                    'content_type': new_content.content_type,
                    'genres': new_content.genres,
                    'languages': new_content.languages,
                    'rating': new_content.rating,
                    'poster_path': new_content.poster_path,
                    'overview': new_content.overview,
                    'popularity': new_content.popularity,
                    'vote_count': new_content.vote_count,
                    'is_trending': new_content.is_trending
                }
                
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            try:
                self.db.session.rollback()
            except:
                pass
            return None

    def _format_trending_content(self, content, trending: TrendingContent) -> Dict:
        """Format content for API response - now accepts dict or object"""
        # Handle both dictionary and object inputs
        if isinstance(content, dict):
            content_dict = content
        else:
            # Convert object to dictionary if needed
            content_dict = {
                'id': getattr(content, 'id', random.randint(1000, 9999)),
                'tmdb_id': getattr(content, 'tmdb_id', None),
                'title': getattr(content, 'title', 'Unknown'),
                'content_type': getattr(content, 'content_type', 'movie'),
                'genres': getattr(content, 'genres', '[]'),
                'languages': getattr(content, 'languages', '[]'),
                'rating': getattr(content, 'rating', 0),
                'poster_path': getattr(content, 'poster_path', None),
                'overview': getattr(content, 'overview', '')
            }
        
        # Process poster URL
        poster_url = None
        if content_dict.get('poster_path'):
            if content_dict['poster_path'].startswith('http'):
                poster_url = content_dict['poster_path']
            else:
                poster_url = f"https://image.tmdb.org/t/p/w500{content_dict['poster_path']}"
        
        # Parse JSON fields
        genres = content_dict.get('genres', '[]')
        if isinstance(genres, str):
            try:
                genres = json.loads(genres)
            except:
                genres = []
        
        languages = content_dict.get('languages', '[]')
        if isinstance(languages, str):
            try:
                languages = json.loads(languages)
            except:
                languages = []
        
        result = {
            'id': content_dict.get('id', random.randint(1000, 9999)),
            'tmdb_id': content_dict.get('tmdb_id'),
            'title': content_dict.get('title', 'Unknown'),
            'content_type': content_dict.get('content_type', 'movie'),
            'genres': genres,
            'languages': languages,
            'rating': content_dict.get('rating', 0),
            'poster_path': poster_url,
            'overview': content_dict.get('overview', ''),
            'unified_score': trending.unified_score,
            'confidence': trending.confidence,
            'category': trending.category.value,
            'trending_reasons': trending.trending_reasons,
            'velocity': trending.metrics.velocity,
            'momentum': trending.metrics.momentum,
            'is_viral': trending.metrics.viral_score > 0,
            'is_trending': True,
            'is_priority_language': trending.metrics.language in PRIORITY_LANGUAGES,
            'language': trending.metrics.language,
            'priority_boost': LANGUAGE_PRIORITY_MULTIPLIERS.get(trending.metrics.language, 1.0)
        }
        
        if trending.metrics.vector_metrics:
            result['vector_score'] = trending.metrics.vector_metrics.similarity_score
            result['cluster_confidence'] = trending.metrics.vector_metrics.cluster_confidence
        
        return result
    
    def _ensure_priority_distribution(self, content_list: List[Dict], 
                                    limit: int) -> List[Dict]:
        """Ensure priority language distribution in results"""
        # Group by language
        by_language = defaultdict(list)
        for content in content_list:
            lang = content.get('language', 'unknown')
            by_language[lang].append(content)
        
        result = []
        used_ids = set()  # Use a set of IDs instead of objects
        
        # First pass: ensure each priority language gets at least one slot
        for language in PRIORITY_LANGUAGES:
            if language in by_language and by_language[language]:
                best = max(by_language[language], 
                        key=lambda x: x.get('unified_score', 0))
                content_id = best.get('id') or best.get('tmdb_id')  # Get unique identifier
                if content_id and content_id not in used_ids:
                    result.append(best)
                    used_ids.add(content_id)
        
        # Second pass: fill remaining slots by score
        all_remaining = []
        for contents in by_language.values():
            for c in contents:
                content_id = c.get('id') or c.get('tmdb_id')
                if content_id and content_id not in used_ids:
                    all_remaining.append(c)
        
        all_remaining.sort(key=lambda x: x.get('unified_score', 0), reverse=True)
        
        slots_left = limit - len(result)
        result.extend(all_remaining[:slots_left])
        
        return result[:limit]
    
    def _get_trending_movies(self, languages: List[str], limit: int) -> List[Dict]:
        """Get trending movies with priority language support"""
        all_trending = []
        
        for language in PRIORITY_LANGUAGES:
            if language not in languages:
                continue
                
            cache_key = f"trending:{language}:movies"
            cached = self.cache.get(cache_key)
            
            if cached:
                all_trending.extend(cached)
            else:
                config = LANGUAGE_CONFIGS[language]
                params = {
                    'api_key': self.tmdb_api_key,
                    'with_original_language': config.tmdb_code,
                    'sort_by': 'popularity.desc',
                    'page': 1,
                    'region': 'IN'
                }
                
                response = self.session.get(f"{self.base_url}/discover/movie", 
                                          params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for movie in data.get('results', [])[:10]:
                        try:
                            trending = self.unified_engine.calculate_unified_score(
                                movie['id'], language, {'tmdb_data': movie}
                            )
                            
                            threshold = config.trending_threshold * 0.85
                            
                            if trending.unified_score > threshold:
                                content = self._save_content(movie, 'movie', language)
                                if content:
                                    formatted = self._format_trending_content(
                                        content, trending
                                    )
                                    formatted['language'] = language
                                    all_trending.append(formatted)
                        except Exception as e:
                            logger.error(f"Error processing movie {movie['id']}: {e}")
                
                self.cache.set(cache_key, all_trending[-10:], timeout=1800)
        
        all_trending = self.priority_engine.enforce_priority_distribution(
            [self._dict_to_trending_content(d) for d in all_trending],
            limit
        )
        
        return [self._format_trending_content(tc.metrics, tc) 
                for tc in all_trending]
    
    def _get_trending_tv(self, languages: List[str], limit: int) -> List[Dict]:
        """Get trending TV shows with priority language support"""
        all_trending = []
        
        for language in PRIORITY_LANGUAGES:
            if language not in languages:
                continue
                
            config = LANGUAGE_CONFIGS[language]
            params = {
                'api_key': self.tmdb_api_key,
                'with_original_language': config.tmdb_code,
                'sort_by': 'popularity.desc',
                'page': 1,
                'region': 'IN'
            }
            
            response = self.session.get(f"{self.base_url}/discover/tv", 
                                      params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for show in data.get('results', [])[:5]:
                    try:
                        trending = self.unified_engine.calculate_unified_score(
                            show['id'], language, {'tmdb_data': show}
                        )
                        
                        threshold = config.trending_threshold * 0.85
                        if trending.unified_score > threshold:
                            content = self._save_content(show, 'tv', language)
                            if content:
                                formatted = self._format_trending_content(content, trending)
                                formatted['language'] = language
                                all_trending.append(formatted)
                    except Exception as e:
                        logger.error(f"Error processing TV show: {e}")
        
        all_trending = self._ensure_priority_distribution(all_trending, limit)
        return all_trending
    
    def _get_trending_anime(self, limit: int) -> List[Dict]:
        """Get trending anime"""
        cache_key = "trending:anime:latest"
        cached = self.cache.get(cache_key)
        
        if cached:
            return cached[:limit]
        
        anime_list = []
        
        try:
            jikan_url = "https://api.jikan.moe/v4/top/anime"
            params = {'limit': limit, 'type': 'tv', 'filter': 'airing'}
            
            response = self.session.get(jikan_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                for anime in data.get('data', []):
                    score = anime.get('score', 0) * 10
                    popularity = 100 - min(100, anime.get('popularity', 100) / 10)
                    
                    unified_score = (score * 0.6 + popularity * 0.4)
                    
                    formatted = {
                        'mal_id': anime['mal_id'],
                        'title': anime['title'],
                        'content_type': 'anime',
                        'genres': [g['name'] for g in anime.get('genres', [])],
                        'rating': anime.get('score'),
                        'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                        'overview': anime.get('synopsis'),
                        'unified_score': unified_score,
                        'category': 'trending_anime',
                        'language': 'japanese'
                    }
                    anime_list.append(formatted)
            
            self.cache.set(cache_key, anime_list, timeout=3600)
            
        except Exception as e:
            logger.error(f"Error fetching anime: {e}")
        
        return anime_list[:limit]
    
    def _get_rising_fast(self, languages: List[str], limit: int) -> List[Dict]:
        """Get rapidly rising content with priority language focus"""
        rising = []
        
        for language in PRIORITY_LANGUAGES:
            if language not in languages:
                continue
                
            config = LANGUAGE_CONFIGS[language]
            params = {
                'api_key': self.tmdb_api_key,
                'with_original_language': config.tmdb_code,
                'primary_release_date.gte': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'sort_by': 'popularity.desc',
                'page': 1,
                'region': 'IN'
            }
            
            response = self.session.get(f"{self.base_url}/discover/movie", 
                                      params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for movie in data.get('results', [])[:5]:
                    try:
                        trending = self.unified_engine.calculate_unified_score(
                            movie['id'], language, {'tmdb_data': movie}
                        )
                        
                        if trending.metrics.velocity > 3:
                            content = self._save_content(movie, 'movie', language)
                            if content:
                                formatted = self._format_trending_content(content, trending)
                                formatted['category'] = 'rising_fast'
                                formatted['language'] = language
                                rising.append(formatted)
                    except Exception as e:
                        logger.error(f"Error processing rising content: {e}")
        
        rising = self._ensure_priority_distribution(rising, limit)
        rising.sort(key=lambda x: x.get('velocity', 0), reverse=True)
        return rising
    
    def _get_popular_regional(self, languages: List[str], limit: int) -> List[Dict]:
        """Get regionally popular content with priority language focus"""
        regional = []
        
        for language in PRIORITY_LANGUAGES:
            if language not in languages:
                continue
                
            config = LANGUAGE_CONFIGS[language]
            params = {
                'api_key': self.tmdb_api_key,
                'with_original_language': config.tmdb_code,
                'region': 'IN',
                'sort_by': 'popularity.desc',
                'page': 1
            }
            
            response = self.session.get(f"{self.base_url}/discover/movie", 
                                      params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for movie in data.get('results', [])[:5]:
                    try:
                        trending = self.unified_engine.calculate_unified_score(
                            movie['id'], language, {'tmdb_data': movie}
                        )
                        
                        trending.metrics.geographic_score *= 2.0
                        
                        content = self._save_content(movie, 'movie', language)
                        if content:
                            formatted = self._format_trending_content(content, trending)
                            formatted['category'] = 'popular_regional'
                            formatted['region'] = config.primary_regions[0]
                            formatted['language'] = language
                            regional.append(formatted)
                    except Exception as e:
                        logger.error(f"Error processing regional content: {e}")
        
        regional = self._ensure_priority_distribution(regional, limit)
        return regional
    
    def _dict_to_trending_content(self, data: Dict) -> TrendingContent:
        """Convert dictionary back to TrendingContent object"""
        # Parse genres if it's a JSON string
        genres = data.get('genres', [])
        if isinstance(genres, str):
            try:
                genres = json.loads(genres)
            except:
                genres = []
        
        metrics = ContentMetrics(
            tmdb_id=data.get('tmdb_id', 0),
            title=data.get('title', ''),
            language=data.get('language', 'unknown'),
            tmdb_score=data.get('unified_score', 0),
            velocity=data.get('velocity', 0),
            momentum=data.get('momentum', 0),
            viral_score=10 if data.get('is_viral') else 0
        )
        
        return TrendingContent(
            content_id=data.get('id', 0),
            metrics=metrics,
            category=TrendingCategory.STRONG_TRENDING,
            unified_score=data.get('unified_score', 0),
            confidence=data.get('confidence', 0.5),
            trending_reasons=data.get('trending_reasons', [])
        )
# ================== Service Initialization ==================

def init_advanced_trending_service(db, cache, tmdb_api_key, app=None):
    """Initialize the advanced trending service with priority language support"""
    service = AdvancedTrendingService(db, cache, tmdb_api_key, app)
    service.start_background_updates()
    logger.info(f"Initialized trending service with priority languages: {PRIORITY_LANGUAGES}")
    return service

def get_trending_service():
    """Get the trending service instance"""
    return None