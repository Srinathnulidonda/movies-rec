#backend/trending.py
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
import random

logger = logging.getLogger(__name__)

# ================== Configuration & Constants ==================

class TrendingCategory(Enum):
    """Trending content categories"""
    BLOCKBUSTER_TRENDING = "blockbuster_trending"
    VIRAL_TRENDING = "viral_trending"
    STRONG_TRENDING = "strong_trending"
    MODERATE_TRENDING = "moderate_trending"
    RISING_FAST = "rising_fast"
    POPULAR_REGIONAL = "popular_regional"
    CROSS_LANGUAGE_TRENDING = "cross_language_trending"
    FESTIVAL_TRENDING = "festival_trending"
    CRITICS_TRENDING = "critics_trending"

@dataclass
class LanguageConfig:
    """Language-specific configuration"""
    code: str
    tmdb_code: str
    weight_matrix: Dict[str, float]
    trending_threshold: float
    momentum_threshold: float
    viral_z_score: float
    min_absolute_score: int
    market_adjustment: float
    primary_regions: List[str]
    secondary_regions: List[str]
    festivals: List[Tuple[str, float]]  # (festival_name, boost_factor)

# Language Configurations
LANGUAGE_CONFIGS = {
    'telugu': LanguageConfig(
        code='telugu',
        tmdb_code='te',
        weight_matrix={
            'tmdb': 0.3, 'box_office': 0.25, 'ott': 0.2, 
            'social': 0.15, 'search': 0.1
        },
        trending_threshold=82,
        momentum_threshold=15.0,
        viral_z_score=2.5,
        min_absolute_score=500,
        market_adjustment=2.5,
        primary_regions=['Hyderabad', 'Vijayawada', 'Visakhapatnam'],
        secondary_regions=['Bangalore', 'Chennai'],
        festivals=[
            ('Sankranti', 1.8), ('Ugadi', 1.6), 
            ('Vinayaka Chavithi', 1.4), ('Dussehra', 1.7)
        ]
    ),
    'hindi': LanguageConfig(
        code='hindi',
        tmdb_code='hi',
        weight_matrix={
            'tmdb': 0.25, 'box_office': 0.3, 'ott': 0.2,
            'social': 0.15, 'search': 0.1
        },
        trending_threshold=85,
        momentum_threshold=12.0,
        viral_z_score=2.0,
        min_absolute_score=2000,
        market_adjustment=1.0,
        primary_regions=['Mumbai', 'Delhi', 'Kolkata'],
        secondary_regions=['All metros'],
        festivals=[
            ('Diwali', 1.5), ('Holi', 1.4),
            ('Karva Chauth', 1.2), ('Eid', 1.3)
        ]
    ),
    'tamil': LanguageConfig(
        code='tamil',
        tmdb_code='ta',
        weight_matrix={
            'tmdb': 0.3, 'box_office': 0.25, 'ott': 0.2,
            'social': 0.15, 'search': 0.1
        },
        trending_threshold=80,
        momentum_threshold=18.0,
        viral_z_score=2.8,
        min_absolute_score=800,
        market_adjustment=2.8,
        primary_regions=['Chennai', 'Coimbatore', 'Madurai'],
        secondary_regions=['Bangalore', 'Mumbai'],
        festivals=[
            ('Pongal', 1.9), ('Tamil New Year', 1.5),
            ('Navaratri', 1.4), ('Deepavali', 1.6)
        ]
    ),
    'malayalam': LanguageConfig(
        code='malayalam',
        tmdb_code='ml',
        weight_matrix={
            'tmdb': 0.35, 'box_office': 0.2, 'ott': 0.2,
            'social': 0.15, 'search': 0.1
        },
        trending_threshold=78,
        momentum_threshold=20.0,
        viral_z_score=3.0,
        min_absolute_score=300,
        market_adjustment=4.0,
        primary_regions=['Kochi', 'Trivandrum', 'Kozhikode'],
        secondary_regions=['Bangalore', 'Chennai'],
        festivals=[
            ('Onam', 2.0), ('Vishu', 1.5),
            ('Thrissur Pooram', 1.3), ('Eid', 1.2)
        ]
    ),
    'english': LanguageConfig(
        code='english',
        tmdb_code='en',
        weight_matrix={
            'tmdb': 0.2, 'box_office': 0.25, 'ott': 0.25,
            'social': 0.2, 'search': 0.1
        },
        trending_threshold=88,
        momentum_threshold=8.0,
        viral_z_score=1.8,
        min_absolute_score=5000,
        market_adjustment=0.8,
        primary_regions=['Mumbai', 'Delhi', 'Bangalore'],
        secondary_regions=['All metros'],
        festivals=[
            ('Christmas', 1.3), ('New Year', 1.2),
            ('Halloween', 1.1), ('Valentine', 1.2)
        ]
    )
}

# Cross-Language Influence Matrix
CROSS_LANGUAGE_INFLUENCE = {
    ('telugu', 'tamil'): 0.6,
    ('tamil', 'telugu'): 0.5,
    ('hindi', 'telugu'): 0.4,
    ('hindi', 'tamil'): 0.4,
    ('hindi', 'malayalam'): 0.4,
    ('english', 'hindi'): 0.7,
    ('malayalam', 'tamil'): 0.3,
    ('tamil', 'malayalam'): 0.3
}

# ================== Data Classes ==================

@dataclass
class ContentMetrics:
    """Comprehensive metrics for content"""
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
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def calculate_unified_score(self, config: LanguageConfig) -> float:
        """Calculate unified trending score using language-specific weights"""
        base_score = (
            config.weight_matrix['tmdb'] * self.tmdb_score +
            config.weight_matrix['box_office'] * self.box_office_score +
            config.weight_matrix['ott'] * self.ott_score +
            config.weight_matrix['social'] * self.social_score +
            config.weight_matrix['search'] * self.search_score
        )
        
        # Apply momentum and viral boosts
        momentum_boost = 1 + (self.momentum / 100) if self.momentum > 0 else 1
        viral_boost = 1 + (self.viral_score / 10) if self.viral_score > 0 else 1
        
        # Apply all multipliers
        final_score = (base_score * momentum_boost * viral_boost * 
                      self.festival_boost * (1 + self.geographic_score / 100))
        
        return min(100, final_score)  # Cap at 100

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

# ================== Advanced Algorithms Implementation ==================

class MultiSourceTrendingAlgorithm:
    """
    Algorithm 1: Multi-Source Real-Time Trending Score Algorithm (MRTSA)
    Aggregates multiple data sources with weighted scoring
    """
    
    def __init__(self, session):
        self.session = session
        self.data_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    def calculate_recency_factor(self, timestamp: datetime, lambda_val: float = 0.05) -> float:
        """Calculate recency factor with exponential decay"""
        time_diff = (datetime.utcnow() - timestamp).total_seconds() / 3600  # hours
        return math.exp(-lambda_val * time_diff)
    
    def normalize_score(self, score: float, max_score: float = 100) -> float:
        """Normalize score to 0-100 range"""
        return min(100, (score / max_score) * 100)
    
    def aggregate_scores(self, content_id: int, language: str) -> ContentMetrics:
        """Aggregate scores from multiple sources"""
        config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['english'])
        metrics = ContentMetrics(tmdb_id=content_id, title='', language=language)
        
        # Fetch from various sources (simulated with realistic values)
        metrics.tmdb_score = self._fetch_tmdb_score(content_id)
        metrics.box_office_score = self._fetch_box_office_score(content_id, language)
        metrics.ott_score = self._fetch_ott_score(content_id, language)
        metrics.social_score = self._fetch_social_score(content_id)
        metrics.search_score = self._fetch_search_score(content_id)
        
        # Apply recency factors
        for source in ['tmdb', 'box_office', 'ott', 'social', 'search']:
            recency = self.calculate_recency_factor(metrics.timestamp)
            score_attr = f"{source}_score"
            current_score = getattr(metrics, score_attr)
            setattr(metrics, score_attr, current_score * recency)
        
        return metrics
    
    def _fetch_tmdb_score(self, content_id: int) -> float:
        """Fetch and normalize TMDb score"""
        # In production, make actual API call
        # Simulated realistic score
        return random.uniform(60, 95)
    
    def _fetch_box_office_score(self, content_id: int, language: str) -> float:
        """Fetch box office performance score"""
        # Simulate box office data fetching
        config = LANGUAGE_CONFIGS[language]
        base_score = random.uniform(40, 90)
        return base_score * config.market_adjustment
    
    def _fetch_ott_score(self, content_id: int, language: str) -> float:
        """Fetch OTT platform engagement score"""
        # Simulate OTT metrics
        return random.uniform(50, 85)
    
    def _fetch_social_score(self, content_id: int) -> float:
        """Fetch social media engagement score"""
        # Simulate social media metrics
        return random.uniform(45, 88)
    
    def _fetch_search_score(self, content_id: int) -> float:
        """Fetch search volume score"""
        # Simulate search trends
        return random.uniform(35, 75)

class VelocityBasedTrendingDetection:
    """
    Algorithm 2: Velocity-Based Trending Detection (VBTD)
    Measures rate of change in popularity metrics
    """
    
    def __init__(self):
        self.history_buffer = defaultdict(lambda: deque(maxlen=168))  # 7 days of hourly data
        
    def calculate_velocity(self, content_id: int, current_score: float) -> Tuple[float, float]:
        """Calculate velocity and acceleration"""
        history = self.history_buffer[content_id]
        
        if len(history) < 2:
            history.append(current_score)
            return 0.0, 0.0
        
        # Calculate velocity (rate of change)
        velocity = current_score - history[-1]
        
        # Calculate acceleration if enough history
        acceleration = 0.0
        if len(history) >= 3:
            prev_velocity = history[-1] - history[-2]
            acceleration = velocity - prev_velocity
        
        history.append(current_score)
        return velocity, acceleration
    
    def calculate_momentum(self, velocity: float, acceleration: float, 
                          consistency_factor: float) -> float:
        """Calculate momentum score"""
        alpha, beta, gamma = 0.5, 0.3, 0.2
        momentum = alpha * velocity + beta * acceleration + gamma * consistency_factor
        return momentum
    
    def calculate_consistency(self, content_id: int) -> float:
        """Calculate consistency factor based on velocity stability"""
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
        """Determine if content is trending based on velocity"""
        consistency = self.calculate_consistency(metrics.tmdb_id)
        metrics.momentum = self.calculate_momentum(
            metrics.velocity, metrics.acceleration, consistency
        )
        
        # Check language-specific thresholds
        return (metrics.momentum > config.momentum_threshold and 
                metrics.tmdb_score > config.min_absolute_score)

class GeographicCascadeModel:
    """
    Algorithm 3: Geographic Cascade Trending Model (GCTM)
    Models how trending spreads geographically
    """
    
    def __init__(self):
        self.city_influence_graph = self._build_influence_graph()
        
    def _build_influence_graph(self) -> Dict[Tuple[str, str], float]:
        """Build geographic influence graph between cities"""
        return {
            ('Hyderabad', 'Vijayawada'): 0.8,
            ('Hyderabad', 'Bangalore'): 0.6,
            ('Chennai', 'Coimbatore'): 0.7,
            ('Chennai', 'Bangalore'): 0.5,
            ('Mumbai', 'Delhi'): 0.7,
            ('Mumbai', 'Pune'): 0.9,
            ('Kochi', 'Trivandrum'): 0.8,
            ('Kochi', 'Bangalore'): 0.4,
            # Add more city pairs as needed
        }
    
    def calculate_cascade_probability(self, source_city: str, target_city: str,
                                     language_overlap: float = 0.5) -> float:
        """Calculate probability of trending cascade between cities"""
        geographic_influence = self.city_influence_graph.get(
            (source_city, target_city), 0.1
        )
        return geographic_influence * language_overlap
    
    def predict_geographic_spread(self, content_id: int, initial_cities: List[str],
                                 language: str) -> float:
        """Predict geographic spread score"""
        config = LANGUAGE_CONFIGS[language]
        spread_score = 0.0
        
        # Calculate spread from primary to secondary regions
        for primary in config.primary_regions:
            if primary in initial_cities:
                for secondary in config.secondary_regions:
                    cascade_prob = self.calculate_cascade_probability(
                        primary, secondary, 0.7
                    )
                    spread_score += cascade_prob
        
        # Normalize to 0-100
        max_possible_spread = len(config.primary_regions) * len(config.secondary_regions)
        if max_possible_spread > 0:
            spread_score = (spread_score / max_possible_spread) * 100
        
        return spread_score

class BoxOfficePerformancePredictor:
    """
    Algorithm 4: Box Office Performance Predictor (BOPP)
    Uses box office data to predict trending status
    """
    
    def __init__(self):
        self.performance_cache = {}
        
    def calculate_performance_score(self, box_office_data: Dict, language: str) -> float:
        """Calculate performance score from box office metrics"""
        config = LANGUAGE_CONFIGS[language]
        
        # Extract metrics
        collection = box_office_data.get('collection', 0)
        budget_estimate = box_office_data.get('budget', 1)
        screen_count = box_office_data.get('screens', 1)
        occupancy_rate = box_office_data.get('occupancy', 0.5)
        
        # Calculate base performance score
        roi = (collection / budget_estimate) if budget_estimate > 0 else 0
        screen_utilization = occupancy_rate * (screen_count / 1000)  # Normalize screens
        
        # Apply market normalization
        performance_score = roi * screen_utilization * config.market_adjustment
        
        return min(100, performance_score * 10)  # Scale to 0-100
    
    def classify_trending_status(self, performance_score: float, 
                                growth_rate: float) -> TrendingCategory:
        """Classify trending status based on performance"""
        if performance_score > 90 and growth_rate > 20:
            return TrendingCategory.BLOCKBUSTER_TRENDING
        elif performance_score > 75 and growth_rate > 15:
            return TrendingCategory.STRONG_TRENDING
        elif performance_score > 60 and growth_rate > 10:
            return TrendingCategory.MODERATE_TRENDING
        else:
            return TrendingCategory.RISING_FAST

class OTTPlatformAggregator:
    """
    Algorithm 5: OTT Platform Aggregation Algorithm (OPAA)
    Monitors OTT platforms for viewing trends
    """
    
    PLATFORM_WEIGHTS = {
        'netflix': 0.3,
        'amazon_prime': 0.25,
        'disney_hotstar': 0.2,
        'sony_liv': 0.15,
        'aha': 0.1,  # Telugu
        'sun_nxt': 0.1,  # Tamil
        'koode': 0.1  # Malayalam
    }
    
    def calculate_engagement_score(self, platform_data: Dict, language: str) -> float:
        """Calculate OTT engagement score"""
        total_score = 0.0
        
        for platform, weight in self.PLATFORM_WEIGHTS.items():
            if platform in platform_data:
                metrics = platform_data[platform]
                
                # Calculate platform-specific score
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
                
                # Apply language relevance
                relevance = self._calculate_language_relevance(platform, language)
                total_score += platform_score * weight * relevance
        
        # Cross-platform boost
        num_platforms = len(platform_data)
        cross_platform_boost = 1 + 0.2 * math.log(max(1, num_platforms))
        
        return min(100, total_score * cross_platform_boost)
    
    def _calculate_language_relevance(self, platform: str, language: str) -> float:
        """Calculate platform relevance for language"""
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
    """
    Algorithm 6: Temporal Pattern Recognition for Trending (TPRT)
    Identifies patterns based on time and cultural events
    """
    
    def __init__(self):
        self.current_festivals = self._get_current_festivals()
        
    def _get_current_festivals(self) -> Dict[str, float]:
        """Get currently active festivals and their boost factors"""
        # In production, use actual calendar
        # Simplified for demonstration
        current_date = datetime.now()
        festivals = {}
        
        # Check for major festivals (simplified)
        if current_date.month == 10:  # October - Dussehra/Diwali season
            festivals['Dussehra'] = 1.7
            festivals['Diwali'] = 1.5
        elif current_date.month == 1:  # January - Pongal/Sankranti
            festivals['Pongal'] = 1.9
            festivals['Sankranti'] = 1.8
        elif current_date.month == 8:  # August - Onam
            festivals['Onam'] = 2.0
        
        return festivals
    
    def calculate_temporal_score(self, base_score: float, language: str,
                                release_date: datetime, genre: str) -> float:
        """Calculate temporal pattern score"""
        config = LANGUAGE_CONFIGS[language]
        
        # Festival boost
        festival_multiplier = 1.0
        for festival, boost in config.festivals:
            if festival in self.current_festivals:
                festival_multiplier = max(festival_multiplier, boost)
        
        # Weekly pattern
        day_of_week = datetime.now().weekday()
        weekly_weights = {
            4: 1.4,  # Friday - New releases
            5: 1.6,  # Saturday - Weekend
            6: 1.5,  # Sunday - Weekend
            0: 0.8,  # Monday - Working day
            1: 0.8,  # Tuesday
            2: 0.9,  # Wednesday
            3: 1.0   # Thursday
        }
        weekly_weight = weekly_weights.get(day_of_week, 1.0)
        
        # Seasonal adjustment
        month = datetime.now().month
        seasonal_adjustment = 1.0
        if month in [4, 5, 6]:  # Summer vacation
            seasonal_adjustment = 1.3
        elif month in [10, 11, 12, 1]:  # Festival season
            seasonal_adjustment = 1.5
        elif month in [3, 11]:  # Exam season
            seasonal_adjustment = 0.7
        
        # Genre-specific boosts
        genre_boosts = {
            'family': festival_multiplier * 1.2,
            'action': 1.1,
            'romance': 1.0,
            'comedy': festival_multiplier * 1.1
        }
        genre_boost = genre_boosts.get(genre.lower(), 1.0)
        
        # Calculate final temporal score
        temporal_score = (base_score * festival_multiplier * weekly_weight * 
                         seasonal_adjustment * genre_boost)
        
        return min(100, temporal_score)

class AnomalyDetector:
    """
    Algorithm 9: Anomaly Detection for Viral Content (ADVC)
    Detects viral spikes in popularity
    """
    
    def __init__(self):
        self.baseline_history = defaultdict(lambda: deque(maxlen=30))
        
    def detect_viral_spike(self, content_id: int, current_score: float,
                          language: str) -> Tuple[bool, float]:
        """Detect if content has gone viral"""
        config = LANGUAGE_CONFIGS[language]
        history = self.baseline_history[content_id]
        
        if len(history) < 7:
            history.append(current_score)
            return False, 0.0
        
        # Calculate baseline statistics
        moving_avg = np.mean(history)
        std_dev = np.std(history)
        
        if std_dev == 0:
            std_dev = 1
        
        # Calculate Z-score
        z_score = (current_score - moving_avg) / std_dev
        
        # Check viral threshold
        is_viral = (z_score > config.viral_z_score and 
                   current_score > config.min_absolute_score)
        
        # Calculate viral boost
        viral_boost = min(3.0, z_score / 2) if is_viral else 1.0
        
        history.append(current_score)
        return is_viral, viral_boost

class CrossLanguageDetector:
    """
    Algorithm 13: Multi-Language Cross-Pollination Detection (MLCPD)
    Detects cross-language trending patterns
    """
    
    def calculate_cross_language_score(self, content_metrics: Dict[str, ContentMetrics]) -> float:
        """Calculate cross-language trending score"""
        total_score = 0.0
        
        for source_lang, source_metrics in content_metrics.items():
            for target_lang in LANGUAGE_CONFIGS.keys():
                if source_lang != target_lang:
                    influence = CROSS_LANGUAGE_INFLUENCE.get(
                        (source_lang, target_lang), 0.1
                    )
                    
                    # Check if content is adaptable
                    adaptability = self._calculate_adaptability(
                        source_metrics, source_lang, target_lang
                    )
                    
                    cross_score = (source_metrics.calculate_unified_score(LANGUAGE_CONFIGS[source_lang]) *
                                 influence * adaptability)
                    
                    total_score += cross_score
        
        # Normalize
        num_pairs = len(content_metrics) * (len(LANGUAGE_CONFIGS) - 1)
        if num_pairs > 0:
            total_score = total_score / num_pairs
        
        return min(100, total_score)
    
    def _calculate_adaptability(self, metrics: ContentMetrics, 
                               source_lang: str, target_lang: str) -> float:
        """Calculate content adaptability between languages"""
        # Universal themes have higher adaptability
        # In production, analyze actual genre and theme data
        return 0.7  # Default moderate adaptability

class UnifiedTrendingScoreEngine:
    """
    Algorithm 15: Unified Trending Score Algorithm (UTSA)
    Master algorithm combining all approaches
    """
    
    def __init__(self, db, cache, tmdb_api_key):
        self.db = db
        self.cache = cache
        self.tmdb_api_key = tmdb_api_key
        
        # Initialize all algorithm components
        session = self._create_http_session()
        self.mrtsa = MultiSourceTrendingAlgorithm(session)
        self.vbtd = VelocityBasedTrendingDetection()
        self.gctm = GeographicCascadeModel()
        self.bopp = BoxOfficePerformancePredictor()
        self.opaa = OTTPlatformAggregator()
        self.tprt = TemporalPatternRecognizer()
        self.advc = AnomalyDetector()
        self.mlcpd = CrossLanguageDetector()
        
        # Component weights (can be learned/adjusted)
        self.component_weights = {
            'realtime': 0.2,
            'velocity': 0.15,
            'geographic': 0.1,
            'box_office': 0.15,
            'ott': 0.15,
            'temporal': 0.1,
            'viral': 0.1,
            'cross_language': 0.05
        }
    
    def _create_http_session(self):
        """Create HTTP session with retry logic"""
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
        """Calculate unified trending score using all algorithms"""
        config = LANGUAGE_CONFIGS[language]
        
        # Get base metrics from MRTSA
        metrics = self.mrtsa.aggregate_scores(content_id, language)
        
        # Calculate velocity and momentum
        metrics.velocity, metrics.acceleration = self.vbtd.calculate_velocity(
            content_id, metrics.tmdb_score
        )
        
        # Geographic spread score
        initial_cities = config.primary_regions[:2]  # Simplified
        metrics.geographic_score = self.gctm.predict_geographic_spread(
            content_id, initial_cities, language
        )
        
        # Box office performance (if available)
        if additional_data and 'box_office' in additional_data:
            metrics.box_office_score = self.bopp.calculate_performance_score(
                additional_data['box_office'], language
            )
        
        # OTT engagement (if available)
        if additional_data and 'ott_data' in additional_data:
            metrics.ott_score = self.opaa.calculate_engagement_score(
                additional_data['ott_data'], language
            )
        
        # Temporal patterns
        base_unified = metrics.calculate_unified_score(config)
        temporal_score = self.tprt.calculate_temporal_score(
            base_unified, language, 
            datetime.now(),  # Use actual release date in production
            'action'  # Use actual genre in production
        )
        
        # Viral detection
        is_viral, viral_boost = self.advc.detect_viral_spike(
            content_id, metrics.tmdb_score, language
        )
        metrics.viral_score = viral_boost * 10 if is_viral else 0
        
        # Calculate final unified score
        component_scores = {
            'realtime': metrics.calculate_unified_score(config),
            'velocity': min(100, abs(metrics.momentum) * 5),
            'geographic': metrics.geographic_score,
            'box_office': metrics.box_office_score,
            'ott': metrics.ott_score,
            'temporal': temporal_score,
            'viral': metrics.viral_score,
            'cross_language': 0  # Calculate separately if needed
        }
        
        # Weighted average of all components
        final_score = sum(
            self.component_weights[component] * score 
            for component, score in component_scores.items()
        )
        
        # Determine category
        category = self._determine_category(final_score, metrics, config)
        
        # Calculate confidence
        confidence = self._calculate_confidence(component_scores)
        
        # Create trending content object
        trending_content = TrendingContent(
            content_id=content_id,
            metrics=metrics,
            category=category,
            unified_score=final_score,
            confidence=confidence,
            trending_reasons=self._generate_reasons(metrics, component_scores)
        )
        
        return trending_content
    
    def _determine_category(self, score: float, metrics: ContentMetrics,
                           config: LanguageConfig) -> TrendingCategory:
        """Determine trending category based on score and metrics"""
        if metrics.viral_score > 20:
            return TrendingCategory.VIRAL_TRENDING
        elif score > 85:
            return TrendingCategory.BLOCKBUSTER_TRENDING
        elif score > config.trending_threshold:
            return TrendingCategory.STRONG_TRENDING
        elif score > 60 and metrics.velocity > 0:
            return TrendingCategory.RISING_FAST
        elif metrics.geographic_score > 70:
            return TrendingCategory.POPULAR_REGIONAL
        else:
            return TrendingCategory.MODERATE_TRENDING
    
    def _calculate_confidence(self, component_scores: Dict[str, float]) -> float:
        """Calculate confidence score based on component agreement"""
        scores = list(component_scores.values())
        if not scores:
            return 0.5
        
        # Calculate variance in component scores
        mean_score = np.mean(scores)
        variance = np.var(scores)
        
        # Lower variance = higher confidence
        confidence = 1 - (variance / (mean_score + 1))
        return max(0.3, min(1.0, confidence))
    
    def _generate_reasons(self, metrics: ContentMetrics, 
                         component_scores: Dict[str, float]) -> List[str]:
        """Generate human-readable trending reasons"""
        reasons = []
        
        if metrics.viral_score > 0:
            reasons.append("Viral spike detected")
        if metrics.momentum > 15:
            reasons.append("High momentum growth")
        if component_scores['box_office'] > 80:
            reasons.append("Strong box office performance")
        if component_scores['ott'] > 75:
            reasons.append("Popular on streaming platforms")
        if component_scores['temporal'] > 85:
            reasons.append("Festival/seasonal boost")
        if metrics.geographic_score > 70:
            reasons.append("Spreading across regions")
        
        return reasons

# ================== Main Trending Service ==================

class AdvancedTrendingService:
    """
    Main service integrating all trending algorithms
    """
    
    def __init__(self, db, cache, tmdb_api_key):
        self.db = db
        self.cache = cache
        self.tmdb_api_key = tmdb_api_key
        self.unified_engine = UnifiedTrendingScoreEngine(db, cache, tmdb_api_key)
        self.session = self._create_http_session()
        self.base_url = 'https://api.themoviedb.org/3'
        
        # Store Flask app and Content model references
        self.app = None
        self.Content = None
        
        # Background thread for continuous updates
        self.update_thread = None
        self.stop_updates = False
        
    def _create_http_session(self):
        """Create HTTP session with retry logic"""
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
    
    def set_app_context(self, app, content_model):
        """Set the Flask app and Content model references"""
        self.app = app
        self.Content = content_model
    
    def start_background_updates(self):
        """Start background thread for continuous trending updates"""
        if not self.update_thread:
            self.stop_updates = False
            self.update_thread = threading.Thread(target=self._update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            logger.info("Started background trending updates")
    
    def stop_background_updates(self):
        """Stop background updates"""
        self.stop_updates = True
        if self.update_thread:
            self.update_thread.join(timeout=5)
            self.update_thread = None
            logger.info("Stopped background trending updates")
    
    def _update_loop(self):
        """Background update loop"""
        while not self.stop_updates:
            try:
                if self.app:
                    with self.app.app_context():
                        # Update trending for each language
                        for language in LANGUAGE_CONFIGS.keys():
                            self._update_language_trending(language)
                
                # Sleep for 5 minutes before next update
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in background update: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _update_language_trending(self, language: str):
        """Update trending data for specific language"""
        try:
            cache_key = f"trending:{language}:latest"
            
            # Fetch latest data from TMDb
            config = LANGUAGE_CONFIGS[language]
            params = {
                'api_key': self.tmdb_api_key,
                'with_original_language': config.tmdb_code,
                'sort_by': 'popularity.desc',
                'page': 1,
                'primary_release_date.gte': (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            }
            
            response = self.session.get(f"{self.base_url}/discover/movie", params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                trending_list = []
                
                for movie in data.get('results', [])[:20]:
                    # Calculate unified score for each movie
                    trending_content = self.unified_engine.calculate_unified_score(
                        movie['id'], language, {'tmdb_data': movie}
                    )
                    
                    # Only include if meets threshold
                    if trending_content.unified_score > config.trending_threshold:
                        trending_list.append(trending_content)
                
                # Sort by score and cache
                trending_list.sort(key=lambda x: x.unified_score, reverse=True)
                self.cache.set(cache_key, trending_list, timeout=600)  # 10 minutes
                
                logger.info(f"Updated {language} trending: {len(trending_list)} items")
                
        except Exception as e:
            logger.error(f"Error updating {language} trending: {e}")
    
    def get_trending(self, languages: List[str] = None, 
                    categories: List[str] = None,
                    limit: int = 20) -> Dict[str, List[Dict]]:
        """
        Get trending content with advanced filtering
        
        Args:
            languages: List of languages to include
            categories: List of categories (movies, tv, anime, etc.)
            limit: Maximum number of results per category
        
        Returns:
            Dictionary with trending content by category
        """
        if not languages:
            languages = ['telugu', 'english', 'hindi', 'tamil', 'malayalam']
        
        if not categories:
            categories = ['trending_movies', 'trending_tv', 'trending_anime', 
                        'rising_fast', 'popular_regional']
        
        results = {}
        
        # Handle Flask app context
        if self.app:
            with self.app.app_context():
                results = self._get_trending_with_context(languages, categories, limit)
        else:
            results = self._get_trending_with_context(languages, categories, limit)
        
        return results
    
    def _get_trending_with_context(self, languages, categories, limit):
        """Get trending with proper context"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
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
            
            # Collect results
            for category, future in futures.items():
                try:
                    results[category] = future.result(timeout=15)
                except Exception as e:
                    logger.error(f"Error getting {category}: {e}")
                    results[category] = []
        
        return results
    
    def _get_trending_movies(self, languages: List[str], limit: int) -> List[Dict]:
        """Get trending movies with unified scoring"""
        all_trending = []
        
        for language in languages:
            cache_key = f"trending:{language}:movies"
            cached = self.cache.get(cache_key)
            
            if cached:
                all_trending.extend(cached)
            else:
                # Fetch and calculate trending
                config = LANGUAGE_CONFIGS[language]
                params = {
                    'api_key': self.tmdb_api_key,
                    'with_original_language': config.tmdb_code,
                    'sort_by': 'popularity.desc',
                    'page': 1
                }
                
                response = self.session.get(f"{self.base_url}/discover/movie", 
                                          params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for movie in data.get('results', [])[:10]:
                        try:
                            # Calculate unified score
                            trending = self.unified_engine.calculate_unified_score(
                                movie['id'], language, {'tmdb_data': movie}
                            )
                            
                            if trending.unified_score > config.trending_threshold:
                                # Save to database
                                content = self._save_content(movie, 'movie', language)
                                if content:
                                    formatted = self._format_trending_content(
                                        content, trending
                                    )
                                    all_trending.append(formatted)
                        except Exception as e:
                            logger.error(f"Error processing movie {movie['id']}: {e}")
                
                # Cache results
                self.cache.set(cache_key, all_trending[-10:], timeout=1800)
        
        # Sort by unified score and return top results
        all_trending.sort(key=lambda x: x['unified_score'], reverse=True)
        return all_trending[:limit]
    
    def _get_trending_tv(self, languages: List[str], limit: int) -> List[Dict]:
        """Get trending TV shows"""
        all_trending = []
        
        # Create app context if needed
        def process_tv_shows():
            for language in languages:
                config = LANGUAGE_CONFIGS[language]
                params = {
                    'api_key': self.tmdb_api_key,
                    'with_original_language': config.tmdb_code,
                    'sort_by': 'popularity.desc',
                    'page': 1
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
                            
                            if trending.unified_score > config.trending_threshold * 0.9:
                                content = self._save_content(show, 'tv', language)
                                if content:
                                    formatted = self._format_trending_content(content, trending)
                                    all_trending.append(formatted)
                        except Exception as e:
                            logger.error(f"Error processing TV show: {e}")
            
            all_trending.sort(key=lambda x: x['unified_score'], reverse=True)
            return all_trending[:limit]
        
        # Execute with app context if available
        if self.app:
            with self.app.app_context():
                return process_tv_shows()
        else:
            return process_tv_shows()
    
    def _get_trending_anime(self, limit: int) -> List[Dict]:
        """Get trending anime"""
        cache_key = "trending:anime:latest"
        cached = self.cache.get(cache_key)
        
        if cached:
            return cached[:limit]
        
        anime_list = []
        
        try:
            # Fetch from Jikan API
            jikan_url = "https://api.jikan.moe/v4/top/anime"
            params = {'limit': limit, 'type': 'tv', 'filter': 'airing'}
            
            response = self.session.get(jikan_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                for anime in data.get('data', []):
                    # Simple scoring for anime (can be enhanced)
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
                        'category': 'trending_anime'
                    }
                    anime_list.append(formatted)
            
            # Cache results
            self.cache.set(cache_key, anime_list, timeout=3600)
            
        except Exception as e:
            logger.error(f"Error fetching anime: {e}")
        
        return anime_list[:limit]
    
    def _get_rising_fast(self, languages: List[str], limit: int) -> List[Dict]:
        """Get rapidly rising content"""
        def process_rising():
            rising = []
            
            # Get recent releases with high velocity
            for language in languages:
                config = LANGUAGE_CONFIGS[language]
                params = {
                    'api_key': self.tmdb_api_key,
                    'with_original_language': config.tmdb_code,
                    'primary_release_date.gte': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
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
                            
                            # Check velocity for "rising fast"
                            if trending.metrics.velocity > 5:
                                content = self._save_content(movie, 'movie', language)
                                if content:
                                    formatted = self._format_trending_content(content, trending)
                                    formatted['category'] = 'rising_fast'
                                    rising.append(formatted)
                        except Exception as e:
                            logger.error(f"Error processing rising content: {e}")
            
            rising.sort(key=lambda x: x.get('velocity', 0), reverse=True)
            return rising[:limit]
        
        if self.app:
            with self.app.app_context():
                return process_rising()
        else:
            return process_rising()
    
    def _get_popular_regional(self, languages: List[str], limit: int) -> List[Dict]:
        """Get regionally popular content"""
        def process_regional():
            regional = []
            
            for language in languages:
                config = LANGUAGE_CONFIGS[language]
                
                # Focus on regional content
                if language in ['telugu', 'tamil', 'malayalam', 'kannada']:
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
                                
                                # Boost regional score
                                trending.metrics.geographic_score *= 1.5
                                
                                content = self._save_content(movie, 'movie', language)
                                if content:
                                    formatted = self._format_trending_content(content, trending)
                                    formatted['category'] = 'popular_regional'
                                    formatted['region'] = config.primary_regions[0]
                                    regional.append(formatted)
                            except Exception as e:
                                logger.error(f"Error processing regional content: {e}")
            
            regional.sort(key=lambda x: x['unified_score'], reverse=True)
            return regional[:limit]
        
        if self.app:
            with self.app.app_context():
                return process_regional()
        else:
            return process_regional()
    
    def _save_content(self, tmdb_data: Dict, content_type: str, language: str):
        """Save content to database with proper context handling"""
        try:
            if not self.Content:
                logger.warning("Content model not set")
                return None
            
            # Check if we need app context
            def save_content_inner():
                # Check if exists
                existing = self.db.session.query(self.Content).filter_by(
                    tmdb_id=tmdb_data['id']
                ).first()
                
                if existing:
                    # Update existing
                    existing.popularity = tmdb_data.get('popularity', existing.popularity)
                    existing.vote_count = tmdb_data.get('vote_count', existing.vote_count)
                    existing.rating = tmdb_data.get('vote_average', existing.rating)
                    existing.is_trending = True
                    existing.updated_at = datetime.utcnow()
                    
                    self.db.session.commit()
                    return existing
                
                # Create new
                new_content = self.Content(
                    tmdb_id=tmdb_data['id'],
                    title=tmdb_data.get('title') or tmdb_data.get('name'),
                    original_title=tmdb_data.get('original_title'),
                    content_type=content_type,
                    genres=json.dumps([]),  # Fetch genres separately if needed
                    languages=json.dumps([language]),
                    release_date=None,  # Parse date if needed
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
                return new_content
            
            # Execute with proper context
            if self.app:
                with self.app.app_context():
                    return save_content_inner()
            else:
                return save_content_inner()
                
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            self.db.session.rollback()
            return None
    
    def _format_trending_content(self, content, trending: TrendingContent) -> Dict:
        """Format content for API response"""
        poster_url = None
        if content.poster_path:
            if content.poster_path.startswith('http'):
                poster_url = content.poster_path
            else:
                poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
        
        return {
            'id': content.id,
            'tmdb_id': content.tmdb_id,
            'title': content.title,
            'content_type': content.content_type,
            'genres': json.loads(content.genres or '[]'),
            'languages': json.loads(content.languages or '[]'),
            'rating': content.rating,
            'poster_path': poster_url,
            'overview': content.overview,
            'unified_score': trending.unified_score,
            'confidence': trending.confidence,
            'category': trending.category.value,
            'trending_reasons': trending.trending_reasons,
            'velocity': trending.metrics.velocity,
            'momentum': trending.metrics.momentum,
            'is_viral': trending.metrics.viral_score > 0,
            'is_trending': True
        }

# ================== Service Initialization ==================

def init_advanced_trending_service(db, cache, tmdb_api_key):
    """Initialize the advanced trending service"""
    service = AdvancedTrendingService(db, cache, tmdb_api_key)
    
    # Get Flask app and Content model from app context
    try:
        from flask import current_app
        import sys
        
        # Get the app module
        app_module = sys.modules.get('app')
        if app_module:
            # Set Flask app if available
            if hasattr(app_module, 'app'):
                flask_app = getattr(app_module, 'app')
                
                # Set Content model if available
                if hasattr(app_module, 'Content'):
                    content_model = getattr(app_module, 'Content')
                    service.set_app_context(flask_app, content_model)
                    logger.info("Set Flask app and Content model in trending service")
    except Exception as e:
        logger.warning(f"Could not set app context: {e}")
    
    # Start background updates
    service.start_background_updates()
    
    return service

def get_trending_service():
    """Get the trending service instance"""
    # This will be set from app.py
    return None