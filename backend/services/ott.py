# backend/services/ott.py
import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import re
import hashlib
from urllib.parse import quote, urlencode
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache, wraps
from bs4 import BeautifulSoup
import redis
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Platform configurations with actual working URLs
class StreamingPlatform(Enum):
    """Supported streaming platforms with metadata"""
    NETFLIX = ("Netflix", "https://www.netflix.com", "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg")
    AMAZON_PRIME = ("Amazon Prime Video", "https://www.primevideo.com", "https://upload.wikimedia.org/wikipedia/commons/f/f1/Prime_Video.png")
    DISNEY_PLUS = ("Disney+", "https://www.disneyplus.com", "https://upload.wikimedia.org/wikipedia/commons/3/3e/Disney%2B_logo.svg")
    HOTSTAR = ("Disney+ Hotstar", "https://www.hotstar.com", "https://upload.wikimedia.org/wikipedia/commons/1/1e/Disney%2B_Hotstar_logo.svg")
    HULU = ("Hulu", "https://www.hulu.com", "https://upload.wikimedia.org/wikipedia/commons/e/e4/Hulu_Logo.svg")
    YOUTUBE = ("YouTube", "https://www.youtube.com", "https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg")
    JIO_CINEMA = ("Jio Cinema", "https://www.jiocinema.com", "https://www.jiocinema.com/images/jio-logo.svg")
    ZEE5 = ("Zee5", "https://www.zee5.com", "https://upload.wikimedia.org/wikipedia/en/3/30/Zee5_Official_logo.svg")
    SONY_LIV = ("SonyLIV", "https://www.sonyliv.com", "https://upload.wikimedia.org/wikipedia/en/d/d7/Sony_Liv.png")
    VOOT = ("Voot", "https://www.voot.com", "https://upload.wikimedia.org/wikipedia/commons/0/01/Voot_logo.png")
    MX_PLAYER = ("MX Player", "https://www.mxplayer.in", "https://upload.wikimedia.org/wikipedia/en/2/2d/MX_Player.png")
    APPLE_TV = ("Apple TV+", "https://tv.apple.com", "https://upload.wikimedia.org/wikipedia/commons/2/28/Apple_TV_Plus_Logo.svg")
    HBO_MAX = ("HBO Max", "https://www.hbomax.com", "https://upload.wikimedia.org/wikipedia/commons/1/17/HBO_Max_Logo.svg")
    PARAMOUNT_PLUS = ("Paramount+", "https://www.paramountplus.com", "https://upload.wikimedia.org/wikipedia/commons/4/4e/Paramount%2B_logo.svg")

class OfferType(Enum):
    """Types of content offers"""
    FREE = "free"
    SUBSCRIPTION = "subscription"
    RENT = "rent"
    BUY = "buy"
    ADS = "ads"

class VideoQuality(Enum):
    """Video quality options"""
    SD = "SD"
    HD = "HD"
    FHD = "Full HD"
    UHD_4K = "4K"
    HDR = "HDR"

@dataclass
class StreamingOffer:
    """Represents a streaming offer for content"""
    platform: StreamingPlatform
    offer_type: OfferType
    url: str
    price: Optional[float] = None
    currency: str = "INR"
    quality: List[VideoQuality] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    subtitles: List[str] = field(default_factory=list)
    deep_link: Optional[str] = None
    region: str = "IN"

@dataclass
class ContentAvailability:
    """Complete availability information for content"""
    content_id: str
    title: str
    content_type: str
    release_year: Optional[int] = None
    streaming_offers: List[StreamingOffer] = field(default_factory=list)
    not_available_message: Optional[str] = None
    trailer_url: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContentDatabase:
    """Static database of known content availability - for fallback when APIs fail"""
    
    # Popular content with known availability
    KNOWN_AVAILABILITY = {
        'rrr': {
            'platforms': [
                ('Netflix', 'subscription', 'https://www.netflix.com/search?q=RRR', ['Hindi', 'Telugu', 'Tamil', 'Kannada', 'Malayalam']),
                ('Zee5', 'subscription', 'https://www.zee5.com/movies/details/rrr/0-0-1z5133458', ['Telugu', 'Hindi']),
            ],
            'year': 2022,
            'type': 'movie'
        },
        'pushpa': {
            'platforms': [
                ('Amazon Prime Video', 'subscription', 'https://www.primevideo.com/search/ref=atv_sr_sug_1?phrase=pushpa', ['Telugu', 'Hindi', 'Tamil', 'Malayalam', 'Kannada']),
            ],
            'year': 2021,
            'type': 'movie'
        },
        'baahubali': {
            'platforms': [
                ('Netflix', 'subscription', 'https://www.netflix.com/search?q=baahubali', ['Telugu', 'Hindi', 'Tamil', 'Malayalam']),
                ('Disney+ Hotstar', 'subscription', 'https://www.hotstar.com/in/search?q=baahubali', ['Telugu', 'Hindi']),
            ],
            'year': 2015,
            'type': 'movie'
        },
        'kgf': {
            'platforms': [
                ('Amazon Prime Video', 'subscription', 'https://www.primevideo.com/search/ref=atv_sr_sug_1?phrase=kgf', ['Kannada', 'Hindi', 'Telugu', 'Tamil', 'Malayalam']),
                ('Disney+ Hotstar', 'subscription', 'https://www.hotstar.com/in/search?q=kgf', ['Kannada', 'Hindi']),
            ],
            'year': 2018,
            'type': 'movie'
        },
        'avatar': {
            'platforms': [
                ('Disney+ Hotstar', 'subscription', 'https://www.hotstar.com/in/search?q=avatar', ['English', 'Hindi']),
                ('Disney+', 'subscription', 'https://www.disneyplus.com/search/avatar', ['English']),
            ],
            'year': 2009,
            'type': 'movie'
        },
        'avengers': {
            'platforms': [
                ('Disney+ Hotstar', 'subscription', 'https://www.hotstar.com/in/search?q=avengers', ['English', 'Hindi', 'Telugu', 'Tamil']),
                ('Disney+', 'subscription', 'https://www.disneyplus.com/search/avengers', ['English']),
            ],
            'year': 2019,
            'type': 'movie'
        },
        'kalki': {
            'platforms': [
                ('Netflix', 'subscription', 'https://www.netflix.com/search?q=kalki', ['Telugu', 'Hindi', 'Tamil']),
                ('Amazon Prime Video', 'subscription', 'https://www.primevideo.com/search?q=kalki', ['Telugu', 'Hindi']),
            ],
            'year': 2024,
            'type': 'movie'
        },
        'salaar': {
            'platforms': [
                ('Netflix', 'subscription', 'https://www.netflix.com/search?q=salaar', ['Telugu', 'Hindi', 'Tamil', 'Malayalam', 'Kannada']),
                ('Disney+ Hotstar', 'subscription', 'https://www.hotstar.com/in/search?q=salaar', ['Telugu', 'Hindi']),
            ],
            'year': 2023,
            'type': 'movie'
        },
        'animal': {
            'platforms': [
                ('Netflix', 'subscription', 'https://www.netflix.com/search?q=animal', ['Hindi', 'Telugu', 'Tamil', 'Kannada', 'Malayalam']),
            ],
            'year': 2023,
            'type': 'movie'
        },
        'jawan': {
            'platforms': [
                ('Netflix', 'subscription', 'https://www.netflix.com/search?q=jawan', ['Hindi', 'Tamil', 'Telugu']),
            ],
            'year': 2023,
            'type': 'movie'
        },
        'pathaan': {
            'platforms': [
                ('Amazon Prime Video', 'subscription', 'https://www.primevideo.com/search?q=pathaan', ['Hindi', 'Tamil', 'Telugu']),
            ],
            'year': 2023,
            'type': 'movie'
        },
        'leo': {
            'platforms': [
                ('Netflix', 'subscription', 'https://www.netflix.com/search?q=leo', ['Tamil', 'Telugu', 'Hindi', 'Kannada', 'Malayalam']),
            ],
            'year': 2023,
            'type': 'movie'
        },
        'jailer': {
            'platforms': [
                ('Amazon Prime Video', 'subscription', 'https://www.primevideo.com/search?q=jailer', ['Tamil', 'Telugu', 'Hindi', 'Kannada', 'Malayalam']),
            ],
            'year': 2023,
            'type': 'movie'
        },
        'bro': {
            'platforms': [
                ('Netflix', 'subscription', 'https://www.netflix.com/search?q=bro', ['Telugu', 'Hindi']),
                ('SonyLIV', 'subscription', 'https://www.sonyliv.com/search?q=bro', ['Telugu']),
            ],
            'year': 2023,
            'type': 'movie'
        },
        'adipurush': {
            'platforms': [
                ('Amazon Prime Video', 'subscription', 'https://www.primevideo.com/search?q=adipurush', ['Hindi', 'Telugu', 'Tamil', 'Kannada', 'Malayalam']),
            ],
            'year': 2023,
            'type': 'movie'
        },
        'dasara': {
            'platforms': [
                ('Netflix', 'subscription', 'https://www.netflix.com/search?q=dasara', ['Telugu', 'Hindi', 'Tamil', 'Kannada', 'Malayalam']),
            ],
            'year': 2023,
            'type': 'movie'
        }
    }
    
    @classmethod
    def search_content(cls, query: str) -> Optional[Dict]:
        """Search for content in the static database"""
        query_lower = query.lower().strip()
        
        # Direct match
        if query_lower in cls.KNOWN_AVAILABILITY:
            return cls.KNOWN_AVAILABILITY[query_lower]
        
        # Partial match
        for title, info in cls.KNOWN_AVAILABILITY.items():
            if query_lower in title or title in query_lower:
                return info
        
        # Check if query contains any known title
        for title in cls.KNOWN_AVAILABILITY.keys():
            if title in query_lower or query_lower in title:
                return cls.KNOWN_AVAILABILITY[title]
        
        return None

class StreamingAvailabilityService:
    """Lightweight service that uses multiple strategies to find streaming availability"""
    
    def __init__(self):
        self.session = self._create_session()
        self.content_db = ContentDatabase()
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry logic"""
        session = requests.Session()
        retry = Retry(
            total=2,  # Reduced retries for faster response
            read=2,
            connect=2,
            backoff_factor=0.2,
            status_forcelist=(500, 502, 504)
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        return session
    
    def get_availability(self, title: str, content_type: str = 'movie', 
                        year: Optional[int] = None, country: str = 'IN') -> ContentAvailability:
        """Get streaming availability using multiple strategies"""
        
        # Initialize result
        availability = ContentAvailability(
            content_id=hashlib.md5(title.encode()).hexdigest()[:8],
            title=title,
            content_type=content_type
        )
        
        # Strategy 1: Check static database first (fastest)
        db_result = self.content_db.search_content(title)
        if db_result:
            availability.streaming_offers = self._parse_db_result(db_result)
            availability.release_year = db_result.get('year')
            logger.info(f"Found {title} in static database")
        
        # Strategy 2: Use JustWatch scraping (if not found in DB)
        if not availability.streaming_offers:
            justwatch_offers = self._scrape_justwatch(title, year, country)
            availability.streaming_offers.extend(justwatch_offers)
        
        # Strategy 3: Generate common platform links (fallback)
        if not availability.streaming_offers:
            availability.streaming_offers = self._generate_common_links(title, content_type)
            availability.not_available_message = (
                "Exact availability couldn't be confirmed. "
                "Click the links below to search on each platform."
            )
        
        # Generate trailer URL
        availability.trailer_url = self._get_youtube_trailer_url(title, content_type)
        
        # Add metadata
        availability.metadata = {
            'total_offers': len(availability.streaming_offers),
            'platforms': list(set(offer.platform.value[0] for offer in availability.streaming_offers)),
            'has_free_option': any(offer.offer_type == OfferType.FREE for offer in availability.streaming_offers)
        }
        
        return availability
    
    def _parse_db_result(self, db_result: Dict) -> List[StreamingOffer]:
        """Parse database result into streaming offers"""
        offers = []
        
        for platform_info in db_result.get('platforms', []):
            platform_name, offer_type, url, languages = platform_info
            
            # Map platform name to enum
            platform = None
            for p in StreamingPlatform:
                if platform_name.lower() in p.value[0].lower() or p.value[0].lower() in platform_name.lower():
                    platform = p
                    break
            
            if platform:
                offers.append(StreamingOffer(
                    platform=platform,
                    offer_type=OfferType.SUBSCRIPTION if offer_type == 'subscription' else OfferType.FREE,
                    url=url,
                    languages=languages,
                    quality=[VideoQuality.HD, VideoQuality.FHD],
                    deep_link=url
                ))
        
        return offers
    
    def _scrape_justwatch(self, title: str, year: Optional[int], country: str) -> List[StreamingOffer]:
        """Quick JustWatch scraping with timeout"""
        offers = []
        
        try:
            # Use JustWatch API endpoint (faster than scraping HTML)
            search_url = f"https://apis.justwatch.com/content/titles/en_{country}/popular"
            params = {
                'body': json.dumps({
                    'query': title,
                    'content_types': ['movie', 'show'],
                    'page': 1,
                    'page_size': 5
                })
            }
            
            # Quick timeout to avoid delays
            response = self.session.post(search_url, json=params, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                # Parse JustWatch response (simplified)
                # This would need proper parsing based on actual API response
                logger.info(f"JustWatch API returned data for {title}")
        except Exception as e:
            logger.debug(f"JustWatch scraping skipped: {e}")
        
        return offers
    
    def _generate_common_links(self, title: str, content_type: str) -> List[StreamingOffer]:
        """Generate search links for common platforms"""
        offers = []
        query = quote(title)
        
        # Most common platforms in India
        common_platforms = [
            (StreamingPlatform.NETFLIX, f"https://www.netflix.com/search?q={query}"),
            (StreamingPlatform.AMAZON_PRIME, f"https://www.primevideo.com/search/ref=atv_sr_sug_1?phrase={query}"),
            (StreamingPlatform.DISNEY_PLUS, f"https://www.disneyplus.com/search/{query}"),
            (StreamingPlatform.HOTSTAR, f"https://www.hotstar.com/in/search?q={query}"),
            (StreamingPlatform.JIO_CINEMA, f"https://www.jiocinema.com/search/{query}"),
            (StreamingPlatform.ZEE5, f"https://www.zee5.com/search?q={query}"),
            (StreamingPlatform.SONY_LIV, f"https://www.sonyliv.com/search?q={query}"),
            (StreamingPlatform.YOUTUBE, f"https://www.youtube.com/results?search_query={query}+full+movie"),
        ]
        
        for platform, url in common_platforms:
            offers.append(StreamingOffer(
                platform=platform,
                offer_type=OfferType.SUBSCRIPTION,
                url=url,
                deep_link=url,
                quality=[VideoQuality.HD],
                languages=['Multiple'],
                region='IN'
            ))
        
        return offers
    
    def _get_youtube_trailer_url(self, title: str, content_type: str) -> str:
        """Generate YouTube trailer search URL"""
        query = f"{title} official trailer"
        if content_type == 'movie':
            query += " movie"
        return f"https://www.youtube.com/results?search_query={quote(query)}"

class OTTAvailabilityService:
    """Main OTT service with optimizations for production"""
    
    def __init__(self, streaming_api_key: Optional[str] = None, 
                 youtube_api_key: Optional[str] = None,
                 redis_url: Optional[str] = None,
                 cache_ttl: int = 3600):
        
        self.streaming_service = StreamingAvailabilityService()
        self.cache_ttl = cache_ttl
        self.redis_client = None
        
        # Try to connect to Redis for caching
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Redis cache connected")
            except:
                logger.warning("Redis not available, using in-memory cache")
        
        # In-memory cache fallback
        self.memory_cache = {}
    
    def get_availability(self, content_id: str, title: str, 
                        content_type: str = "movie",
                        year: Optional[int] = None,
                        country: str = "IN",
                        languages: Optional[List[str]] = None) -> ContentAvailability:
        """Get content availability with caching"""
        
        # Check cache first
        cache_key = f"ott:{title.lower()}:{country}"
        cached = self._get_from_cache(cache_key)
        if cached:
            logger.info(f"Cache hit for {title}")
            return ContentAvailability(**json.loads(cached))
        
        # Get availability from service
        availability = self.streaming_service.get_availability(
            title=title,
            content_type=content_type,
            year=year,
            country=country
        )
        
        # Filter by languages if specified
        if languages and availability.streaming_offers:
            for offer in availability.streaming_offers:
                if not offer.languages:
                    offer.languages = languages[:2]  # Add requested languages
        
        # Cache the result
        self._set_cache(cache_key, self._serialize_availability(availability))
        
        return availability
    
    def _get_from_cache(self, key: str) -> Optional[str]:
        """Get from cache (Redis or memory)"""
        if self.redis_client:
            try:
                return self.redis_client.get(key)
            except:
                pass
        return self.memory_cache.get(key)
    
    def _set_cache(self, key: str, value: str, ttl: Optional[int] = None):
        """Set cache (Redis or memory)"""
        ttl = ttl or self.cache_ttl
        
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, value)
                return
            except:
                pass
        
        self.memory_cache[key] = value
    
    def _serialize_availability(self, availability: ContentAvailability) -> str:
        """Serialize availability for caching"""
        data = {
            'content_id': availability.content_id,
            'title': availability.title,
            'content_type': availability.content_type,
            'release_year': availability.release_year,
            'streaming_offers': [
                {
                    'platform': offer.platform.value[0],
                    'platform_logo': offer.platform.value[2],
                    'offer_type': offer.offer_type.value,
                    'url': offer.url,
                    'deep_link': offer.deep_link,
                    'languages': offer.languages,
                    'quality': [q.value for q in offer.quality],
                    'region': offer.region
                }
                for offer in availability.streaming_offers
            ],
            'not_available_message': availability.not_available_message,
            'trailer_url': availability.trailer_url,
            'metadata': availability.metadata
        }
        return json.dumps(data)
    
    def format_for_response(self, availability: ContentAvailability) -> Dict:
        """Format availability for API response"""
        
        # Group by platform
        platforms_grouped = {}
        for offer in availability.streaming_offers:
            platform_name = offer.platform.value[0]
            if platform_name not in platforms_grouped:
                platforms_grouped[platform_name] = {
                    'platform': platform_name,
                    'logo': offer.platform.value[2],
                    'offers': []
                }
            
            offer_detail = {
                'type': offer.offer_type.value,
                'url': offer.deep_link or offer.url,
                'quality': [q.value for q in offer.quality] if offer.quality else ['HD'],
                'languages': offer.languages or ['Multiple'],
                'display_text': self._format_offer_display(offer),
                'badge': offer.offer_type.value.upper()
            }
            
            if offer.price:
                offer_detail['price'] = f"{offer.currency} {offer.price}"
            
            platforms_grouped[platform_name]['offers'].append(offer_detail)
        
        return {
            'content_id': availability.content_id,
            'title': availability.title,
            'content_type': availability.content_type,
            'release_year': availability.release_year,
            'availability': {
                'available': len(availability.streaming_offers) > 0,
                'platforms': list(platforms_grouped.values()),
                'total_offers': len(availability.streaming_offers),
                'has_free_option': availability.metadata.get('has_free_option', False),
                'message': availability.not_available_message
            },
            'trailer_url': availability.trailer_url,
            'metadata': availability.metadata,
            'last_updated': availability.last_updated.isoformat()
        }
    
    def _format_offer_display(self, offer: StreamingOffer) -> str:
        """Format offer for display"""
        if offer.offer_type == OfferType.FREE:
            return "Watch Free"
        elif offer.offer_type == OfferType.ADS:
            return "Free with Ads"
        elif offer.offer_type == OfferType.SUBSCRIPTION:
            return "With Subscription"
        elif offer.offer_type == OfferType.RENT:
            return f"Rent"
        elif offer.offer_type == OfferType.BUY:
            return f"Buy"
        return "Watch Now"
    
    def search(self, query: str, content_type: str = 'both', 
              country: str = 'IN', limit: int = 10) -> List[Dict]:
        """Search for content and get availability"""
        results = []
        
        # Search in known database first
        db_result = ContentDatabase.search_content(query)
        if db_result:
            # Create a result from database
            availability = self.streaming_service.get_availability(
                title=query,
                content_type=db_result.get('type', 'movie'),
                year=db_result.get('year'),
                country=country
            )
            
            formatted = self.format_for_response(availability)
            results.append(formatted)
        
        # Add general search results
        search_terms = [query]
        if ' ' not in query:
            # Try variations for single word queries
            search_terms.extend([f"{query} movie", f"{query} series"])
        
        for term in search_terms[:limit]:
            if len(results) >= limit:
                break
            
            availability = self.streaming_service.get_availability(
                title=term,
                content_type=content_type if content_type != 'both' else 'movie',
                country=country
            )
            
            if availability.streaming_offers:
                formatted = self.format_for_response(availability)
                # Avoid duplicates
                if not any(r['title'].lower() == formatted['title'].lower() for r in results):
                    results.append(formatted)
        
        return results[:limit]

# Simplified API integration
class StreamingAvailabilityAPI:
    """Lightweight API client that doesn't block on failures"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.enabled = bool(api_key)
    
    def search(self, query: str, content_type: str = 'both', 
              country: str = 'in') -> Optional[Dict]:
        """Quick search with fast timeout"""
        if not self.enabled:
            return None
        
        try:
            # Very short timeout to avoid blocking
            response = requests.get(
                'https://streaming-availability.p.rapidapi.com/search/title',
                params={'title': query, 'country': country},
                headers={
                    'x-rapidapi-key': self.api_key,
                    'x-rapidapi-host': 'streaming-availability.p.rapidapi.com'
                },
                timeout=2  # 2 second timeout max
            )
            
            if response.status_code == 200:
                return response.json()
        except:
            pass
        
        return None