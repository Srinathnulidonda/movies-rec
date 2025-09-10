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

# Platform configurations
class StreamingPlatform(Enum):
    """Supported streaming platforms with metadata"""
    NETFLIX = ("Netflix", "https://www.netflix.com", "https://i.imgur.com/netflix-logo.png")
    AMAZON_PRIME = ("Amazon Prime Video", "https://www.primevideo.com", "https://i.imgur.com/prime-logo.png")
    DISNEY_PLUS = ("Disney+", "https://www.disneyplus.com", "https://i.imgur.com/disney-logo.png")
    HOTSTAR = ("Disney+ Hotstar", "https://www.hotstar.com", "https://i.imgur.com/hotstar-logo.png")
    HULU = ("Hulu", "https://www.hulu.com", "https://i.imgur.com/hulu-logo.png")
    YOUTUBE = ("YouTube", "https://www.youtube.com", "https://i.imgur.com/youtube-logo.png")
    ETV_WIN = ("ETV Win", "https://www.etvwin.com", "https://i.imgur.com/etv-logo.png")
    AHA = ("Aha", "https://www.aha.video", "https://i.imgur.com/aha-logo.png")
    JIO_CINEMA = ("Jio Cinema", "https://www.jiocinema.com", "https://i.imgur.com/jio-logo.png")
    MX_PLAYER = ("MX Player", "https://www.mxplayer.in", "https://i.imgur.com/mx-logo.png")
    ZEE5 = ("Zee5", "https://www.zee5.com", "https://i.imgur.com/zee5-logo.png")
    SONY_LIV = ("SonyLIV", "https://www.sonyliv.com", "https://i.imgur.com/sony-logo.png")
    VOOT = ("Voot", "https://www.voot.com", "https://i.imgur.com/voot-logo.png")
    ALT_BALAJI = ("ALTBalaji", "https://www.altbalaji.com", "https://i.imgur.com/alt-logo.png")
    APPLE_TV = ("Apple TV+", "https://tv.apple.com", "https://i.imgur.com/appletv-logo.png")
    HBO_MAX = ("HBO Max", "https://www.hbomax.com", "https://i.imgur.com/hbo-logo.png")
    PARAMOUNT_PLUS = ("Paramount+", "https://www.paramountplus.com", "https://i.imgur.com/paramount-logo.png")
    PEACOCK = ("Peacock", "https://www.peacocktv.com", "https://i.imgur.com/peacock-logo.png")
    CRUNCHYROLL = ("Crunchyroll", "https://www.crunchyroll.com", "https://i.imgur.com/crunchyroll-logo.png")
    FUNIMATION = ("Funimation", "https://www.funimation.com", "https://i.imgur.com/funimation-logo.png")

class OfferType(Enum):
    """Types of content offers"""
    FREE = "free"
    SUBSCRIPTION = "subscription"
    RENT = "rent"
    BUY = "buy"
    ADS = "ads"  # Free with ads

class VideoQuality(Enum):
    """Video quality options"""
    SD = "SD"
    HD = "HD"
    FHD = "Full HD"
    UHD_4K = "4K"
    UHD_8K = "8K"
    HDR = "HDR"
    DOLBY_VISION = "Dolby Vision"

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
    audio_formats: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    region: str = "IN"
    deep_link: Optional[str] = None
    package_info: Optional[str] = None  # e.g., "Premium Plan", "Basic Plan"

@dataclass
class ContentAvailability:
    """Complete availability information for content"""
    content_id: str
    title: str
    content_type: str  # movie, series, anime
    release_year: Optional[int] = None
    runtime: Optional[int] = None
    streaming_offers: List[StreamingOffer] = field(default_factory=list)
    not_available_message: Optional[str] = None
    trailer_url: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

class CacheManager:
    """Redis-based cache manager for OTT data"""
    
    def __init__(self, redis_url: Optional[str] = None, default_ttl: int = 3600):
        self.redis_client = None
        self.default_ttl = default_ttl
        
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
                self.redis_client = None
        
        # Fallback to in-memory cache
        self.memory_cache = {}
        self.cache_timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            except Exception as e:
                logger.debug(f"Redis get error: {e}")
        
        # Fallback to memory cache
        if key in self.memory_cache:
            timestamp = self.cache_timestamps.get(key)
            if timestamp and (datetime.utcnow() - timestamp).seconds < self.default_ttl:
                return self.memory_cache[key]
            else:
                del self.memory_cache[key]
                del self.cache_timestamps[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value))
                return True
            except Exception as e:
                logger.debug(f"Redis set error: {e}")
        
        # Fallback to memory cache
        self.memory_cache[key] = value
        self.cache_timestamps[key] = datetime.utcnow()
        return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.debug(f"Redis delete error: {e}")
        
        if key in self.memory_cache:
            del self.memory_cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]
        return True

class StreamingAvailabilityAPI:
    """Primary API client for streaming availability"""
    
    BASE_URL = "https://streaming-availability.p.rapidapi.com"
    
    def __init__(self, api_key: str, api_host: str = "streaming-availability.p.rapidapi.com"):
        self.api_key = api_key
        self.api_host = api_host
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry logic"""
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
        session.headers.update({
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': self.api_host
        })
        return session
    
    def get_by_id(self, content_type: str, content_id: str, 
                  country: str = "in") -> Optional[Dict]:
        """Get streaming availability by content ID"""
        try:
            url = f"{self.BASE_URL}/shows/{content_type}/{content_id}"
            params = {"country": country}
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.info(f"Content {content_id} not found in streaming API")
            else:
                logger.warning(f"API returned {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Streaming API error: {e}")
        return None
    
    def search(self, query: str, content_type: str = "both", 
               country: str = "in") -> Optional[Dict]:
        """Search for content availability"""
        try:
            url = f"{self.BASE_URL}/search/title"
            params = {
                "title": query,
                "country": country,
                "type": content_type,
                "output_language": "en"
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Search API error: {e}")
        return None

class WebScraperFallback:
    """Fallback web scraping for streaming availability"""
    
    def __init__(self):
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create session with headers to avoid detection"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        return session
    
    def scrape_justwatch(self, title: str, year: Optional[int] = None,
                        country: str = "IN") -> List[StreamingOffer]:
        """Scrape JustWatch for availability"""
        offers = []
        try:
            # Build search URL
            search_query = quote(title)
            url = f"https://www.justwatch.com/{country.lower()}/search?q={search_query}"
            
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Parse streaming offers (simplified example)
                offer_elements = soup.find_all('div', class_='price-comparison__grid__row')
                
                for element in offer_elements:
                    platform_name = element.find('img', class_='provider-logo')
                    if platform_name:
                        platform_name = platform_name.get('alt', '').strip()
                        
                        # Map to our platform enum
                        platform = self._map_platform(platform_name)
                        if platform:
                            # Extract offer details
                            offer_type = self._extract_offer_type(element)
                            link = element.find('a', class_='price-comparison__grid__row__link')
                            
                            if link:
                                offers.append(StreamingOffer(
                                    platform=platform,
                                    offer_type=offer_type,
                                    url=f"https://www.justwatch.com{link.get('href', '')}",
                                    region=country
                                ))
        except Exception as e:
            logger.error(f"JustWatch scraping error: {e}")
        
        return offers
    
    def scrape_google_knowledge_graph(self, title: str, year: Optional[int] = None) -> List[StreamingOffer]:
        """Scrape Google's knowledge graph for watch options"""
        offers = []
        try:
            query = f"{title} {year if year else ''} watch online"
            url = f"https://www.google.com/search?q={quote(query)}"
            
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                # Parse Google's watch now widget
                # This is simplified - actual implementation would need more robust parsing
                if "Watch now" in response.text:
                    # Extract platform links from Google's watch widget
                    soup = BeautifulSoup(response.content, 'html.parser')
                    watch_widget = soup.find('div', {'data-attrid': 'kc:/film/film:watch'})
                    
                    if watch_widget:
                        platforms = watch_widget.find_all('a', class_='watch-link')
                        for platform in platforms:
                            # Parse platform details
                            pass
        except Exception as e:
            logger.error(f"Google scraping error: {e}")
        
        return offers
    
    def _map_platform(self, platform_name: str) -> Optional[StreamingPlatform]:
        """Map platform name to enum"""
        platform_map = {
            'netflix': StreamingPlatform.NETFLIX,
            'amazon prime': StreamingPlatform.AMAZON_PRIME,
            'prime video': StreamingPlatform.AMAZON_PRIME,
            'disney+': StreamingPlatform.DISNEY_PLUS,
            'disney plus': StreamingPlatform.DISNEY_PLUS,
            'hotstar': StreamingPlatform.HOTSTAR,
            'hulu': StreamingPlatform.HULU,
            'youtube': StreamingPlatform.YOUTUBE,
            'etv win': StreamingPlatform.ETV_WIN,
            'aha': StreamingPlatform.AHA,
            'jio cinema': StreamingPlatform.JIO_CINEMA,
            'mx player': StreamingPlatform.MX_PLAYER,
            'zee5': StreamingPlatform.ZEE5,
            'sony liv': StreamingPlatform.SONY_LIV,
            'voot': StreamingPlatform.VOOT,
            'alt balaji': StreamingPlatform.ALT_BALAJI,
            'apple tv': StreamingPlatform.APPLE_TV,
            'hbo max': StreamingPlatform.HBO_MAX,
            'paramount+': StreamingPlatform.PARAMOUNT_PLUS,
            'peacock': StreamingPlatform.PEACOCK,
            'crunchyroll': StreamingPlatform.CRUNCHYROLL,
            'funimation': StreamingPlatform.FUNIMATION
        }
        
        platform_lower = platform_name.lower()
        for key, value in platform_map.items():
            if key in platform_lower:
                return value
        return None
    
    def _extract_offer_type(self, element) -> OfferType:
        """Extract offer type from element"""
        text = element.get_text().lower()
        if 'free' in text:
            return OfferType.FREE if 'ads' not in text else OfferType.ADS
        elif 'rent' in text:
            return OfferType.RENT
        elif 'buy' in text:
            return OfferType.BUY
        else:
            return OfferType.SUBSCRIPTION

class YouTubeTrailerFetcher:
    """Fetch YouTube trailers as fallback"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
    
    def get_trailer(self, title: str, year: Optional[int] = None,
                    language: str = "en") -> Optional[str]:
        """Get YouTube trailer URL"""
        if not self.api_key:
            return self._scrape_trailer(title, year)
        
        try:
            # Build search query
            query = f"{title} {year if year else ''} official trailer {language}"
            
            url = f"{self.base_url}/search"
            params = {
                'key': self.api_key,
                'q': query,
                'part': 'snippet',
                'type': 'video',
                'maxResults': 3,
                'order': 'relevance'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('items'):
                    # Return first relevant trailer
                    video_id = data['items'][0]['id']['videoId']
                    return f"https://www.youtube.com/watch?v={video_id}"
        except Exception as e:
            logger.error(f"YouTube API error: {e}")
        
        return self._scrape_trailer(title, year)
    
    def _scrape_trailer(self, title: str, year: Optional[int] = None) -> Optional[str]:
        """Scrape YouTube for trailer without API"""
        try:
            query = f"{title} {year if year else ''} official trailer"
            search_url = f"https://www.youtube.com/results?search_query={quote(query)}"
            
            response = requests.get(search_url, timeout=10)
            if response.status_code == 200:
                # Extract video ID from response
                match = re.search(r'"videoId":"([^"]+)"', response.text)
                if match:
                    video_id = match.group(1)
                    return f"https://www.youtube.com/watch?v={video_id}"
        except Exception as e:
            logger.error(f"YouTube scraping error: {e}")
        return None

class DeepLinkGenerator:
    """Generate deep links for streaming platforms"""
    
    @staticmethod
    def generate(platform: StreamingPlatform, content_id: str,
                title: str, content_type: str = "movie") -> str:
        """Generate platform-specific deep link"""
        
        # Clean title for URL
        clean_title = re.sub(r'[^\w\s-]', '', title.lower())
        clean_title = re.sub(r'[-\s]+', '-', clean_title)
        
        deep_links = {
            StreamingPlatform.NETFLIX: f"https://www.netflix.com/search?q={quote(title)}",
            StreamingPlatform.AMAZON_PRIME: f"https://www.primevideo.com/search/ref=atv_sr_sug_1?phrase={quote(title)}",
            StreamingPlatform.DISNEY_PLUS: f"https://www.disneyplus.com/search/{quote(title)}",
            StreamingPlatform.HOTSTAR: f"https://www.hotstar.com/in/search?q={quote(title)}",
            StreamingPlatform.HULU: f"https://www.hulu.com/search?q={quote(title)}",
            StreamingPlatform.YOUTUBE: f"https://www.youtube.com/results?search_query={quote(title)}+full+movie",
            StreamingPlatform.ETV_WIN: f"https://www.etvwin.com/search?q={quote(title)}",
            StreamingPlatform.AHA: f"https://www.aha.video/search/{quote(title)}",
            StreamingPlatform.JIO_CINEMA: f"https://www.jiocinema.com/search/{quote(title)}",
            StreamingPlatform.MX_PLAYER: f"https://www.mxplayer.in/search?query={quote(title)}",
            StreamingPlatform.ZEE5: f"https://www.zee5.com/search?q={quote(title)}",
            StreamingPlatform.SONY_LIV: f"https://www.sonyliv.com/search?q={quote(title)}",
            StreamingPlatform.VOOT: f"https://www.voot.com/search/{quote(title)}",
            StreamingPlatform.ALT_BALAJI: f"https://www.altbalaji.com/search/{quote(title)}",
            StreamingPlatform.APPLE_TV: f"https://tv.apple.com/search?term={quote(title)}",
            StreamingPlatform.HBO_MAX: f"https://play.hbomax.com/search?q={quote(title)}",
            StreamingPlatform.PARAMOUNT_PLUS: f"https://www.paramountplus.com/search/?q={quote(title)}",
            StreamingPlatform.PEACOCK: f"https://www.peacocktv.com/search?q={quote(title)}",
            StreamingPlatform.CRUNCHYROLL: f"https://www.crunchyroll.com/search?q={quote(title)}",
            StreamingPlatform.FUNIMATION: f"https://www.funimation.com/search/?q={quote(title)}"
        }
        
        return deep_links.get(platform, f"https://www.google.com/search?q={quote(title)}+watch+online")

class OTTAvailabilityService:
    """Main OTT Availability Service with all features"""
    
    def __init__(self, 
                 streaming_api_key: str = "6212e018b9mshb44a2716d211c51p1c493ejsn73408baa28be",
                 youtube_api_key: Optional[str] = None,
                 redis_url: Optional[str] = None,
                 cache_ttl: int = 3600):
        
        self.streaming_api = StreamingAvailabilityAPI(streaming_api_key)
        self.scraper = WebScraperFallback()
        self.youtube_fetcher = YouTubeTrailerFetcher(youtube_api_key)
        self.cache = CacheManager(redis_url, cache_ttl)
        self.deep_link_gen = DeepLinkGenerator()
        
        # Multi-language support
        self.supported_languages = [
            'English', 'Hindi', 'Telugu', 'Tamil', 'Kannada', 
            'Malayalam', 'Bengali', 'Marathi', 'Gujarati', 'Punjabi'
        ]
        
        # Regional platform preferences
        self.regional_preferences = {
            'IN': [StreamingPlatform.HOTSTAR, StreamingPlatform.AMAZON_PRIME, 
                   StreamingPlatform.NETFLIX, StreamingPlatform.JIO_CINEMA,
                   StreamingPlatform.AHA, StreamingPlatform.ETV_WIN],
            'US': [StreamingPlatform.NETFLIX, StreamingPlatform.HULU, 
                   StreamingPlatform.AMAZON_PRIME, StreamingPlatform.DISNEY_PLUS,
                   StreamingPlatform.HBO_MAX, StreamingPlatform.PEACOCK]
        }
    
    async def get_availability_async(self, 
                                    content_id: str,
                                    title: str,
                                    content_type: str = "movie",
                                    year: Optional[int] = None,
                                    country: str = "IN",
                                    languages: Optional[List[str]] = None) -> ContentAvailability:
        """Async method to get content availability"""
        
        # Check cache first
        cache_key = f"ott:{content_id}:{country}:{':'.join(languages or [])}"
        cached = self.cache.get(cache_key)
        if cached:
            return ContentAvailability(**cached)
        
        # Initialize result
        availability = ContentAvailability(
            content_id=content_id,
            title=title,
            content_type=content_type,
            release_year=year
        )
        
        # Fetch from multiple sources concurrently
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            # Task 1: Streaming API
            tasks.append(self._fetch_streaming_api_async(session, content_id, content_type, country))
            
            # Task 2: Web scraping fallback
            tasks.append(self._fetch_scraper_async(title, year, country))
            
            # Task 3: YouTube trailer
            tasks.append(self._fetch_trailer_async(title, year, languages))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process API results
            if results[0] and not isinstance(results[0], Exception):
                availability.streaming_offers.extend(self._parse_api_response(results[0], languages))
            
            # Process scraping results
            if results[1] and not isinstance(results[1], Exception):
                availability.streaming_offers.extend(results[1])
            
            # Add trailer
            if results[2] and not isinstance(results[2], Exception):
                availability.trailer_url = results[2]
        
        # Generate deep links
        for offer in availability.streaming_offers:
            if not offer.deep_link:
                offer.deep_link = self.deep_link_gen.generate(
                    offer.platform, content_id, title, content_type
                )
        
        # Sort offers by preference
        availability.streaming_offers = self._sort_offers(
            availability.streaming_offers, country
        )
        
        # Set not available message if no offers
        if not availability.streaming_offers:
            availability.not_available_message = (
                "Currently not available for streaming. Check back later or watch the trailer."
            )
        
        # Cache the result
        self.cache.set(cache_key, availability.__dict__, ttl=self.cache.default_ttl)
        
        return availability
    
    def get_availability(self, 
                         content_id: str,
                         title: str,
                         content_type: str = "movie",
                         year: Optional[int] = None,
                         country: str = "IN",
                         languages: Optional[List[str]] = None) -> ContentAvailability:
        """Synchronous method to get content availability"""
        
        # Check cache first
        cache_key = f"ott:{content_id}:{country}:{':'.join(languages or [])}"
        cached = self.cache.get(cache_key)
        if cached:
            return ContentAvailability(**cached)
        
        # Initialize result
        availability = ContentAvailability(
            content_id=content_id,
            title=title,
            content_type=content_type,
            release_year=year
        )
        
        # Use ThreadPoolExecutor for concurrent fetching
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks
            api_future = executor.submit(
                self.streaming_api.get_by_id, content_type, content_id, country.lower()
            )
            scraper_future = executor.submit(
                self.scraper.scrape_justwatch, title, year, country
            )
            trailer_future = executor.submit(
                self.youtube_fetcher.get_trailer, title, year
            )
            
            # Get results
            try:
                api_result = api_future.result(timeout=10)
                if api_result:
                    availability.streaming_offers.extend(
                        self._parse_api_response(api_result, languages)
                    )
            except Exception as e:
                logger.error(f"API fetch error: {e}")
            
            try:
                scraper_result = scraper_future.result(timeout=10)
                if scraper_result:
                    availability.streaming_offers.extend(scraper_result)
            except Exception as e:
                logger.error(f"Scraper error: {e}")
            
            try:
                trailer_result = trailer_future.result(timeout=10)
                if trailer_result:
                    availability.trailer_url = trailer_result
            except Exception as e:
                logger.error(f"Trailer fetch error: {e}")
        
        # Generate deep links
        for offer in availability.streaming_offers:
            if not offer.deep_link:
                offer.deep_link = self.deep_link_gen.generate(
                    offer.platform, content_id, title, content_type
                )
        
        # Remove duplicate offers
        availability.streaming_offers = self._deduplicate_offers(availability.streaming_offers)
        
        # Sort offers by preference
        availability.streaming_offers = self._sort_offers(
            availability.streaming_offers, country
        )
        
        # Set not available message if no offers
        if not availability.streaming_offers:
            availability.not_available_message = (
                "Currently not available for streaming. Check back later or watch the trailer."
            )
        
        # Add metadata
        availability.metadata = {
            'total_offers': len(availability.streaming_offers),
            'platforms': list(set(offer.platform.value[0] for offer in availability.streaming_offers)),
            'has_free_option': any(offer.offer_type == OfferType.FREE for offer in availability.streaming_offers),
            'languages_available': list(set(
                lang for offer in availability.streaming_offers 
                for lang in offer.languages
            ))
        }
        
        # Cache the result
        cache_data = {
            'content_id': availability.content_id,
            'title': availability.title,
            'content_type': availability.content_type,
            'release_year': availability.release_year,
            'runtime': availability.runtime,
            'streaming_offers': [self._offer_to_dict(offer) for offer in availability.streaming_offers],
            'not_available_message': availability.not_available_message,
            'trailer_url': availability.trailer_url,
            'last_updated': availability.last_updated.isoformat(),
            'metadata': availability.metadata
        }
        self.cache.set(cache_key, cache_data, ttl=self.cache.default_ttl)
        
        return availability
    
    def _parse_api_response(self, response: Dict, languages: Optional[List[str]] = None) -> List[StreamingOffer]:
        """Parse streaming API response into offers"""
        offers = []
        
        try:
            # Parse streaming options from response
            streaming_info = response.get('streamingInfo', {})
            
            for country_code, country_info in streaming_info.items():
                for service in country_info:
                    platform_name = service.get('service', '').upper()
                    
                    # Map to our platform enum
                    platform = None
                    for p in StreamingPlatform:
                        if platform_name in p.value[0].upper():
                            platform = p
                            break
                    
                    if not platform:
                        continue
                    
                    # Determine offer type
                    streaming_type = service.get('streamingType', 'subscription')
                    offer_type_map = {
                        'subscription': OfferType.SUBSCRIPTION,
                        'rent': OfferType.RENT,
                        'buy': OfferType.BUY,
                        'free': OfferType.FREE,
                        'ads': OfferType.ADS
                    }
                    offer_type = offer_type_map.get(streaming_type, OfferType.SUBSCRIPTION)
                    
                    # Extract quality
                    quality = []
                    if service.get('quality'):
                        quality_map = {
                            'sd': VideoQuality.SD,
                            'hd': VideoQuality.HD,
                            'uhd': VideoQuality.UHD_4K,
                            '4k': VideoQuality.UHD_4K,
                            'hdr': VideoQuality.HDR
                        }
                        for q in service.get('quality', '').lower().split(','):
                            if q in quality_map:
                                quality.append(quality_map[q])
                    
                    # Extract audio/subtitle languages
                    audio_langs = service.get('audios', [])
                    subtitle_langs = service.get('subtitles', [])
                    
                    # Filter by requested languages if specified
                    if languages:
                        matching_langs = [lang for lang in audio_langs if lang in languages]
                        if not matching_langs and audio_langs:
                            matching_langs = audio_langs[:1]  # Default to first available
                    else:
                        matching_langs = audio_langs
                    
                    # Create offer
                    offer = StreamingOffer(
                        platform=platform,
                        offer_type=offer_type,
                        url=service.get('link', ''),
                        price=service.get('price', {}).get('amount'),
                        currency=service.get('price', {}).get('currency', 'INR'),
                        quality=quality,
                        languages=matching_langs,
                        subtitles=subtitle_langs,
                        region=country_code.upper()
                    )
                    
                    offers.append(offer)
        
        except Exception as e:
            logger.error(f"Error parsing API response: {e}")
        
        return offers
    
    def _deduplicate_offers(self, offers: List[StreamingOffer]) -> List[StreamingOffer]:
        """Remove duplicate offers, keeping best quality/price"""
        seen = {}
        
        for offer in offers:
            key = (offer.platform, offer.offer_type)
            
            if key not in seen:
                seen[key] = offer
            else:
                # Keep better offer (more languages, better quality, lower price)
                existing = seen[key]
                
                # Compare and keep better offer
                if (len(offer.languages) > len(existing.languages) or
                    len(offer.quality) > len(existing.quality) or
                    (offer.price and existing.price and offer.price < existing.price)):
                    seen[key] = offer
        
        return list(seen.values())
    
    def _sort_offers(self, offers: List[StreamingOffer], country: str) -> List[StreamingOffer]:
        """Sort offers by preference (free first, then regional preferences)"""
        
        def sort_key(offer: StreamingOffer) -> Tuple:
            # Priority 1: Free content
            free_priority = 0 if offer.offer_type in [OfferType.FREE, OfferType.ADS] else 1
            
            # Priority 2: Regional platform preference
            regional_prefs = self.regional_preferences.get(country, [])
            if offer.platform in regional_prefs:
                platform_priority = regional_prefs.index(offer.platform)
            else:
                platform_priority = 100
            
            # Priority 3: Offer type preference
            offer_type_priority = {
                OfferType.FREE: 0,
                OfferType.ADS: 1,
                OfferType.SUBSCRIPTION: 2,
                OfferType.RENT: 3,
                OfferType.BUY: 4
            }
            type_priority = offer_type_priority.get(offer.offer_type, 5)
            
            # Priority 4: Number of languages
            lang_priority = -len(offer.languages)
            
            return (free_priority, platform_priority, type_priority, lang_priority)
        
        return sorted(offers, key=sort_key)
    
    def _offer_to_dict(self, offer: StreamingOffer) -> Dict:
        """Convert offer to dictionary for caching"""
        return {
            'platform': offer.platform.value[0],
            'platform_logo': offer.platform.value[2],
            'offer_type': offer.offer_type.value,
            'url': offer.url,
            'deep_link': offer.deep_link,
            'price': offer.price,
            'currency': offer.currency,
            'quality': [q.value for q in offer.quality],
            'languages': offer.languages,
            'subtitles': offer.subtitles,
            'audio_formats': offer.audio_formats,
            'expires_at': offer.expires_at.isoformat() if offer.expires_at else None,
            'region': offer.region,
            'package_info': offer.package_info
        }
    
    def format_for_response(self, availability: ContentAvailability) -> Dict:
        """Format availability data for API response"""
        
        # Group offers by platform
        platforms_grouped = {}
        for offer in availability.streaming_offers:
            platform_name = offer.platform.value[0]
            if platform_name not in platforms_grouped:
                platforms_grouped[platform_name] = {
                    'platform': platform_name,
                    'logo': offer.platform.value[2],
                    'offers': []
                }
            
            # Format offer details
            offer_detail = {
                'type': offer.offer_type.value,
                'url': offer.deep_link or offer.url,
                'quality': [q.value for q in offer.quality],
                'languages': offer.languages,
                'subtitles': offer.subtitles,
                'display_text': self._format_offer_display(offer),
                'badge': self._get_offer_badge(offer)
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
                'has_free_option': any(
                    offer.offer_type in [OfferType.FREE, OfferType.ADS] 
                    for offer in availability.streaming_offers
                ),
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
            return f"Subscription"
        elif offer.offer_type == OfferType.RENT:
            return f"Rent{f' - {offer.currency} {offer.price}' if offer.price else ''}"
        elif offer.offer_type == OfferType.BUY:
            return f"Buy{f' - {offer.currency} {offer.price}' if offer.price else ''}"
        return "Watch Now"
    
    def _get_offer_badge(self, offer: StreamingOffer) -> str:
        """Get badge for offer type"""
        badges = {
            OfferType.FREE: "FREE",
            OfferType.ADS: "FREE",
            OfferType.SUBSCRIPTION: "SUBSCRIPTION",
            OfferType.RENT: "RENT",
            OfferType.BUY: "BUY"
        }
        return badges.get(offer.offer_type, "")
    
    async def _fetch_streaming_api_async(self, session, content_id, content_type, country):
        """Async helper for API fetching"""
        # Implementation would use aiohttp
        pass
    
    async def _fetch_scraper_async(self, title, year, country):
        """Async helper for scraping"""
        # Implementation would use aiohttp
        pass
    
    async def _fetch_trailer_async(self, title, year, languages):
        """Async helper for trailer fetching"""
        # Implementation would use aiohttp
        pass

# Flask integration helper
def init_ott_service(app, cache_backend=None):
    """Initialize OTT service with Flask app"""
    
    redis_url = app.config.get('REDIS_URL')
    youtube_api_key = app.config.get('YOUTUBE_API_KEY')
    
    service = OTTAvailabilityService(
        youtube_api_key=youtube_api_key,
        redis_url=redis_url,
        cache_ttl=3600
    )
    
    # Add routes
    @app.route('/api/ott/availability/<content_id>', methods=['GET'])
    def get_ott_availability(content_id):
        """Get OTT availability for content"""
        from flask import request, jsonify
        
        try:
            # Get parameters
            title = request.args.get('title', '')
            content_type = request.args.get('type', 'movie')
            year = request.args.get('year', type=int)
            country = request.args.get('country', 'IN')
            languages = request.args.getlist('languages')
            
            if not title:
                return jsonify({'error': 'Title parameter required'}), 400
            
            # Get availability
            availability = service.get_availability(
                content_id=content_id,
                title=title,
                content_type=content_type,
                year=year,
                country=country,
                languages=languages if languages else None
            )
            
            # Format response
            response = service.format_for_response(availability)
            
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"OTT availability error: {e}")
            return jsonify({'error': 'Failed to get OTT availability'}), 500
    
    @app.route('/api/ott/search', methods=['GET'])
    def search_ott_availability():
        """Search for content availability"""
        from flask import request, jsonify
        
        try:
            query = request.args.get('q', '')
            content_type = request.args.get('type', 'both')
            country = request.args.get('country', 'IN')
            
            if not query:
                return jsonify({'error': 'Query parameter required'}), 400
            
            # Search using API
            results = service.streaming_api.search(query, content_type, country.lower())
            
            if results and results.get('results'):
                formatted_results = []
                for item in results['results'][:10]:
                    # Get availability for each result
                    availability = service.get_availability(
                        content_id=str(item.get('id', '')),
                        title=item.get('title', ''),
                        content_type=item.get('type', 'movie'),
                        year=item.get('year'),
                        country=country
                    )
                    formatted_results.append(service.format_for_response(availability))
                
                return jsonify({'results': formatted_results}), 200
            
            return jsonify({'results': [], 'message': 'No results found'}), 200
            
        except Exception as e:
            logger.error(f"OTT search error: {e}")
            return jsonify({'error': 'Search failed'}), 500
    
    return service

# Example usage
if __name__ == "__main__":
    # Initialize service
    service = OTTAvailabilityService()
    
    # Get availability for a movie
    availability = service.get_availability(
        content_id="tt0137523",  # Fight Club
        title="Fight Club",
        content_type="movie",
        year=1999,
        country="IN",
        languages=["English", "Hindi"]
    )
    
    # Format for response
    formatted = service.format_for_response(availability)
    print(json.dumps(formatted, indent=2))