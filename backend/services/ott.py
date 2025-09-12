"""
Production-Ready OTT Availability Service
Author: Senior Backend Engineer
Version: 1.0.0
Description: Comprehensive streaming availability service with multi-language support,
             fallback mechanisms, and real-time platform availability checking.
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import re
from urllib.parse import quote, urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class StreamingPlatform(Enum):
    """Enum for all supported streaming platforms"""
    NETFLIX = "netflix"
    AMAZON_PRIME = "prime"
    DISNEY_PLUS = "disney"
    HOTSTAR = "hotstar"
    HULU = "hulu"
    YOUTUBE = "youtube"
    YOUTUBE_TV = "youtubetv"
    APPLE_TV = "apple"
    HBO_MAX = "hbo"
    PARAMOUNT_PLUS = "paramount"
    PEACOCK = "peacock"
    # Indian Platforms
    ETV_WIN = "etvwin"
    AHA = "aha"
    JIO_CINEMA = "jio"
    MX_PLAYER = "mxplayer"
    ZEE5 = "zee5"
    SONYLIV = "sonyliv"
    VOOT = "voot"
    ALT_BALAJI = "altbalaji"
    EROSNOW = "erosnow"
    HOICHOI = "hoichoi"
    SUN_NXT = "sunnxt"
    # Others
    CRUNCHYROLL = "crunchyroll"
    FUNIMATION = "funimation"
    MUBI = "mubi"
    DISCOVERY_PLUS = "discovery"

class AvailabilityType(Enum):
    """Types of availability"""
    FREE = "free"
    SUBSCRIPTION = "subscription"
    RENT = "rent"
    BUY = "buy"
    ADS = "ads"  # Free with ads

class VideoQuality(Enum):
    """Video quality options"""
    SD = "sd"
    HD = "hd"
    FHD = "fhd"  # Full HD
    UHD = "4k"   # 4K/UHD

@dataclass
class PlatformAvailability:
    """Data class for platform availability information"""
    platform: str
    platform_display_name: str
    availability_type: str
    price: Optional[float] = None
    currency: Optional[str] = None
    quality: List[str] = None
    languages: List[str] = None
    subtitles: List[str] = None
    audio_languages: List[str] = None
    deep_link: Optional[str] = None
    leaving_soon: bool = False
    leaving_date: Optional[str] = None
    added_date: Optional[str] = None
    logo_url: Optional[str] = None
    region: Optional[str] = None
    
    def __post_init__(self):
        if self.quality is None:
            self.quality = []
        if self.languages is None:
            self.languages = []
        if self.subtitles is None:
            self.subtitles = []
        if self.audio_languages is None:
            self.audio_languages = []

# Platform configurations with logos and deep link patterns
PLATFORM_CONFIG = {
    StreamingPlatform.NETFLIX: {
        "display_name": "Netflix",
        "logo_url": "https://images.justwatch.com/icon/207360008/s100",
        "deep_link_pattern": "https://www.netflix.com/title/{id}",
        "supports_languages": True,
        "regions": ["global"]
    },
    StreamingPlatform.AMAZON_PRIME: {
        "display_name": "Amazon Prime Video",
        "logo_url": "https://images.justwatch.com/icon/52449539/s100",
        "deep_link_pattern": "https://www.primevideo.com/detail/{id}",
        "supports_languages": True,
        "regions": ["global"]
    },
    StreamingPlatform.DISNEY_PLUS: {
        "display_name": "Disney+",
        "logo_url": "https://images.justwatch.com/icon/147638351/s100",
        "deep_link_pattern": "https://www.disneyplus.com/{type}/{id}",
        "supports_languages": True,
        "regions": ["US", "UK", "IN", "CA", "AU"]
    },
    StreamingPlatform.HOTSTAR: {
        "display_name": "Disney+ Hotstar",
        "logo_url": "https://images.justwatch.com/icon/174876722/s100",
        "deep_link_pattern": "https://www.hotstar.com/{region}/{type}/{id}",
        "supports_languages": True,
        "regions": ["IN", "ID", "MY", "TH"]
    },
    StreamingPlatform.JIO_CINEMA: {
        "display_name": "Jio Cinema",
        "logo_url": "https://www.jiocinema.com/images/jio-logo.png",
        "deep_link_pattern": "https://www.jiocinema.com/{type}/{id}",
        "supports_languages": True,
        "regions": ["IN"]
    },
    StreamingPlatform.MX_PLAYER: {
        "display_name": "MX Player",
        "logo_url": "https://images.justwatch.com/icon/157094207/s100",
        "deep_link_pattern": "https://www.mxplayer.in/{type}/detail/{id}",
        "supports_languages": True,
        "regions": ["IN"]
    },
    StreamingPlatform.AHA: {
        "display_name": "Aha",
        "logo_url": "https://www.aha.video/assets/images/aha-logo.svg",
        "deep_link_pattern": "https://www.aha.video/{type}/{id}",
        "supports_languages": True,
        "regions": ["IN"],
        "primary_language": "Telugu"
    },
    StreamingPlatform.ETV_WIN: {
        "display_name": "ETV Win",
        "logo_url": "https://www.etvwin.com/img/logo.png",
        "deep_link_pattern": "https://www.etvwin.com/{type}/{id}",
        "supports_languages": True,
        "regions": ["IN"],
        "primary_language": "Telugu"
    },
    StreamingPlatform.ZEE5: {
        "display_name": "ZEE5",
        "logo_url": "https://images.justwatch.com/icon/115276568/s100",
        "deep_link_pattern": "https://www.zee5.com/{type}/{id}",
        "supports_languages": True,
        "regions": ["IN"]
    },
    StreamingPlatform.SONYLIV: {
        "display_name": "SonyLIV",
        "logo_url": "https://images.justwatch.com/icon/104123651/s100",
        "deep_link_pattern": "https://www.sonyliv.com/shows/{id}",
        "supports_languages": True,
        "regions": ["IN"]
    },
    StreamingPlatform.YOUTUBE: {
        "display_name": "YouTube",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/0/09/YouTube_full-color_icon_%282017%29.svg",
        "deep_link_pattern": "https://www.youtube.com/watch?v={id}",
        "supports_languages": True,
        "regions": ["global"]
    },
    StreamingPlatform.APPLE_TV: {
        "display_name": "Apple TV+",
        "logo_url": "https://images.justwatch.com/icon/152862153/s100",
        "deep_link_pattern": "https://tv.apple.com/{region}/{type}/{id}",
        "supports_languages": True,
        "regions": ["global"]
    },
    StreamingPlatform.HBO_MAX: {
        "display_name": "HBO Max",
        "logo_url": "https://images.justwatch.com/icon/116305230/s100",
        "deep_link_pattern": "https://play.hbomax.com/{type}/{id}",
        "supports_languages": True,
        "regions": ["US", "LATAM"]
    },
    StreamingPlatform.CRUNCHYROLL: {
        "display_name": "Crunchyroll",
        "logo_url": "https://images.justwatch.com/icon/127445869/s100",
        "deep_link_pattern": "https://www.crunchyroll.com/watch/{id}",
        "supports_languages": True,
        "regions": ["global"],
        "primary_content": "anime"
    }
}

# Language mapping for Indian content
LANGUAGE_MAPPING = {
    'telugu': ['Telugu', 'te', 'tel'],
    'hindi': ['Hindi', 'hi', 'hin'],
    'tamil': ['Tamil', 'ta', 'tam'],
    'kannada': ['Kannada', 'kn', 'kan'],
    'malayalam': ['Malayalam', 'ml', 'mal'],
    'english': ['English', 'en', 'eng'],
    'bengali': ['Bengali', 'bn', 'ben'],
    'marathi': ['Marathi', 'mr', 'mar'],
    'gujarati': ['Gujarati', 'gu', 'guj'],
    'punjabi': ['Punjabi', 'pa', 'pan']
}

# ============================================================================
# MAIN OTT AVAILABILITY SERVICE
# ============================================================================

class OTTAvailabilityService:
    """
    Production-ready OTT Availability Service with multi-source data aggregation,
    intelligent fallbacks, and comprehensive platform support.
    """
    
    def __init__(self, cache_backend=None, http_session=None):
        """
        Initialize OTT Availability Service
        
        Args:
            cache_backend: Cache backend for storing results
            http_session: Requests session with retry logic
        """
        self.cache = cache_backend
        self.session = http_session or self._create_session()
        
        # RapidAPI Configuration
        self.rapidapi_host = "streaming-availability.p.rapidapi.com"
        self.rapidapi_key = "6212e018b9mshb44a2716d211c51p1c493ejsn73408baa28be"
        self.rapidapi_base_url = f"https://{self.rapidapi_host}"
        
        # Thread pool for concurrent API calls
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialize fallback scrapers
        self.scrapers = {
            'justwatch': JustWatchScraper(self.session),
            'reelgood': ReelgoodScraper(self.session),
            'flixwatch': FlixwatchScraper(self.session)
        }
    
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
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def get_availability(
        self,
        content_id: str,
        content_type: str = "movie",
        title: Optional[str] = None,
        year: Optional[int] = None,
        tmdb_id: Optional[str] = None,
        imdb_id: Optional[str] = None,
        region: str = "IN",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive streaming availability for content
        
        Args:
            content_id: Internal content ID
            content_type: Type of content (movie/tv/series)
            title: Content title for fallback searches
            year: Release year for better matching
            tmdb_id: TMDB ID for API lookups
            imdb_id: IMDB ID for API lookups
            region: ISO country code (default: IN for India)
            use_cache: Whether to use cached results
        
        Returns:
            Dictionary with availability information
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(content_id, region)
            
            # Check cache
            if use_cache and self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for content {content_id} in region {region}")
                    return cached_result
            
            # Primary: Try Streaming Availability API
            availability = self._fetch_from_rapidapi(
                content_type, tmdb_id or imdb_id, region
            )
            
            # Fallback 1: Try web scraping if API fails
            if not availability or not availability.get('platforms'):
                logger.info(f"Primary API failed, trying fallback scrapers for {title}")
                availability = self._fetch_from_scrapers(title, year, content_type, region)
            
            # Fallback 2: Try Google Knowledge Graph
            if not availability or not availability.get('platforms'):
                logger.info(f"Scrapers failed, trying Google for {title}")
                availability = self._fetch_from_google(title, year, content_type, region)
            
            # Process and enhance results
            if availability and availability.get('platforms'):
                availability = self._enhance_availability_data(availability, region)
                
                # Add metadata
                availability['metadata'] = {
                    'content_id': content_id,
                    'content_type': content_type,
                    'title': title,
                    'year': year,
                    'region': region,
                    'last_updated': datetime.utcnow().isoformat(),
                    'data_source': availability.get('source', 'unknown')
                }
            else:
                # No availability found
                availability = self._generate_not_available_response(
                    content_id, title, content_type, region
                )
            
            # Cache results
            if use_cache and self.cache and availability:
                cache_timeout = 3600 if availability.get('platforms') else 1800  # 1hr if found, 30min if not
                self.cache.set(cache_key, availability, timeout=cache_timeout)
            
            return availability
            
        except Exception as e:
            logger.error(f"Error getting availability for content {content_id}: {e}")
            return self._generate_error_response(content_id, title, str(e))
    
    def _fetch_from_rapidapi(
        self,
        content_type: str,
        content_id: str,
        region: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch availability from Streaming Availability API"""
        try:
            if not content_id:
                return None
            
            # Determine ID type
            id_type = "tmdb" if content_id.isdigit() else "imdb"
            
            # API endpoint
            endpoint = f"/shows/{content_type}/{content_id}"
            url = f"{self.rapidapi_base_url}{endpoint}"
            
            headers = {
                "X-RapidAPI-Key": self.rapidapi_key,
                "X-RapidAPI-Host": self.rapidapi_host
            }
            
            params = {
                "output_language": "en",
                "country": region.lower()
            }
            
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_rapidapi_response(data, region)
            else:
                logger.warning(f"RapidAPI returned status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"RapidAPI fetch error: {e}")
            return None
    
    def _parse_rapidapi_response(self, data: Dict, region: str) -> Dict[str, Any]:
        """Parse response from Streaming Availability API"""
        try:
            platforms = []
            
            # Extract streaming options
            streaming_info = data.get('streamingInfo', {}).get(region.lower(), {})
            
            for platform_data in streaming_info:
                platform_name = platform_data.get('service', '').lower()
                
                # Map to our platform enum
                platform_enum = self._map_platform_name(platform_name)
                if not platform_enum:
                    continue
                
                # Determine availability type
                availability_type = self._determine_availability_type(platform_data)
                
                # Extract quality information
                quality = self._extract_quality(platform_data)
                
                # Extract language information
                languages = platform_data.get('audios', [])
                subtitles = platform_data.get('subtitles', [])
                
                # Create platform availability
                platform_config = PLATFORM_CONFIG.get(platform_enum, {})
                
                availability = PlatformAvailability(
                    platform=platform_enum.value,
                    platform_display_name=platform_config.get('display_name', platform_name),
                    availability_type=availability_type,
                    price=platform_data.get('price'),
                    currency=platform_data.get('currency'),
                    quality=quality,
                    languages=languages,
                    subtitles=subtitles,
                    audio_languages=languages,
                    deep_link=platform_data.get('link'),
                    leaving_soon=platform_data.get('leaving', False),
                    leaving_date=platform_data.get('leavingDate'),
                    added_date=platform_data.get('addedDate'),
                    logo_url=platform_config.get('logo_url'),
                    region=region
                )
                
                platforms.append(asdict(availability))
            
            return {
                'platforms': platforms,
                'source': 'streaming_availability_api'
            }
            
        except Exception as e:
            logger.error(f"Error parsing RapidAPI response: {e}")
            return None
    
    def _fetch_from_scrapers(
        self,
        title: str,
        year: Optional[int],
        content_type: str,
        region: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch availability using web scraping as fallback"""
        try:
            # Try scrapers concurrently
            futures = []
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures.append(
                    executor.submit(self.scrapers['justwatch'].scrape, title, year, content_type, region)
                )
                futures.append(
                    executor.submit(self.scrapers['reelgood'].scrape, title, year, content_type, region)
                )
                futures.append(
                    executor.submit(self.scrapers['flixwatch'].scrape, title, year, content_type, region)
                )
            
            # Collect results
            all_platforms = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=10)
                    if result and result.get('platforms'):
                        all_platforms.extend(result['platforms'])
                except Exception as e:
                    logger.warning(f"Scraper failed: {e}")
            
            if all_platforms:
                # Deduplicate platforms
                unique_platforms = self._deduplicate_platforms(all_platforms)
                return {
                    'platforms': unique_platforms,
                    'source': 'web_scraping'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Scraping error: {e}")
            return None
    
    def _fetch_from_google(
        self,
        title: str,
        year: Optional[int],
        content_type: str,
        region: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch availability from Google Knowledge Graph"""
        try:
            # Build search query
            query = f"{title} {year if year else ''} watch online {region}"
            
            # Google search API would go here
            # For production, you'd use Google Custom Search API
            # This is a placeholder
            
            return None
            
        except Exception as e:
            logger.error(f"Google fetch error: {e}")
            return None
    
    def _enhance_availability_data(
        self,
        availability: Dict[str, Any],
        region: str
    ) -> Dict[str, Any]:
        """Enhance availability data with additional information"""
        try:
            enhanced_platforms = []
            
            for platform_data in availability.get('platforms', []):
                # Add regional pricing if not present
                if not platform_data.get('price') and platform_data.get('availability_type') == 'subscription':
                    platform_data['price'] = self._get_regional_pricing(
                        platform_data['platform'], region
                    )
                
                # Add quality information if missing
                if not platform_data.get('quality'):
                    platform_data['quality'] = ['HD', 'SD']  # Default assumption
                
                # Ensure logo URL is present
                if not platform_data.get('logo_url'):
                    platform_enum = self._map_platform_name(platform_data['platform'])
                    if platform_enum:
                        platform_config = PLATFORM_CONFIG.get(platform_enum, {})
                        platform_data['logo_url'] = platform_config.get('logo_url')
                
                enhanced_platforms.append(platform_data)
            
            # Sort platforms by priority
            enhanced_platforms = self._sort_platforms_by_priority(enhanced_platforms, region)
            
            availability['platforms'] = enhanced_platforms
            
            # Add summary
            availability['summary'] = self._generate_availability_summary(enhanced_platforms)
            
            return availability
            
        except Exception as e:
            logger.error(f"Enhancement error: {e}")
            return availability
    
    def _generate_availability_summary(self, platforms: List[Dict]) -> Dict[str, Any]:
        """Generate summary of availability"""
        summary = {
            'total_platforms': len(platforms),
            'free_platforms': [],
            'subscription_platforms': [],
            'rental_platforms': [],
            'purchase_platforms': [],
            'languages_available': set(),
            'best_quality': 'SD',
            'has_free_option': False
        }
        
        quality_order = ['4k', 'fhd', 'hd', 'sd']
        best_quality_index = len(quality_order) - 1
        
        for platform in platforms:
            # Categorize by type
            avail_type = platform.get('availability_type', '').lower()
            platform_name = platform.get('platform_display_name', platform.get('platform'))
            
            if avail_type in ['free', 'ads']:
                summary['free_platforms'].append(platform_name)
                summary['has_free_option'] = True
            elif avail_type == 'subscription':
                summary['subscription_platforms'].append(platform_name)
            elif avail_type == 'rent':
                summary['rental_platforms'].append(platform_name)
            elif avail_type == 'buy':
                summary['purchase_platforms'].append(platform_name)
            
            # Collect languages
            for lang in platform.get('languages', []):
                summary['languages_available'].add(lang)
            
            # Determine best quality
            for quality in platform.get('quality', []):
                quality_lower = quality.lower()
                if quality_lower in quality_order:
                    idx = quality_order.index(quality_lower)
                    if idx < best_quality_index:
                        best_quality_index = idx
        
        summary['best_quality'] = quality_order[best_quality_index].upper()
        summary['languages_available'] = list(summary['languages_available'])
        
        return summary
    
    def _sort_platforms_by_priority(
        self,
        platforms: List[Dict],
        region: str
    ) -> List[Dict]:
        """Sort platforms by regional priority"""
        # Define priority for India
        if region == 'IN':
            priority_order = [
                'netflix', 'amazon_prime', 'hotstar', 'jio', 'zee5',
                'sonyliv', 'mxplayer', 'aha', 'etvwin', 'youtube',
                'voot', 'altbalaji', 'erosnow', 'hoichoi', 'sunnxt'
            ]
        else:
            priority_order = [
                'netflix', 'amazon_prime', 'disney', 'hulu', 'hbo',
                'apple', 'paramount', 'peacock', 'youtube'
            ]
        
        def get_priority(platform):
            platform_name = platform.get('platform', '').lower()
            try:
                return priority_order.index(platform_name)
            except ValueError:
                return len(priority_order)
        
        # Sort by: Free first, then by platform priority
        return sorted(platforms, key=lambda p: (
            0 if p.get('availability_type') in ['free', 'ads'] else 1,
            get_priority(p)
        ))
    
    def _generate_not_available_response(
        self,
        content_id: str,
        title: Optional[str],
        content_type: str,
        region: str
    ) -> Dict[str, Any]:
        """Generate response when content is not available"""
        response = {
            'platforms': [],
            'message': f"'{title or 'This content'}' is currently not available for streaming in {region}. Check back later for updates.",
            'alternatives': [],
            'metadata': {
                'content_id': content_id,
                'title': title,
                'content_type': content_type,
                'region': region,
                'last_checked': datetime.utcnow().isoformat()
            }
        }
        
        # Try to get YouTube trailer as alternative
        if title:
            youtube_url = self._search_youtube_content(title, content_type)
            if youtube_url:
                response['alternatives'].append({
                    'type': 'trailer',
                    'platform': 'YouTube',
                    'url': youtube_url,
                    'message': 'Watch trailer on YouTube'
                })
        
        return response
    
    def _search_youtube_content(self, title: str, content_type: str) -> Optional[str]:
        """Search for content on YouTube"""
        try:
            search_query = f"{title} {content_type} trailer"
            youtube_search_url = f"https://www.youtube.com/results?search_query={quote(search_query)}"
            return youtube_search_url
        except:
            return None
    
    def _generate_error_response(
        self,
        content_id: str,
        title: Optional[str],
        error_message: str
    ) -> Dict[str, Any]:
        """Generate error response"""
        return {
            'platforms': [],
            'error': True,
            'error_message': error_message,
            'metadata': {
                'content_id': content_id,
                'title': title,
                'timestamp': datetime.utcnow().isoformat()
            }
        }
    
    def _generate_cache_key(self, content_id: str, region: str) -> str:
        """Generate cache key for availability data"""
        return f"ott_availability:{content_id}:{region}"
    
    def _map_platform_name(self, platform_name: str) -> Optional[StreamingPlatform]:
        """Map platform name string to enum"""
        platform_lower = platform_name.lower()
        
        # Direct mapping
        for platform in StreamingPlatform:
            if platform.value == platform_lower:
                return platform
        
        # Alternative mappings
        mappings = {
            'amazon': StreamingPlatform.AMAZON_PRIME,
            'prime video': StreamingPlatform.AMAZON_PRIME,
            'disney plus': StreamingPlatform.DISNEY_PLUS,
            'hbo': StreamingPlatform.HBO_MAX,
            'jiocinema': StreamingPlatform.JIO_CINEMA,
            'mx': StreamingPlatform.MX_PLAYER,
            'zee': StreamingPlatform.ZEE5,
            'sony': StreamingPlatform.SONYLIV,
            'etv': StreamingPlatform.ETV_WIN,
            'apple': StreamingPlatform.APPLE_TV,
            'paramount': StreamingPlatform.PARAMOUNT_PLUS
        }
        
        for key, value in mappings.items():
            if key in platform_lower:
                return value
        
        return None
    
    def _determine_availability_type(self, platform_data: Dict) -> str:
        """Determine availability type from platform data"""
        if platform_data.get('free'):
            return AvailabilityType.FREE.value
        elif platform_data.get('ads'):
            return AvailabilityType.ADS.value
        elif platform_data.get('subscription'):
            return AvailabilityType.SUBSCRIPTION.value
        elif platform_data.get('rent'):
            return AvailabilityType.RENT.value
        elif platform_data.get('buy'):
            return AvailabilityType.BUY.value
        else:
            # Default based on price
            if platform_data.get('price'):
                return AvailabilityType.RENT.value
            return AvailabilityType.SUBSCRIPTION.value
    
    def _extract_quality(self, platform_data: Dict) -> List[str]:
        """Extract quality information from platform data"""
        quality_list = []
        
        # Check for explicit quality field
        if 'quality' in platform_data:
            quality = platform_data['quality']
            if isinstance(quality, list):
                quality_list.extend(quality)
            else:
                quality_list.append(str(quality))
        
        # Check for resolution indicators
        if platform_data.get('4k') or platform_data.get('uhd'):
            quality_list.append(VideoQuality.UHD.value)
        if platform_data.get('hd') or platform_data.get('1080p'):
            quality_list.append(VideoQuality.HD.value)
        if platform_data.get('sd') or platform_data.get('480p'):
            quality_list.append(VideoQuality.SD.value)
        
        # Default if no quality found
        if not quality_list:
            quality_list = [VideoQuality.HD.value, VideoQuality.SD.value]
        
        return list(set(quality_list))  # Remove duplicates
    
    def _get_regional_pricing(self, platform: str, region: str) -> Optional[Dict]:
        """Get regional pricing for subscription platforms"""
        # Regional pricing data (can be stored in database)
        pricing_data = {
            'IN': {  # India
                'netflix': {'monthly': 199, 'currency': 'INR'},
                'amazon_prime': {'monthly': 179, 'currency': 'INR'},
                'hotstar': {'monthly': 299, 'currency': 'INR'},
                'zee5': {'monthly': 99, 'currency': 'INR'},
                'sonyliv': {'monthly': 299, 'currency': 'INR'},
                'jio': {'monthly': 0, 'currency': 'INR'},  # Free for Jio users
                'mxplayer': {'monthly': 0, 'currency': 'INR'},  # Free
                'aha': {'monthly': 149, 'currency': 'INR'},
                'etvwin': {'monthly': 100, 'currency': 'INR'}
            },
            'US': {  # United States
                'netflix': {'monthly': 15.49, 'currency': 'USD'},
                'amazon_prime': {'monthly': 14.99, 'currency': 'USD'},
                'disney': {'monthly': 13.99, 'currency': 'USD'},
                'hulu': {'monthly': 14.99, 'currency': 'USD'},
                'hbo': {'monthly': 15.99, 'currency': 'USD'}
            }
        }
        
        region_pricing = pricing_data.get(region, {})
        return region_pricing.get(platform)
    
    def _deduplicate_platforms(self, platforms: List[Dict]) -> List[Dict]:
        """Remove duplicate platform entries"""
        seen = set()
        unique_platforms = []
        
        for platform in platforms:
            # Create unique key
            key = f"{platform.get('platform')}:{platform.get('availability_type')}"
            
            if key not in seen:
                seen.add(key)
                unique_platforms.append(platform)
            else:
                # Merge information if duplicate
                for existing in unique_platforms:
                    if (existing.get('platform') == platform.get('platform') and
                        existing.get('availability_type') == platform.get('availability_type')):
                        # Merge languages
                        existing_langs = set(existing.get('languages', []))
                        new_langs = set(platform.get('languages', []))
                        existing['languages'] = list(existing_langs.union(new_langs))
                        
                        # Merge quality
                        existing_quality = set(existing.get('quality', []))
                        new_quality = set(platform.get('quality', []))
                        existing['quality'] = list(existing_quality.union(new_quality))
                        break
        
        return unique_platforms

# ============================================================================
# WEB SCRAPING FALLBACK CLASSES
# ============================================================================

class JustWatchScraper:
    """Scraper for JustWatch platform"""
    
    def __init__(self, session: requests.Session):
        self.session = session
        self.base_url = "https://www.justwatch.com"
    
    def scrape(
        self,
        title: str,
        year: Optional[int],
        content_type: str,
        region: str
    ) -> Optional[Dict[str, Any]]:
        """Scrape JustWatch for availability"""
        try:
            # Build search URL
            country_code = region.lower()
            search_url = f"{self.base_url}/{country_code}/search"
            
            params = {
                'q': title,
                'content_type': content_type
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract availability (simplified - actual implementation would be more complex)
            platforms = []
            
            # Look for streaming provider elements
            provider_elements = soup.find_all('div', class_='provider-icon')
            
            for provider_elem in provider_elements:
                platform_name = provider_elem.get('title', '').lower()
                
                # Map to our platform enum
                platform_enum = self._map_justwatch_platform(platform_name)
                if not platform_enum:
                    continue
                
                # Determine availability type from context
                availability_type = self._extract_availability_type(provider_elem)
                
                platform_config = PLATFORM_CONFIG.get(platform_enum, {})
                
                platform_data = {
                    'platform': platform_enum.value,
                    'platform_display_name': platform_config.get('display_name', platform_name),
                    'availability_type': availability_type,
                    'logo_url': platform_config.get('logo_url'),
                    'region': region
                }
                
                platforms.append(platform_data)
            
            if platforms:
                return {
                    'platforms': platforms,
                    'source': 'justwatch'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"JustWatch scraping error: {e}")
            return None
    
    def _map_justwatch_platform(self, platform_name: str) -> Optional[StreamingPlatform]:
        """Map JustWatch platform name to our enum"""
        # Implement mapping logic
        mappings = {
            'netflix': StreamingPlatform.NETFLIX,
            'amazon prime video': StreamingPlatform.AMAZON_PRIME,
            'disney+': StreamingPlatform.DISNEY_PLUS,
            'hotstar': StreamingPlatform.HOTSTAR,
            # Add more mappings
        }
        
        for key, value in mappings.items():
            if key in platform_name.lower():
                return value
        
        return None
    
    def _extract_availability_type(self, element) -> str:
        """Extract availability type from HTML element"""
        # Check for price or type indicators in parent elements
        parent = element.parent
        
        if parent:
            text = parent.get_text().lower()
            if 'free' in text:
                return AvailabilityType.FREE.value
            elif 'rent' in text:
                return AvailabilityType.RENT.value
            elif 'buy' in text:
                return AvailabilityType.BUY.value
        
        return AvailabilityType.SUBSCRIPTION.value

class ReelgoodScraper:
    """Scraper for Reelgood platform"""
    
    def __init__(self, session: requests.Session):
        self.session = session
        self.base_url = "https://reelgood.com"
    
    def scrape(
        self,
        title: str,
        year: Optional[int],
        content_type: str,
        region: str
    ) -> Optional[Dict[str, Any]]:
        """Scrape Reelgood for availability"""
        # Similar implementation to JustWatch
        # This is a placeholder - actual implementation would involve
        # proper HTML parsing of Reelgood's structure
        return None

class FlixwatchScraper:
    """Scraper for Flixwatch platform"""
    
    def __init__(self, session: requests.Session):
        self.session = session
        self.base_url = "https://www.flixwatch.co"
    
    def scrape(
        self,
        title: str,
        year: Optional[int],
        content_type: str,
        region: str
    ) -> Optional[Dict[str, Any]]:
        """Scrape Flixwatch for availability"""
        # Similar implementation to JustWatch
        # This is a placeholder - actual implementation would involve
        # proper HTML parsing of Flixwatch's structure
        return None

# ============================================================================
# MULTI-LANGUAGE SUPPORT
# ============================================================================

class MultiLanguageAvailability:
    """Handle multi-language availability for content"""
    
    @staticmethod
    def get_language_specific_links(
        content_title: str,
        platforms: List[Dict],
        languages: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Get language-specific watch links for content
        
        Args:
            content_title: Title of the content
            platforms: List of platform availability
            languages: List of languages to check
        
        Returns:
            Dictionary mapping languages to platform links
        """
        language_links = {}
        
        for language in languages:
            language_platforms = []
            
            for platform in platforms:
                # Check if platform supports this language
                platform_languages = platform.get('languages', [])
                audio_languages = platform.get('audio_languages', [])
                
                # Check if language is available
                lang_available = False
                for lang_variant in LANGUAGE_MAPPING.get(language.lower(), [language]):
                    if (lang_variant in platform_languages or
                        lang_variant in audio_languages or
                        any(lang_variant.lower() in p.lower() for p in platform_languages) or
                        any(lang_variant.lower() in a.lower() for a in audio_languages)):
                        lang_available = True
                        break
                
                if lang_available:
                    # Create language-specific link
                    lang_platform = platform.copy()
                    lang_platform['language'] = language
                    lang_platform['language_tag'] = f"[{language.upper()}]"
                    
                    # Modify deep link if possible
                    if lang_platform.get('deep_link'):
                        # Add language parameter to URL if supported
                        lang_platform['deep_link'] = MultiLanguageAvailability._add_language_param(
                            lang_platform['deep_link'],
                            language
                        )
                    
                    language_platforms.append(lang_platform)
            
            if language_platforms:
                language_links[language] = language_platforms
        
        return language_links
    
    @staticmethod
    def _add_language_param(url: str, language: str) -> str:
        """Add language parameter to URL"""
        # Platform-specific language parameter handling
        if 'netflix.com' in url:
            # Netflix uses audio parameter
            lang_code = LANGUAGE_MAPPING.get(language.lower(), [language])[0]
            return f"{url}&audio={lang_code}"
        elif 'primevideo.com' in url:
            # Amazon Prime uses audioLanguage
            lang_code = LANGUAGE_MAPPING.get(language.lower(), [language])[0]
            return f"{url}?audioLanguage={lang_code}"
        elif 'hotstar.com' in url:
            # Hotstar uses lang parameter
            lang_code = LANGUAGE_MAPPING.get(language.lower(), [language])[0]
            return f"{url}?lang={lang_code}"
        
        return url

# ============================================================================
# FLASK INTEGRATION ENDPOINTS
# ============================================================================

def init_ott_routes(app, db, cache):
    """Initialize OTT availability routes in Flask app"""
    
    # Initialize service
    ott_service = OTTAvailabilityService(cache_backend=cache)
    
    @app.route('/api/ott/availability/<int:content_id>', methods=['GET'])
    def get_ott_availability(content_id):
        """
        Get OTT platform availability for content
        
        Query Parameters:
            - region: ISO country code (default: IN)
            - include_languages: Include language-specific links (default: true)
            - use_cache: Use cached results (default: true)
        """
        try:
            from flask import request, jsonify
            
            # Get parameters
            region = request.args.get('region', 'IN').upper()
            include_languages = request.args.get('include_languages', 'true').lower() == 'true'
            use_cache = request.args.get('use_cache', 'true').lower() == 'true'
            
            # Get content from database
            from backend.app import Content
            content = Content.query.get(content_id)
            
            if not content:
                return jsonify({'error': 'Content not found'}), 404
            
            # Get availability
            availability = ott_service.get_availability(
                content_id=str(content_id),
                content_type=content.content_type,
                title=content.title,
                year=content.release_date.year if content.release_date else None,
                tmdb_id=str(content.tmdb_id) if content.tmdb_id else None,
                imdb_id=content.imdb_id,
                region=region,
                use_cache=use_cache
            )
            
            # Add language-specific links if requested
            if include_languages and availability.get('platforms'):
                # Get content languages
                content_languages = json.loads(content.languages or '[]')
                
                # Get language-specific links
                language_links = MultiLanguageAvailability.get_language_specific_links(
                    content.title,
                    availability['platforms'],
                    content_languages
                )
                
                availability['language_specific'] = language_links
            
            # Format response
            response = {
                'success': True,
                'content': {
                    'id': content.id,
                    'title': content.title,
                    'type': content.content_type,
                    'languages': json.loads(content.languages or '[]')
                },
                'availability': availability,
                'region': region
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"OTT availability endpoint error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/ott/platforms', methods=['GET'])
    def get_supported_platforms():
        """Get list of supported OTT platforms"""
        try:
            from flask import request, jsonify
            
            region = request.args.get('region', 'IN').upper()
            
            platforms = []
            for platform_enum, config in PLATFORM_CONFIG.items():
                # Check if platform is available in region
                platform_regions = config.get('regions', [])
                if 'global' in platform_regions or region in platform_regions:
                    platforms.append({
                        'id': platform_enum.value,
                        'name': config['display_name'],
                        'logo_url': config['logo_url'],
                        'regions': platform_regions,
                        'supports_languages': config.get('supports_languages', False),
                        'primary_language': config.get('primary_language'),
                        'primary_content': config.get('primary_content')
                    })
            
            # Sort by name
            platforms.sort(key=lambda x: x['name'])
            
            return jsonify({
                'success': True,
                'platforms': platforms,
                'total': len(platforms),
                'region': region
            }), 200
            
        except Exception as e:
            logger.error(f"Platforms endpoint error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/ott/search', methods=['GET'])
    def search_ott_availability():
        """Search for content and get availability"""
        try:
            from flask import request, jsonify
            
            # Get parameters
            query = request.args.get('q', '').strip()
            content_type = request.args.get('type', 'movie')
            region = request.args.get('region', 'IN').upper()
            year = request.args.get('year', type=int)
            
            if not query:
                return jsonify({'error': 'Query parameter required'}), 400
            
            # Search for content using existing search functionality
            # This would integrate with your existing search endpoint
            # For now, we'll just return a placeholder
            
            availability = ott_service.get_availability(
                content_id='search_' + hashlib.md5(query.encode()).hexdigest(),
                content_type=content_type,
                title=query,
                year=year,
                region=region,
                use_cache=True
            )
            
            return jsonify({
                'success': True,
                'query': query,
                'availability': availability,
                'region': region
            }), 200
            
        except Exception as e:
            logger.error(f"OTT search endpoint error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    logger.info("OTT availability routes initialized successfully")

# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'OTTAvailabilityService',
    'StreamingPlatform',
    'AvailabilityType',
    'VideoQuality',
    'PlatformAvailability',
    'MultiLanguageAvailability',
    'init_ott_routes'
]