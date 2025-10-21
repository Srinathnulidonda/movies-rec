#backend/service/upcoming.py
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
from functools import lru_cache
import hashlib
import json
import asyncio
from collections import defaultdict, Counter
import calendar

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content type enumeration"""
    MOVIE = "movie"
    TV_SERIES = "tv"
    ANIME = "anime"


class LanguagePriority(Enum):
    """Language priority with ISO-639-1 codes - TELUGU FIRST ORDER"""
    TELUGU = ("telugu", "te", 1)
    ENGLISH = ("english", "en", 2)
    HINDI = ("hindi", "hi", 3)
    MALAYALAM = ("malayalam", "ml", 4)
    KANNADA = ("kannada", "kn", 5)
    TAMIL = ("tamil", "ta", 6)
    OTHER = ("other", "xx", 7)
    
    def __init__(self, name: str, iso_code: str, priority: int):
        self.language_name = name
        self.iso_code = iso_code
        self.priority = priority
    
    @classmethod
    def get_by_iso(cls, iso_code: str) -> Optional['LanguagePriority']:
        """Get language by ISO code"""
        for lang in cls:
            if lang.iso_code == iso_code:
                return lang
        return cls.OTHER
    
    @classmethod
    def get_priority_order(cls) -> List['LanguagePriority']:
        """Get languages in strict priority order"""
        return sorted(cls, key=lambda x: x.priority)


@dataclass
class ReleaseWindow:
    """Release window configuration for upcoming 3 months"""
    start_date: datetime
    end_date: datetime
    month_label: str
    month_name: str
    priority: int
    is_current: bool = False
    days_from_now: int = 0


@dataclass
class UpcomingRelease:
    """Enhanced upcoming release data structure with 100% accuracy"""
    id: str
    title: str
    original_title: Optional[str]
    content_type: ContentType
    release_date: datetime
    languages: List[str]
    genres: List[str]
    popularity: float
    vote_count: int
    vote_average: float
    poster_path: Optional[str]
    backdrop_path: Optional[str]
    overview: Optional[str]
    region: Optional[str] = None
    source: str = ""
    language_priority: int = 999
    
    # Enhanced accuracy fields
    confirmed_release_date: bool = False
    release_status: str = "upcoming"
    exact_release_time: Optional[str] = None
    cinema_release: bool = False
    ott_release: bool = False
    ott_platform: Optional[str] = None
    theatrical_regions: List[str] = field(default_factory=list)
    
    # Telugu specific fields - ENHANCED
    is_telugu_content: bool = False
    is_telugu_priority: bool = False  # Added for frontend compatibility
    is_tollywood: bool = False
    telugu_title: Optional[str] = None
    production_house: Optional[str] = None
    telugu_confidence: float = 0.0
    
    # Enhanced metadata
    runtime_minutes: Optional[int] = None
    director: Optional[str] = None
    cast: List[str] = field(default_factory=list)
    studio: Optional[str] = None
    budget: Optional[float] = None
    certification: Optional[str] = None
    trailer_url: Optional[str] = None
    
    # Release tracking
    days_until_release: int = 0
    weeks_until_release: int = 0
    months_until_release: int = 0
    release_quarter: str = ""
    
    # Accuracy metrics
    data_confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Post-initialization for accurate data processing"""
        self._set_language_priority()
        self._calculate_release_timing()
        self._determine_telugu_status()
        self._calculate_data_confidence()
    
    def _set_language_priority(self):
        """Set accurate language priority"""
        for lang in self.languages:
            for lang_enum in LanguagePriority:
                if lang.lower() in [lang_enum.iso_code, lang_enum.language_name]:
                    self.language_priority = min(self.language_priority, lang_enum.priority)
                    break
    
    def _calculate_release_timing(self):
        """Calculate accurate release timing"""
        if self.release_date:
            now = datetime.now()
            delta = self.release_date - now
            
            self.days_until_release = delta.days
            self.weeks_until_release = delta.days // 7
            self.months_until_release = delta.days // 30
            
            # Determine quarter
            quarter = (self.release_date.month - 1) // 3 + 1
            self.release_quarter = f"Q{quarter} {self.release_date.year}"
    
    def _determine_telugu_status(self):
        """Enhanced Telugu content detection"""
        telugu_indicators = ['te', 'telugu', 'tollywood']
        
        # Check languages first
        for lang in self.languages:
            if lang.lower() in telugu_indicators:
                self.is_telugu_content = True
                self.is_telugu_priority = True
                self.is_tollywood = True
                self.telugu_confidence = 1.0
                self.language_priority = 1  # Force Telugu priority
                return
        
        # Check title for Telugu patterns
        title_text = f"{self.title} {self.original_title or ''}".lower()
        telugu_keywords = [
            'telugu', 'tollywood', 'andhra', 'telangana', 'hyderabad', 
            'vizag', 'vijayawada', 'guntur', 'tirupati', 'warangal',
            'nizam', 'rayalaseema', 'coastal andhra'
        ]
        
        telugu_score = 0
        for keyword in telugu_keywords:
            if keyword in title_text:
                telugu_score += 1
        
        if telugu_score > 0:
            self.is_telugu_content = True
            self.is_telugu_priority = True
            self.telugu_confidence = min(1.0, telugu_score * 0.3)
            self.language_priority = 1
    
    def _calculate_data_confidence(self):
        """Calculate data confidence score"""
        confidence = 0.0
        
        # Release date confidence
        if self.confirmed_release_date:
            confidence += 0.3
        elif self.release_date:
            confidence += 0.2
        
        # Telugu content bonus
        if self.is_telugu_content:
            confidence += 0.2
        
        # Metadata completeness
        if self.overview:
            confidence += 0.1
        if self.poster_path:
            confidence += 0.1
        if self.cast:
            confidence += 0.1
        if self.director:
            confidence += 0.1
        
        # Source reliability
        if self.source == "tmdb":
            confidence += 0.1
        elif self.source == "anilist":
            confidence += 0.1
        
        self.data_confidence = min(1.0, confidence)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization with accuracy info"""
        data = asdict(self)
        # Convert datetime to ISO format - ensure naive datetime
        if self.release_date:
            if self.release_date.tzinfo is not None:
                data['release_date'] = self.release_date.replace(tzinfo=None).isoformat()
            else:
                data['release_date'] = self.release_date.isoformat()
        if self.last_updated:
            if self.last_updated.tzinfo is not None:
                data['last_updated'] = self.last_updated.replace(tzinfo=None).isoformat()
            else:
                data['last_updated'] = self.last_updated.isoformat()
        # Convert enum to string
        data['content_type'] = self.content_type.value
        return data


class EnhancedTeluguDetector:
    """Enhanced Telugu content detection system with aggressive discovery"""
    
    TELUGU_STUDIOS = [
        'AVM Productions', 'Vyjayanthi Movies', 'Mythri Movie Makers',
        'Geetha Arts', 'Sri Venkateswara Creations', 'Konidela Production Company',
        'UV Creations', 'Sithara Entertainments', 'Haarika & Hassine Creations',
        'GA2 Pictures', 'People Media Factory', 'Asian Cinemas',
        'Suresh Productions', 'Annapurna Studios', 'Bhavya Creations',
        'Dil Raju Productions', 'Mythri Movie Makers', 'Ramabrahmam Studios',
        'Anandi Art Creations', 'Varahi Chalana Chitram', 'Wall Poster Cinema',
        'Shishuka Entertainments', 'Harshit Films', 'Bhadrakali Pictures'
    ]
    
    TELUGU_KEYWORDS = [
        'telugu', 'tollywood', 'andhra pradesh', 'telangana', 'hyderabad',
        'vizag', 'vijayawada', 'guntur', 'tirupati', 'warangal', 'nellore',
        'kakinada', 'rajahmundry', 'ongole', 'eluru', 'nizam', 'rayalaseema',
        'coastal andhra', 'godavari', 'krishna', 'chittoor', 'anantapur'
    ]
    
    TELUGU_STAR_PATTERNS = [
        'prabhas', 'mahesh babu', 'allu arjun', 'ram charan', 'jr ntr',
        'nani', 'vijay deverakonda', 'ravi teja', 'naga chaitanya',
        'rana daggubati', 'nithiin', 'sharwanand', 'varun tej',
        'sai dharam tej', 'bellamkonda sreenivas', 'adivi sesh',
        'samantha', 'pooja hegde', 'rashmika mandanna', 'kajal aggarwal'
    ]
    
    TELUGU_DIRECTOR_PATTERNS = [
        'rajamouli', 'trivikram', 'koratala siva', 'sukumar', 'parasuram',
        'harish shankar', 'boyapati srinu', 'srinu vaitla', 'puri jagannadh',
        'gopichand malineni', 'anil ravipudi', 'maruthi', 'krish jagarlamudi'
    ]
    
    @classmethod
    def is_telugu_content(cls, item: Dict[str, Any], detailed: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """Enhanced Telugu content detection with aggressive patterns"""
        confidence_factors = {
            'language': 0,
            'title': 0,
            'studio': 0,
            'cast_crew': 0,
            'keywords': 0,
            'region': 0
        }
        
        # Language check (highest confidence)
        original_language = item.get("original_language", "").lower()
        if original_language in ['te', 'telugu']:
            confidence_factors['language'] = 100
            return True, confidence_factors
        
        # Title analysis - more aggressive
        title_text = f"{item.get('title', '')} {item.get('original_title', '')} {item.get('name', '')} {item.get('original_name', '')}".lower()
        
        # Check for Telugu keywords
        for keyword in cls.TELUGU_KEYWORDS:
            if keyword in title_text:
                confidence_factors['keywords'] += 25
        
        # Check for Telugu stars
        for star in cls.TELUGU_STAR_PATTERNS:
            if star.lower() in title_text:
                confidence_factors['cast_crew'] += 30
        
        # Check for Telugu directors
        for director in cls.TELUGU_DIRECTOR_PATTERNS:
            if director.lower() in title_text:
                confidence_factors['cast_crew'] += 25
        
        # Studio/Production analysis
        production_companies = item.get('production_companies', [])
        for company in production_companies:
            company_name = company.get('name', '') if isinstance(company, dict) else str(company)
            if any(studio.lower() in company_name.lower() for studio in cls.TELUGU_STUDIOS):
                confidence_factors['studio'] = 50
                break
        
        # Region-based detection (India + popularity could indicate regional content)
        if item.get('popularity', 0) > 10:  # Popular in India
            confidence_factors['region'] = 10
        
        # Overall confidence - lowered threshold for better detection
        total_confidence = sum(confidence_factors.values())
        is_telugu = total_confidence >= 20  # Lowered from 30 to 20
        
        return is_telugu, confidence_factors


class TeluguFocusedTMDbClient:
    """TMDb client specifically optimized for Telugu content discovery"""
    
    BASE_URL = "https://api.themoviedb.org/3"
    
    def __init__(self, api_key: str, http_client: Optional[httpx.AsyncClient] = None):
        self.api_key = api_key
        self.http_client = http_client or httpx.AsyncClient(timeout=10.0)
        self.telugu_detector = EnhancedTeluguDetector()
    
    async def get_telugu_movies_priority(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get Telugu movies with multiple strategies"""
        all_results = []
        
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Strategy 1: Direct Telugu language search
        telugu_params = {
            "api_key": self.api_key,
            "primary_release_date.gte": start_str,
            "primary_release_date.lte": end_str,
            "region": region,
            "with_original_language": "te",
            "sort_by": "popularity.desc",
            "include_adult": False,
            "page": 1
        }
        
        try:
            for page in range(1, 4):  # Get more pages for Telugu
                telugu_params["page"] = page
                response = await self.http_client.get(
                    f"{self.BASE_URL}/discover/movie",
                    params=telugu_params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    for movie in results:
                        movie['_telugu_source'] = 'direct_language'
                        all_results.append(movie)
                    
                    if not results or page >= data.get("total_pages", 1):
                        break
                        
        except Exception as e:
            logger.error(f"Telugu direct search error: {e}")
        
        # Strategy 2: Regional search in India with Telugu detection
        regional_params = {
            "api_key": self.api_key,
            "primary_release_date.gte": start_str,
            "primary_release_date.lte": end_str,
            "region": "IN",
            "sort_by": "popularity.desc",
            "include_adult": False,
            "page": 1
        }
        
        try:
            for page in range(1, 3):
                regional_params["page"] = page
                response = await self.http_client.get(
                    f"{self.BASE_URL}/discover/movie",
                    params=regional_params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    for movie in results:
                        is_telugu, confidence = self.telugu_detector.is_telugu_content(movie)
                        if is_telugu:
                            movie['_telugu_source'] = 'regional_detection'
                            movie['_telugu_confidence'] = confidence
                            all_results.append(movie)
                    
                    if not results:
                        break
                        
        except Exception as e:
            logger.error(f"Regional Telugu search error: {e}")
        
        # Strategy 3: Search by Telugu keywords
        telugu_keywords = ['telugu', 'tollywood', 'prabhas', 'mahesh babu', 'allu arjun']
        
        for keyword in telugu_keywords[:2]:  # Limit to prevent timeout
            try:
                search_params = {
                    "api_key": self.api_key,
                    "query": keyword,
                    "primary_release_year": start_date.year,
                    "region": "IN",
                    "include_adult": False,
                    "page": 1
                }
                
                response = await self.http_client.get(
                    f"{self.BASE_URL}/search/movie",
                    params=search_params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    for movie in results:
                        # Check if release date is in our window
                        release_date_str = movie.get("release_date", "")
                        if release_date_str:
                            try:
                                release_date = datetime.strptime(release_date_str, "%Y-%m-%d")
                                if start_date.replace(tzinfo=None) <= release_date <= end_date.replace(tzinfo=None):
                                    movie['_telugu_source'] = f'keyword_{keyword}'
                                    all_results.append(movie)
                            except ValueError:
                                continue
                                
            except Exception as e:
                logger.error(f"Keyword search error for {keyword}: {e}")
        
        # Remove duplicates based on ID
        seen_ids = set()
        unique_results = []
        for movie in all_results:
            movie_id = movie.get("id")
            if movie_id and movie_id not in seen_ids:
                seen_ids.add(movie_id)
                unique_results.append(movie)
        
        logger.info(f"Found {len(unique_results)} Telugu movies for period {start_str} to {end_str}")
        return unique_results
    
    async def get_telugu_tv_priority(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get Telugu TV shows with multiple strategies"""
        all_results = []
        
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Strategy 1: Direct Telugu language search
        telugu_params = {
            "api_key": self.api_key,
            "first_air_date.gte": start_str,
            "first_air_date.lte": end_str,
            "watch_region": region,
            "with_original_language": "te",
            "sort_by": "popularity.desc",
            "include_adult": False,
            "page": 1
        }
        
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/discover/tv",
                params=telugu_params
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                for show in results:
                    show['_telugu_source'] = 'direct_language'
                    all_results.append(show)
                        
        except Exception as e:
            logger.error(f"Telugu TV direct search error: {e}")
        
        # Strategy 2: Regional search with detection
        regional_params = {
            "api_key": self.api_key,
            "first_air_date.gte": start_str,
            "first_air_date.lte": end_str,
            "watch_region": "IN",
            "sort_by": "popularity.desc",
            "include_adult": False,
            "page": 1
        }
        
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/discover/tv",
                params=regional_params
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                for show in results:
                    is_telugu, confidence = self.telugu_detector.is_telugu_content(show)
                    if is_telugu:
                        show['_telugu_source'] = 'regional_detection'
                        show['_telugu_confidence'] = confidence
                        all_results.append(show)
                        
        except Exception as e:
            logger.error(f"Regional Telugu TV search error: {e}")
        
        # Remove duplicates
        seen_ids = set()
        unique_results = []
        for show in all_results:
            show_id = show.get("id")
            if show_id and show_id not in seen_ids:
                seen_ids.add(show_id)
                unique_results.append(show)
        
        logger.info(f"Found {len(unique_results)} Telugu TV shows for period {start_str} to {end_str}")
        return unique_results
    
    async def get_general_content(
        self,
        content_type: str,
        region: str,
        start_date: datetime,
        end_date: datetime,
        max_results: int = 40
    ) -> List[Dict[str, Any]]:
        """Get general content (non-Telugu)"""
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        if content_type == "movie":
            date_param = "primary_release_date"
            discover_endpoint = "discover/movie"
        else:
            date_param = "first_air_date"
            discover_endpoint = "discover/tv"
            region = f"watch_region"
        
        params = {
            "api_key": self.api_key,
            f"{date_param}.gte": start_str,
            f"{date_param}.lte": end_str,
            "sort_by": f"{date_param}.asc,popularity.desc",
            "include_adult": False,
            "page": 1
        }
        
        if content_type == "movie":
            params["region"] = region
        else:
            params["watch_region"] = region
        
        all_results = []
        
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/{discover_endpoint}",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                # Filter by date range and exclude Telugu content already found
                for item in results:
                    date_str = item.get("release_date" if content_type == "movie" else "first_air_date", "")
                    if date_str:
                        try:
                            item_date = datetime.strptime(date_str, "%Y-%m-%d")
                            if start_date.replace(tzinfo=None) <= item_date <= end_date.replace(tzinfo=None):
                                # Skip if it's Telugu content (we handle that separately)
                                is_telugu, _ = self.telugu_detector.is_telugu_content(item)
                                if not is_telugu:
                                    all_results.append(item)
                        except ValueError:
                            continue
                            
        except Exception as e:
            logger.error(f"General content fetch error: {e}")
        
        return all_results[:max_results]


class OptimizedAniListClient:
    """Optimized AniList client for anime content"""
    
    BASE_URL = "https://graphql.anilist.co"
    
    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self.http_client = http_client or httpx.AsyncClient(timeout=8.0)
    
    def _build_optimized_query(self) -> str:
        """Build optimized GraphQL query"""
        return """
        query ($startDate: Int, $endDate: Int, $page: Int) {
            Page(page: $page, perPage: 25) {
                pageInfo {
                    hasNextPage
                }
                media(
                    type: ANIME
                    format_in: [TV, TV_SHORT, ONA, OVA, MOVIE]
                    startDate_greater: $startDate
                    startDate_lesser: $endDate
                    sort: [START_DATE_DESC, POPULARITY_DESC]
                    isAdult: false
                    status_in: [NOT_YET_RELEASED, RELEASING]
                ) {
                    id
                    title {
                        romaji
                        english
                        native
                    }
                    startDate {
                        year
                        month
                        day
                    }
                    format
                    genres
                    popularity
                    averageScore
                    favourites
                    coverImage {
                        large
                    }
                    bannerImage
                    description(asHtml: false)
                    studios {
                        nodes {
                            name
                            isAnimationStudio
                        }
                    }
                }
            }
        }
        """
    
    async def get_upcoming_anime_optimized(
        self,
        start_date: datetime,
        end_date: datetime,
        max_results: int = 25
    ) -> List[Dict[str, Any]]:
        """Get upcoming anime with optimized query"""
        start_int = int(start_date.strftime("%Y%m%d"))
        end_int = int(end_date.strftime("%Y%m%d"))
        
        variables = {
            "startDate": start_int,
            "endDate": end_int,
            "page": 1
        }
        
        try:
            response = await self.http_client.post(
                self.BASE_URL,
                json={
                    "query": self._build_optimized_query(),
                    "variables": variables
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if "errors" in data:
                    logger.error(f"AniList GraphQL errors: {data['errors']}")
                    return []
                
                results = data.get("data", {}).get("Page", {}).get("media", [])
                
                # Validate dates
                valid_results = []
                for anime in results:
                    start_date_info = anime.get("startDate", {})
                    if start_date_info and start_date_info.get("year"):
                        try:
                            anime_start = datetime(
                                start_date_info.get("year"),
                                start_date_info.get("month", 1),
                                start_date_info.get("day", 1)
                            )
                            
                            if start_date.replace(tzinfo=None) <= anime_start <= end_date.replace(tzinfo=None):
                                valid_results.append(anime)
                        except ValueError:
                            continue
                
                return valid_results[:max_results]
                
        except Exception as e:
            logger.error(f"AniList fetch error: {e}")
        
        return []


class TeluguPriorityUpcomingService:
    """
    Enhanced service with aggressive Telugu content discovery
    """
    
    def __init__(
        self,
        tmdb_api_key: str,
        cache_backend: Optional[Any] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        enable_analytics: bool = True
    ):
        self.tmdb_client = TeluguFocusedTMDbClient(tmdb_api_key, http_client)
        self.anilist_client = OptimizedAniListClient(http_client)
        self.cache = cache_backend
        self.http_client = http_client or httpx.AsyncClient(timeout=10.0)
        self.enable_analytics = enable_analytics
        self.telugu_detector = EnhancedTeluguDetector()
    
    def _get_accurate_release_windows(self, user_timezone: str) -> List[ReleaseWindow]:
        """Get accurate 3-month release windows"""
        try:
            tz = pytz.timezone(user_timezone)
        except pytz.UnknownTimeZoneError:
            logger.warning(f"Unknown timezone {user_timezone}, using UTC")
            tz = pytz.UTC
        
        now = datetime.now(tz).replace(tzinfo=None)
        windows = []
        
        for month_offset in range(3):
            if month_offset == 0:
                month_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                target_date = now + timedelta(days=30 * month_offset)
                month_start = target_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            next_month = month_start.replace(day=28) + timedelta(days=4)
            month_end = next_month - timedelta(days=next_month.day)
            month_end = month_end.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            month_name = calendar.month_name[month_start.month]
            
            windows.append(ReleaseWindow(
                start_date=month_start,
                end_date=month_end,
                month_label=f"month_{month_offset}",
                month_name=month_name,
                priority=month_offset + 1,
                is_current=month_offset == 0,
                days_from_now=(month_start - now).days if month_offset > 0 else 0
            ))
        
        return windows
    
    def _parse_telugu_movie(self, item: Dict[str, Any], region: str) -> UpcomingRelease:
        """Parse Telugu movie with enhanced detection"""
        release_date_str = item.get("release_date", "")
        confirmed_date = False
        
        try:
            release_date = datetime.strptime(release_date_str, "%Y-%m-%d")
            confirmed_date = True
        except (ValueError, TypeError):
            release_date = datetime.now() + timedelta(days=30)
            confirmed_date = False
        
        # Enhanced Telugu detection
        is_telugu, confidence_data = self.telugu_detector.is_telugu_content(item)
        telugu_confidence = sum(confidence_data.values()) / 100.0
        
        # Force Telugu language priority
        original_lang = item.get("original_language", "")
        if is_telugu or original_lang == "te":
            languages = ["te"]
            if original_lang and original_lang != "te":
                languages.append(original_lang)
        else:
            languages = [original_lang] if original_lang else ["te"]  # Default to Telugu if detected
            is_telugu = True  # Force Telugu if in Telugu search results
        
        # Image URLs
        poster_path = item.get("poster_path")
        if poster_path and not poster_path.startswith("http"):
            poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        backdrop_path = item.get("backdrop_path")
        if backdrop_path and not backdrop_path.startswith("http"):
            backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        
        release = UpcomingRelease(
            id=f"tmdb_movie_{item['id']}",
            title=item.get("title", ""),
            original_title=item.get("original_title"),
            content_type=ContentType.MOVIE,
            release_date=release_date,
            languages=languages,
            genres=self._get_genre_names(item.get("genre_ids", [])),
            popularity=item.get("popularity", 0),
            vote_count=item.get("vote_count", 0),
            vote_average=item.get("vote_average", 0),
            poster_path=poster_path,
            backdrop_path=backdrop_path,
            overview=item.get("overview"),
            region=region,
            source="tmdb",
            confirmed_release_date=confirmed_date,
            cinema_release=True,
            ott_release=False,
            is_telugu_content=True,  # Force Telugu
            is_telugu_priority=True,  # Force Telugu priority
            is_tollywood=True,
            telugu_confidence=telugu_confidence,
            language_priority=1  # Force top priority
        )
        
        return release
    
    def _parse_telugu_tv(self, item: Dict[str, Any], region: str) -> UpcomingRelease:
        """Parse Telugu TV show with enhanced detection"""
        air_date_str = item.get("first_air_date", "")
        confirmed_date = False
        
        try:
            air_date = datetime.strptime(air_date_str, "%Y-%m-%d")
            confirmed_date = True
        except (ValueError, TypeError):
            air_date = datetime.now() + timedelta(days=30)
            confirmed_date = False
        
        # Enhanced Telugu detection
        is_telugu, confidence_data = self.telugu_detector.is_telugu_content(item)
        telugu_confidence = sum(confidence_data.values()) / 100.0
        
        # Force Telugu language priority
        original_lang = item.get("original_language", "")
        if is_telugu or original_lang == "te":
            languages = ["te"]
            if original_lang and original_lang != "te":
                languages.append(original_lang)
        else:
            languages = [original_lang] if original_lang else ["te"]
            is_telugu = True
        
        # Image URLs
        poster_path = item.get("poster_path")
        if poster_path and not poster_path.startswith("http"):
            poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        backdrop_path = item.get("backdrop_path")
        if backdrop_path and not backdrop_path.startswith("http"):
            backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        
        release = UpcomingRelease(
            id=f"tmdb_tv_{item['id']}",
            title=item.get("name", ""),
            original_title=item.get("original_name"),
            content_type=ContentType.TV_SERIES,
            release_date=air_date,
            languages=languages,
            genres=self._get_genre_names(item.get("genre_ids", [])),
            popularity=item.get("popularity", 0),
            vote_count=item.get("vote_count", 0),
            vote_average=item.get("vote_average", 0),
            poster_path=poster_path,
            backdrop_path=backdrop_path,
            overview=item.get("overview"),
            region=region,
            source="tmdb",
            confirmed_release_date=confirmed_date,
            cinema_release=False,
            ott_release=True,
            ott_platform="Television/Streaming",
            is_telugu_content=True,
            is_telugu_priority=True,
            is_tollywood=True,
            telugu_confidence=telugu_confidence,
            language_priority=1
        )
        
        return release
    
    def _parse_general_content(self, item: Dict[str, Any], region: str, content_type: str) -> UpcomingRelease:
        """Parse general (non-Telugu priority) content"""
        if content_type == "movie":
            date_field = "release_date"
            title_field = "title"
            original_title_field = "original_title"
            id_prefix = "tmdb_movie"
            content_type_enum = ContentType.MOVIE
        else:
            date_field = "first_air_date"
            title_field = "name"
            original_title_field = "original_name"
            id_prefix = "tmdb_tv"
            content_type_enum = ContentType.TV_SERIES
        
        date_str = item.get(date_field, "")
        confirmed_date = False
        
        try:
            release_date = datetime.strptime(date_str, "%Y-%m-%d")
            confirmed_date = True
        except (ValueError, TypeError):
            release_date = datetime.now() + timedelta(days=30)
            confirmed_date = False
        
        # Check if it's actually Telugu content
        is_telugu, confidence_data = self.telugu_detector.is_telugu_content(item)
        
        original_lang = item.get("original_language", "")
        languages = [original_lang] if original_lang else ["en"]
        
        if is_telugu:
            if "te" not in languages:
                languages.insert(0, "te")
        
        # Image URLs
        poster_path = item.get("poster_path")
        if poster_path and not poster_path.startswith("http"):
            poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        backdrop_path = item.get("backdrop_path")
        if backdrop_path and not backdrop_path.startswith("http"):
            backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        
        release = UpcomingRelease(
            id=f"{id_prefix}_{item['id']}",
            title=item.get(title_field, ""),
            original_title=item.get(original_title_field),
            content_type=content_type_enum,
            release_date=release_date,
            languages=languages,
            genres=self._get_genre_names(item.get("genre_ids", [])),
            popularity=item.get("popularity", 0),
            vote_count=item.get("vote_count", 0),
            vote_average=item.get("vote_average", 0),
            poster_path=poster_path,
            backdrop_path=backdrop_path,
            overview=item.get("overview"),
            region=region,
            source="tmdb",
            confirmed_release_date=confirmed_date,
            cinema_release=content_type == "movie",
            ott_release=content_type == "tv",
            ott_platform="Television/Streaming" if content_type == "tv" else None,
            is_telugu_content=is_telugu,
            is_telugu_priority=is_telugu,
            is_tollywood=is_telugu
        )
        
        return release
    
    def _parse_anime_optimized(self, item: Dict[str, Any]) -> Optional[UpcomingRelease]:
        """Parse anime data with optimized processing"""
        start_date_info = item.get("startDate", {})
        if not start_date_info or not start_date_info.get("year"):
            return None
        
        try:
            release_date = datetime(
                start_date_info.get("year"),
                start_date_info.get("month", 1),
                start_date_info.get("day", 1)
            )
            confirmed_date = bool(start_date_info.get("day"))
        except (ValueError, TypeError):
            return None
        
        title_info = item.get("title", {})
        title = title_info.get("english") or title_info.get("romaji", "")
        
        if not title:
            return None
        
        cover_image = item.get("coverImage", {})
        poster_url = cover_image.get("large")
        
        studios = item.get("studios", {}).get("nodes", [])
        studio_name = None
        for studio in studios:
            if studio.get("isAnimationStudio"):
                studio_name = studio.get("name")
                break
        
        release = UpcomingRelease(
            id=f"anilist_{item['id']}",
            title=title,
            original_title=title_info.get("native"),
            content_type=ContentType.ANIME,
            release_date=release_date,
            languages=["ja"],
            genres=item.get("genres", []),
            popularity=item.get("popularity", 0),
            vote_count=item.get("favourites", 0),
            vote_average=(item.get("averageScore", 0) / 10) if item.get("averageScore") else 0,
            poster_path=poster_url,
            backdrop_path=item.get("bannerImage"),
            overview=item.get("description"),
            source="anilist",
            confirmed_release_date=confirmed_date,
            cinema_release=item.get("format") == "MOVIE",
            ott_release=True,
            ott_platform="Anime Platforms",
            studio=studio_name
        )
        
        return release
    
    def _get_genre_names(self, genre_ids: List[int]) -> List[str]:
        """Convert TMDb genre IDs to names"""
        genre_map = {
            28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
            80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
            14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
            9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
            10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
            10759: "Action & Adventure", 10762: "Kids", 10763: "News",
            10764: "Reality", 10765: "Sci-Fi & Fantasy", 10766: "Soap",
            10767: "Talk", 10768: "War & Politics"
        }
        return [genre_map.get(gid, "Unknown") for gid in genre_ids if gid in genre_map]
    
    def _sort_telugu_first(self, releases: List[UpcomingRelease]) -> List[UpcomingRelease]:
        """Sort releases with STRICT Telugu priority"""
        def sort_key(release):
            # Telugu content ALWAYS comes first (priority 0)
            if release.is_telugu_content or release.is_telugu_priority:
                telugu_priority = 0
            else:
                telugu_priority = release.language_priority
            
            return (
                telugu_priority,
                release.release_date,
                -release.data_confidence,
                -release.popularity
            )
        
        sorted_releases = sorted(releases, key=sort_key)
        
        # Log Telugu content found
        telugu_count = sum(1 for r in sorted_releases if r.is_telugu_content)
        logger.info(f"Sorted releases: {telugu_count} Telugu content out of {len(sorted_releases)} total")
        
        return sorted_releases
    
    async def get_upcoming_releases(
        self,
        region: str = "IN",
        timezone_name: str = "Asia/Kolkata",
        categories: Optional[List[str]] = None,
        use_cache: bool = True,
        include_analytics: bool = True
    ) -> Dict[str, Any]:
        """
        Get upcoming releases with AGGRESSIVE Telugu priority
        """
        if categories is None:
            categories = ["movies", "tv", "anime"]
        
        # Check cache
        cache_key = self._generate_cache_key(region, timezone_name, categories)
        if use_cache and self.cache:
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.info(f"Cache hit for Telugu priority upcoming releases")
                return cached_data
        
        # Get release windows
        windows = self._get_accurate_release_windows(timezone_name)
        
        # Initialize results
        results = {
            "movies": [],
            "tv_series": [],
            "anime": [],
            "metadata": {
                "region": region,
                "timezone": timezone_name,
                "language_priorities": ["telugu", "english", "hindi", "malayalam", "kannada", "tamil"],
                "telugu_priority": True,
                "telugu_aggressive_search": True,
                "accuracy_level": "Telugu Focused",
                "windows": [
                    {
                        "label": w.month_label,
                        "month_name": w.month_name,
                        "start": w.start_date.isoformat(),
                        "end": w.end_date.isoformat(),
                        "is_current": w.is_current,
                        "days_from_now": w.days_from_now
                    }
                    for w in windows
                ],
                "fetched_at": datetime.now().isoformat()
            }
        }
        
        # Fetch data with Telugu priority
        try:
            for window in windows:
                # 1. PRIORITY: Telugu Movies
                if "movies" in categories:
                    try:
                        telugu_movies = await asyncio.wait_for(
                            self.tmdb_client.get_telugu_movies_priority(
                                region, window.start_date, window.end_date
                            ), timeout=15.0
                        )
                        
                        for movie in telugu_movies:
                            try:
                                release = self._parse_telugu_movie(movie, region)
                                results["movies"].append(release)
                            except Exception as e:
                                logger.warning(f"Error parsing Telugu movie: {e}")
                    
                        logger.info(f"Found {len(telugu_movies)} Telugu movies for {window.month_name}")
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout fetching Telugu movies for {window.month_name}")
                    except Exception as e:
                        logger.error(f"Error fetching Telugu movies: {e}")
                
                # 2. PRIORITY: Telugu TV Shows
                if "tv" in categories:
                    try:
                        telugu_tv = await asyncio.wait_for(
                            self.tmdb_client.get_telugu_tv_priority(
                                region, window.start_date, window.end_date
                            ), timeout=10.0
                        )
                        
                        for show in telugu_tv:
                            try:
                                release = self._parse_telugu_tv(show, region)
                                results["tv_series"].append(release)
                            except Exception as e:
                                logger.warning(f"Error parsing Telugu TV show: {e}")
                        
                        logger.info(f"Found {len(telugu_tv)} Telugu TV shows for {window.month_name}")
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout fetching Telugu TV for {window.month_name}")
                    except Exception as e:
                        logger.error(f"Error fetching Telugu TV: {e}")
                
                # 3. General content (limited)
                if "movies" in categories:
                    try:
                        general_movies = await asyncio.wait_for(
                            self.tmdb_client.get_general_content(
                                "movie", region, window.start_date, window.end_date, max_results=20
                            ), timeout=8.0
                        )
                        
                        for movie in general_movies:
                            try:
                                release = self._parse_general_content(movie, region, "movie")
                                results["movies"].append(release)
                            except Exception as e:
                                logger.warning(f"Error parsing general movie: {e}")
                                
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout fetching general movies for {window.month_name}")
                    except Exception as e:
                        logger.error(f"Error fetching general movies: {e}")
                
                if "tv" in categories:
                    try:
                        general_tv = await asyncio.wait_for(
                            self.tmdb_client.get_general_content(
                                "tv", region, window.start_date, window.end_date, max_results=15
                            ), timeout=8.0
                        )
                        
                        for show in general_tv:
                            try:
                                release = self._parse_general_content(show, region, "tv")
                                results["tv_series"].append(release)
                            except Exception as e:
                                logger.warning(f"Error parsing general TV show: {e}")
                                
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout fetching general TV for {window.month_name}")
                    except Exception as e:
                        logger.error(f"Error fetching general TV: {e}")
                
                # 4. Anime
                if "anime" in categories:
                    try:
                        anime_list = await asyncio.wait_for(
                            self.anilist_client.get_upcoming_anime_optimized(
                                window.start_date, window.end_date, max_results=15
                            ), timeout=8.0
                        )
                        
                        for anime in anime_list:
                            try:
                                release = self._parse_anime_optimized(anime)
                                if release:
                                    results["anime"].append(release)
                            except Exception as e:
                                logger.warning(f"Error parsing anime: {e}")
                                
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout fetching anime for {window.month_name}")
                    except Exception as e:
                        logger.error(f"Error fetching anime: {e}")
            
            # Remove duplicates and sort with Telugu priority
            for category in ["movies", "tv_series", "anime"]:
                seen_ids = set()
                unique_releases = []
                for release in results[category]:
                    if release.id not in seen_ids:
                        seen_ids.add(release.id)
                        unique_releases.append(release)
                
                # Sort with STRICT Telugu priority
                results[category] = self._sort_telugu_first(unique_releases)
            
            # Calculate statistics
            total_releases = len(results["movies"]) + len(results["tv_series"]) + len(results["anime"])
            telugu_movies = sum(1 for r in results["movies"] if r.is_telugu_content)
            telugu_tv = sum(1 for r in results["tv_series"] if r.is_telugu_content)
            total_telugu = telugu_movies + telugu_tv
            confirmed_dates = sum(1 for r in results["movies"] + results["tv_series"] + results["anime"] if r.confirmed_release_date)
            high_confidence = sum(1 for r in results["movies"] + results["tv_series"] + results["anime"] if r.data_confidence >= 0.8)
            
            results["metadata"]["statistics"] = {
                "total_releases": total_releases,
                "movies": len(results["movies"]),
                "tv_series": len(results["tv_series"]),
                "anime": len(results["anime"]),
                "telugu_releases": total_telugu,
                "telugu_movies": telugu_movies,
                "telugu_tv_series": telugu_tv,
                "telugu_percentage": round((total_telugu / max(1, total_releases)) * 100, 1),
                "confirmed_dates": confirmed_dates,
                "confirmed_percentage": round((confirmed_dates / max(1, total_releases)) * 100, 1),
                "high_confidence_releases": high_confidence,
                "confidence_percentage": round((high_confidence / max(1, total_releases)) * 100, 1),
                "data_sources": ["tmdb", "anilist"],
                "telugu_search_strategies": [
                    "direct_language_search",
                    "regional_detection", 
                    "keyword_search",
                    "studio_detection"
                ]
            }
            
            # Convert to dict for JSON serialization
            results["movies"] = [r.to_dict() for r in results["movies"]]
            results["tv_series"] = [r.to_dict() for r in results["tv_series"]]
            results["anime"] = [r.to_dict() for r in results["anime"]]
            
            # Cache results
            if use_cache and self.cache:
                await self._save_to_cache(cache_key, results, ttl=1800)
            
            logger.info(f"Successfully fetched {total_releases} releases with {total_telugu} Telugu content")
            return results
            
        except Exception as e:
            logger.error(f"Error in Telugu priority upcoming service: {e}")
            return {
                "movies": [],
                "tv_series": [],
                "anime": [],
                "metadata": {
                    "error": str(e),
                    "region": region,
                    "timezone": timezone_name,
                    "fetched_at": datetime.now().isoformat()
                }
            }
    
    def _generate_cache_key(self, region: str, timezone_name: str, categories: List[str]) -> str:
        """Generate cache key"""
        components = [
            "telugu_priority_upcoming",
            region,
            timezone_name,
            "_".join(sorted(categories)),
            datetime.now().strftime("%Y%m%d%H")
        ]
        return hashlib.md5(":".join(components).encode()).hexdigest()
    
    async def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get from cache"""
        if not self.cache:
            return None
        
        try:
            cached = self.cache.get(f"telugu_priority_upcoming:{key}")
            if cached:
                data = json.loads(cached) if isinstance(cached, str) else cached
                return data
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        
        return None
    
    async def _save_to_cache(self, key: str, data: Dict[str, Any], ttl: int = 1800):
        """Save to cache"""
        if not self.cache:
            return
        
        try:
            self.cache.set(
                f"telugu_priority_upcoming:{key}",
                json.dumps(data, default=str),
                timeout=ttl
            )
            logger.info(f"Cached Telugu priority upcoming releases")
        except Exception as e:
            logger.error(f"Cache save error: {e}")
    
    async def close(self):
        """Clean up resources"""
        if self.http_client:
            await self.http_client.aclose()


# Alias for backward compatibility
UpcomingContentService = TeluguPriorityUpcomingService
AccurateUpcomingContentService = TeluguPriorityUpcomingService
OptimizedUpcomingContentService = TeluguPriorityUpcomingService