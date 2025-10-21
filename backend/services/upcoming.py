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
    
    # Telugu specific fields
    is_telugu_content: bool = False
    is_tollywood: bool = False
    telugu_title: Optional[str] = None
    production_house: Optional[str] = None
    
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
        
        # Check languages
        for lang in self.languages:
            if lang.lower() in telugu_indicators:
                self.is_telugu_content = True
                self.is_tollywood = True
                break
        
        # Check title for Telugu patterns
        title_text = f"{self.title} {self.original_title or ''}".lower()
        telugu_keywords = ['telugu', 'tollywood', 'andhra', 'telangana', 'hyderabad']
        
        for keyword in telugu_keywords:
            if keyword in title_text:
                self.is_telugu_content = True
                break
    
    def _calculate_data_confidence(self):
        """Calculate data confidence score"""
        confidence = 0.0
        
        # Release date confidence
        if self.confirmed_release_date:
            confidence += 0.3
        elif self.release_date:
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
        if self.runtime_minutes:
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
                # Convert to naive datetime
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


class AccurateTeluguDetector:
    """Enhanced Telugu content detection system"""
    
    TELUGU_STUDIOS = [
        'AVM Productions', 'Vyjayanthi Movies', 'Mythri Movie Makers',
        'Geetha Arts', 'Sri Venkateswara Creations', 'Konidela Production Company',
        'UV Creations', 'Sithara Entertainments', 'Haarika & Hassine Creations',
        'GA2 Pictures', 'People Media Factory', 'Asian Cinemas',
        'Suresh Productions', 'Annapurna Studios', 'Bhavya Creations'
    ]
    
    TELUGU_KEYWORDS = [
        'telugu', 'tollywood', 'andhra pradesh', 'telangana', 'hyderabad',
        'vizag', 'vijayawada', 'guntur', 'tirupati', 'warangal'
    ]
    
    TELUGU_NAMES_PATTERNS = [
        'charan', 'prabhas', 'mahesh', 'allu', 'trivikram', 'rajamouli',
        'koratala', 'sukumar', 'parasuram', 'harish shankar', 'boyapati'
    ]
    
    @classmethod
    def is_telugu_content(cls, item: Dict[str, Any], detailed: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """Enhanced Telugu content detection with confidence scoring"""
        confidence_factors = {
            'language': 0,
            'title': 0,
            'studio': 0,
            'cast_crew': 0,
            'keywords': 0
        }
        
        # Language check (highest confidence)
        original_language = item.get("original_language", "").lower()
        if original_language in ['te', 'telugu']:
            confidence_factors['language'] = 100
            return True, confidence_factors
        
        # Title analysis
        title = f"{item.get('title', '')} {item.get('original_title', '')} {item.get('name', '')}".lower()
        
        for keyword in cls.TELUGU_KEYWORDS:
            if keyword in title:
                confidence_factors['title'] += 30
        
        for pattern in cls.TELUGU_NAMES_PATTERNS:
            if pattern in title:
                confidence_factors['cast_crew'] += 20
        
        # Studio/Production analysis (if available)
        production_companies = item.get('production_companies', [])
        for company in production_companies:
            company_name = company.get('name', '') if isinstance(company, dict) else str(company)
            if any(studio.lower() in company_name.lower() for studio in cls.TELUGU_STUDIOS):
                confidence_factors['studio'] = 50
                break
        
        # Overall confidence
        total_confidence = sum(confidence_factors.values())
        is_telugu = total_confidence >= 30
        
        return is_telugu, confidence_factors


class OptimizedTMDbClient:
    """Optimized TMDb API client with better timeout handling"""
    
    BASE_URL = "https://api.themoviedb.org/3"
    
    def __init__(self, api_key: str, http_client: Optional[httpx.AsyncClient] = None):
        self.api_key = api_key
        self.http_client = http_client or httpx.AsyncClient(timeout=8.0)
        self.telugu_detector = AccurateTeluguDetector()
    
    async def get_upcoming_movies_optimized(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime,
        max_results: int = 60
    ) -> List[Dict[str, Any]]:
        """Get upcoming movies with optimized API calls"""
        all_results = []
        
        # Convert to naive datetime strings for API
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Single optimized discovery call
        params = {
            "api_key": self.api_key,
            "primary_release_date.gte": start_str,
            "primary_release_date.lte": end_str,
            "region": region,
            "sort_by": "primary_release_date.asc,popularity.desc",
            "with_release_type": "2|3|4|5",
            "include_adult": False,
            "page": 1
        }
        
        try:
            # Fetch first 2 pages only for performance
            for page in range(1, 3):
                params["page"] = page
                
                response = await self.http_client.get(
                    f"{self.BASE_URL}/discover/movie",
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    # Filter by exact date range (all datetimes are naive now)
                    filtered_results = []
                    for movie in results:
                        release_date_str = movie.get("release_date", "")
                        if release_date_str:
                            try:
                                # Parse as naive datetime
                                release_date = datetime.strptime(release_date_str, "%Y-%m-%d")
                                # Compare naive datetimes
                                if start_date.replace(tzinfo=None) <= release_date <= end_date.replace(tzinfo=None):
                                    filtered_results.append(movie)
                            except ValueError:
                                continue
                    
                    all_results.extend(filtered_results)
                    
                    if len(all_results) >= max_results:
                        break
                        
                    if not results or page >= data.get("total_pages", 1):
                        break
                else:
                    logger.warning(f"TMDb API error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"TMDb movie fetch error: {e}")
        
        return all_results[:max_results]
    
    async def get_upcoming_tv_optimized(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime,
        max_results: int = 40
    ) -> List[Dict[str, Any]]:
        """Get upcoming TV shows with optimized API calls"""
        all_results = []
        
        # Convert to naive datetime strings
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        params = {
            "api_key": self.api_key,
            "first_air_date.gte": start_str,
            "first_air_date.lte": end_str,
            "watch_region": region,
            "sort_by": "first_air_date.asc,popularity.desc",
            "include_adult": False,
            "page": 1
        }
        
        try:
            # Single page for TV shows
            response = await self.http_client.get(
                f"{self.BASE_URL}/discover/tv",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                # Filter by date range
                for show in results:
                    air_date_str = show.get("first_air_date", "")
                    if air_date_str:
                        try:
                            air_date = datetime.strptime(air_date_str, "%Y-%m-%d")
                            if start_date.replace(tzinfo=None) <= air_date <= end_date.replace(tzinfo=None):
                                all_results.append(show)
                        except ValueError:
                            continue
                            
        except Exception as e:
            logger.error(f"TMDb TV fetch error: {e}")
        
        return all_results[:max_results]


class OptimizedAniListClient:
    """Optimized AniList client for better performance"""
    
    BASE_URL = "https://graphql.anilist.co"
    
    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self.http_client = http_client or httpx.AsyncClient(timeout=8.0)
    
    def _build_optimized_query(self) -> str:
        """Build optimized GraphQL query"""
        return """
        query ($startDate: Int, $endDate: Int, $page: Int) {
            Page(page: $page, perPage: 30) {
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
                    staff(sort: [ROLE, RELEVANCE]) {
                        edges {
                            role
                            node {
                                name {
                                    full
                                }
                            }
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
        max_results: int = 30
    ) -> List[Dict[str, Any]]:
        """Get upcoming anime with optimized query"""
        # Convert to AniList date format
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


class OptimizedUpcomingContentService:
    """
    Optimized service for fast upcoming releases with Telugu priority
    """
    
    def __init__(
        self,
        tmdb_api_key: str,
        cache_backend: Optional[Any] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        enable_analytics: bool = True
    ):
        self.tmdb_client = OptimizedTMDbClient(tmdb_api_key, http_client)
        self.anilist_client = OptimizedAniListClient(http_client)
        self.cache = cache_backend
        self.http_client = http_client or httpx.AsyncClient(timeout=8.0)
        self.enable_analytics = enable_analytics
        self.telugu_detector = AccurateTeluguDetector()
    
    def _get_accurate_release_windows(self, user_timezone: str) -> List[ReleaseWindow]:
        """Get accurate 3-month release windows"""
        try:
            tz = pytz.timezone(user_timezone)
        except pytz.UnknownTimeZoneError:
            logger.warning(f"Unknown timezone {user_timezone}, using UTC")
            tz = pytz.UTC
        
        # Get current time in timezone and convert to naive
        now = datetime.now(tz).replace(tzinfo=None)
        
        windows = []
        
        for month_offset in range(3):
            if month_offset == 0:
                # Current month: from today
                month_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                # Future months: from 1st day
                target_date = now + timedelta(days=30 * month_offset)
                month_start = target_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # End of month
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
    
    def _parse_movie_optimized(self, item: Dict[str, Any], region: str) -> UpcomingRelease:
        """Parse movie data with optimized processing"""
        # Parse release date
        release_date_str = item.get("release_date", "")
        confirmed_date = False
        
        try:
            release_date = datetime.strptime(release_date_str, "%Y-%m-%d")
            confirmed_date = True
        except (ValueError, TypeError):
            release_date = datetime.now() + timedelta(days=30)
            confirmed_date = False
        
        # Telugu detection
        is_telugu, _ = self.telugu_detector.is_telugu_content(item)
        
        # Language determination
        original_lang = item.get("original_language", "")
        languages = []
        
        if is_telugu or original_lang == "te":
            languages = ["te"]
            if original_lang and original_lang != "te":
                languages.append(original_lang)
        else:
            languages = [original_lang] if original_lang else ["en"]
        
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
            is_telugu_content=is_telugu,
            is_tollywood=is_telugu
        )
        
        return release
    
    def _parse_tv_optimized(self, item: Dict[str, Any], region: str) -> UpcomingRelease:
        """Parse TV show data with optimized processing"""
        # Parse air date
        air_date_str = item.get("first_air_date", "")
        confirmed_date = False
        
        try:
            air_date = datetime.strptime(air_date_str, "%Y-%m-%d")
            confirmed_date = True
        except (ValueError, TypeError):
            air_date = datetime.now() + timedelta(days=30)
            confirmed_date = False
        
        # Telugu detection
        is_telugu, _ = self.telugu_detector.is_telugu_content(item)
        
        # Language determination
        original_lang = item.get("original_language", "")
        languages = []
        
        if is_telugu or original_lang == "te":
            languages = ["te"]
            if original_lang and original_lang != "te":
                languages.append(original_lang)
        else:
            languages = [original_lang] if original_lang else ["en"]
        
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
            is_telugu_content=is_telugu,
            is_tollywood=is_telugu
        )
        
        return release
    
    def _parse_anime_optimized(self, item: Dict[str, Any]) -> Optional[UpcomingRelease]:
        """Parse anime data with optimized processing"""
        # Parse start date
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
        
        # Title parsing
        title_info = item.get("title", {})
        title = title_info.get("english") or title_info.get("romaji", "")
        
        if not title:
            return None
        
        # Image and studio info
        cover_image = item.get("coverImage", {})
        poster_url = cover_image.get("large")
        
        studios = item.get("studios", {}).get("nodes", [])
        studio_name = None
        for studio in studios:
            if studio.get("isAnimationStudio"):
                studio_name = studio.get("name")
                break
        
        # Director
        director = None
        staff = item.get("staff", {}).get("edges", [])
        for staff_member in staff:
            if "director" in staff_member.get("role", "").lower():
                director = staff_member.get("node", {}).get("name", {}).get("full")
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
            director=director,
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
    
    def _sort_by_telugu_priority_and_date(self, releases: List[UpcomingRelease]) -> List[UpcomingRelease]:
        """Sort releases with Telugu priority first, then by date"""
        def sort_key(release):
            # Telugu content gets highest priority (0)
            language_priority = 0 if release.is_telugu_content else release.language_priority
            
            return (
                language_priority,
                release.release_date,
                -release.data_confidence,
                -release.popularity
            )
        
        return sorted(releases, key=sort_key)
    
    async def get_upcoming_releases(
        self,
        region: str = "IN",
        timezone_name: str = "Asia/Kolkata",
        categories: Optional[List[str]] = None,
        use_cache: bool = True,
        include_analytics: bool = True
    ) -> Dict[str, Any]:
        """
        Get optimized upcoming releases for next 3 months
        """
        if categories is None:
            categories = ["movies", "tv", "anime"]
        
        # Check cache first
        cache_key = self._generate_cache_key(region, timezone_name, categories)
        if use_cache and self.cache:
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.info(f"Cache hit for optimized upcoming releases")
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
                "accuracy_level": "Optimized",
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
        
        # Fetch data concurrently with timeout
        try:
            fetch_tasks = []
            
            if "movies" in categories:
                for window in windows:
                    fetch_tasks.append(("movies", self.tmdb_client.get_upcoming_movies_optimized(
                        region, window.start_date, window.end_date
                    )))
            
            if "tv" in categories:
                for window in windows:
                    fetch_tasks.append(("tv_series", self.tmdb_client.get_upcoming_tv_optimized(
                        region, window.start_date, window.end_date
                    )))
            
            if "anime" in categories:
                for window in windows:
                    fetch_tasks.append(("anime", self.anilist_client.get_upcoming_anime_optimized(
                        window.start_date, window.end_date
                    )))
            
            # Execute with timeout
            for category, task in fetch_tasks:
                try:
                    items = await asyncio.wait_for(task, timeout=10.0)
                    
                    for item in items:
                        try:
                            if category == "movies":
                                release = self._parse_movie_optimized(item, region)
                                results["movies"].append(release)
                            elif category == "tv_series":
                                release = self._parse_tv_optimized(item, region)
                                results["tv_series"].append(release)
                            elif category == "anime":
                                release = self._parse_anime_optimized(item)
                                if release:
                                    results["anime"].append(release)
                        except Exception as e:
                            logger.warning(f"Error parsing {category} item: {e}")
                            
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout fetching {category}")
                except Exception as e:
                    logger.error(f"Error fetching {category}: {e}")
            
            # Remove duplicates and sort
            seen_ids = set()
            for category in ["movies", "tv_series", "anime"]:
                unique_releases = []
                for release in results[category]:
                    if release.id not in seen_ids:
                        seen_ids.add(release.id)
                        unique_releases.append(release)
                
                # Sort with Telugu priority
                results[category] = self._sort_by_telugu_priority_and_date(unique_releases)
            
            # Calculate statistics
            total_releases = len(results["movies"]) + len(results["tv_series"]) + len(results["anime"])
            telugu_releases = sum(1 for r in results["movies"] + results["tv_series"] if r.is_telugu_content)
            confirmed_dates = sum(1 for r in results["movies"] + results["tv_series"] + results["anime"] if r.confirmed_release_date)
            high_confidence = sum(1 for r in results["movies"] + results["tv_series"] + results["anime"] if r.data_confidence >= 0.8)
            
            results["metadata"]["statistics"] = {
                "total_releases": total_releases,
                "movies": len(results["movies"]),
                "tv_series": len(results["tv_series"]),
                "anime": len(results["anime"]),
                "telugu_releases": telugu_releases,
                "telugu_percentage": round((telugu_releases / max(1, total_releases)) * 100, 1),
                "confirmed_dates": confirmed_dates,
                "confirmed_percentage": round((confirmed_dates / max(1, total_releases)) * 100, 1),
                "high_confidence_releases": high_confidence,
                "confidence_percentage": round((high_confidence / max(1, total_releases)) * 100, 1),
                "data_sources": ["tmdb", "anilist"],
                "optimization_applied": True
            }
            
            # Convert to dict for JSON serialization
            results["movies"] = [r.to_dict() for r in results["movies"]]
            results["tv_series"] = [r.to_dict() for r in results["tv_series"]]
            results["anime"] = [r.to_dict() for r in results["anime"]]
            
            # Cache results
            if use_cache and self.cache:
                await self._save_to_cache(cache_key, results, ttl=1800)  # 30 min cache
            
            logger.info(f"Successfully fetched {total_releases} upcoming releases")
            return results
            
        except Exception as e:
            logger.error(f"Error in optimized upcoming service: {e}")
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
            "optimized_upcoming",
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
            cached = self.cache.get(f"optimized_upcoming:{key}")
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
                f"optimized_upcoming:{key}",
                json.dumps(data, default=str),
                timeout=ttl
            )
            logger.info(f"Cached optimized upcoming releases")
        except Exception as e:
            logger.error(f"Cache save error: {e}")
    
    async def close(self):
        """Clean up resources"""
        if self.http_client:
            await self.http_client.aclose()


# Alias for backward compatibility
UpcomingContentService = OptimizedUpcomingContentService
AccurateUpcomingContentService = OptimizedUpcomingContentService