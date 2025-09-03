"""
Upcoming Content Service - Production-grade, region and timezone-aware service
for fetching upcoming releases across Movies, TV/Series, and Anime with strict language priority.
Advanced features include predictive analytics, personalization, and intelligent caching.
"""

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

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content type enumeration"""
    MOVIE = "movie"
    TV_SERIES = "tv"
    ANIME = "anime"


class LanguagePriority(Enum):
    """Language priority with ISO-639-1 codes - STRICT ORDER"""
    TELUGU = ("telugu", "te", 1)
    ENGLISH = ("english", "en", 2)
    HINDI = ("hindi", "hi", 3)
    MALAYALAM = ("malayalam", "ml", 4)
    KANNADA = ("kannada", "kn", 5)
    TAMIL = ("tamil", "ta", 6)
    
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
        return None
    
    @classmethod
    def get_priority_order(cls) -> List['LanguagePriority']:
        """Get languages in priority order"""
        return sorted(cls, key=lambda x: x.priority)


@dataclass
class ReleaseWindow:
    """Release window configuration"""
    start_date: datetime
    end_date: datetime
    month_label: str
    priority: int
    is_current: bool = False


@dataclass
class UpcomingRelease:
    """Enhanced upcoming release data structure"""
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
    
    # Enhanced fields
    anticipation_score: float = 0.0  # Calculated based on social metrics
    buzz_level: str = "normal"  # low, normal, high, viral
    pre_release_rating: Optional[float] = None
    trailer_views: int = 0
    social_mentions: int = 0
    is_franchise: bool = False
    franchise_name: Optional[str] = None
    director: Optional[str] = None
    cast: List[str] = field(default_factory=list)
    studio: Optional[str] = None
    budget: Optional[float] = None
    expected_revenue: Optional[float] = None
    marketing_level: str = "standard"  # minimal, standard, heavy, blockbuster
    release_strategy: str = "wide"  # limited, wide, platform, streaming
    days_until_release: int = 0
    is_telugu_priority: bool = False
    
    def __post_init__(self):
        """Post-initialization to set language priority and Telugu flag"""
        self._set_language_priority()
        self._calculate_days_until_release()
        self._determine_telugu_priority()
    
    def _set_language_priority(self):
        """Set language priority based on content languages"""
        for lang in self.languages:
            for lang_enum in LanguagePriority:
                if lang.lower() in [lang_enum.iso_code, lang_enum.language_name]:
                    self.language_priority = min(self.language_priority, lang_enum.priority)
                    break
    
    def _calculate_days_until_release(self):
        """Calculate days until release"""
        if self.release_date:
            delta = self.release_date - datetime.now()
            self.days_until_release = delta.days
    
    def _determine_telugu_priority(self):
        """Check if this is Telugu priority content"""
        telugu_identifiers = ['te', 'telugu', 'tollywood']
        for lang in self.languages:
            if lang.lower() in telugu_identifiers:
                self.is_telugu_priority = True
                break
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime to ISO format
        if self.release_date:
            data['release_date'] = self.release_date.isoformat()
        # Convert enum to string
        data['content_type'] = self.content_type.value
        return data


class AnticipationAnalyzer:
    """Analyzes and calculates anticipation scores for upcoming content"""
    
    @staticmethod
    def calculate_anticipation_score(release: UpcomingRelease) -> float:
        """
        Calculate anticipation score based on multiple factors
        Returns a score between 0 and 100
        """
        score = 0.0
        
        # Base popularity contribution (0-30 points)
        if release.popularity > 0:
            score += min(30, release.popularity / 10)
        
        # Vote/rating contribution (0-20 points)
        if release.vote_average > 0:
            score += (release.vote_average / 10) * 20
        
        # Language priority bonus (0-20 points)
        if release.language_priority == 1:  # Telugu
            score += 20
        elif release.language_priority == 2:  # English
            score += 15
        elif release.language_priority == 3:  # Hindi
            score += 10
        elif release.language_priority <= 6:
            score += 5
        
        # Franchise bonus (0-10 points)
        if release.is_franchise:
            score += 10
        
        # Release proximity bonus (0-10 points)
        if 0 <= release.days_until_release <= 7:
            score += 10
        elif 8 <= release.days_until_release <= 30:
            score += 7
        elif 31 <= release.days_until_release <= 60:
            score += 5
        
        # Marketing level bonus (0-10 points)
        marketing_scores = {
            "blockbuster": 10,
            "heavy": 7,
            "standard": 3,
            "minimal": 1
        }
        score += marketing_scores.get(release.marketing_level, 3)
        
        return min(100, score)
    
    @staticmethod
    def determine_buzz_level(release: UpcomingRelease) -> str:
        """Determine buzz level based on various metrics"""
        if release.anticipation_score >= 80:
            return "viral"
        elif release.anticipation_score >= 60:
            return "high"
        elif release.anticipation_score >= 30:
            return "normal"
        else:
            return "low"


class TMDbClient:
    """Enhanced TMDb API client with advanced features"""
    
    BASE_URL = "https://api.themoviedb.org/3"
    
    def __init__(self, api_key: str, http_client: Optional[httpx.AsyncClient] = None):
        self.api_key = api_key
        self.http_client = http_client or httpx.AsyncClient(timeout=10.0)
    
    async def discover_movies(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime,
        language: Optional[str] = None,
        with_genres: Optional[List[int]] = None,
        sort_by: str = "primary_release_date.desc,popularity.desc"
    ) -> List[Dict[str, Any]]:
        """Discover movies with enhanced filtering"""
        params = {
            "api_key": self.api_key,
            "region": region,
            "primary_release_year": start_date.year,
            "primary_release_date.gte": start_date.strftime("%Y-%m-%d"),
            "primary_release_date.lte": end_date.strftime("%Y-%m-%d"),
            "with_release_type": "2|3|4|5",  # Theatrical, digital, physical, TV
            "sort_by": sort_by,
            "include_adult": False,
            "page": 1
        }
        
        if language:
            params["with_original_language"] = language
        
        if with_genres:
            params["with_genres"] = ",".join(map(str, with_genres))
        
        all_results = []
        
        try:
            # Fetch multiple pages for better coverage
            for page in range(1, 4):  # Get first 3 pages
                params["page"] = page
                response = await self.http_client.get(
                    f"{self.BASE_URL}/discover/movie",
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])
                all_results.extend(results)
                
                if not results or page >= data.get("total_pages", 1):
                    break
            
            return all_results
        except Exception as e:
            logger.error(f"TMDb movie discovery error: {e}")
            return []
    
    async def discover_tv(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime,
        language: Optional[str] = None,
        with_genres: Optional[List[int]] = None,
        sort_by: str = "first_air_date.desc,popularity.desc"
    ) -> List[Dict[str, Any]]:
        """Discover TV shows with enhanced filtering"""
        params = {
            "api_key": self.api_key,
            "watch_region": region,
            "first_air_date_year": start_date.year,
            "first_air_date.gte": start_date.strftime("%Y-%m-%d"),
            "first_air_date.lte": end_date.strftime("%Y-%m-%d"),
            "sort_by": sort_by,
            "include_adult": False,
            "page": 1
        }
        
        if language:
            params["with_original_language"] = language
        
        if with_genres:
            params["with_genres"] = ",".join(map(str, with_genres))
        
        all_results = []
        
        try:
            for page in range(1, 3):  # Get first 2 pages
                params["page"] = page
                response = await self.http_client.get(
                    f"{self.BASE_URL}/discover/tv",
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])
                all_results.extend(results)
                
                if not results or page >= data.get("total_pages", 1):
                    break
            
            return all_results
        except Exception as e:
            logger.error(f"TMDb TV discovery error: {e}")
            return []
    
    async def get_movie_details(self, movie_id: int) -> Dict[str, Any]:
        """Get detailed movie information"""
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/movie/{movie_id}",
                params={
                    "api_key": self.api_key,
                    "append_to_response": "credits,videos,release_dates"
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"TMDb movie details error: {e}")
            return {}
    
    async def get_tv_details(self, tv_id: int) -> Dict[str, Any]:
        """Get detailed TV show information"""
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/tv/{tv_id}",
                params={
                    "api_key": self.api_key,
                    "append_to_response": "credits,videos,content_ratings"
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"TMDb TV details error: {e}")
            return {}


class AniListClient:
    """Enhanced AniList GraphQL client for anime"""
    
    BASE_URL = "https://graphql.anilist.co"
    
    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self.http_client = http_client or httpx.AsyncClient(timeout=10.0)
    
    def _build_upcoming_query(self) -> str:
        """Build GraphQL query for upcoming anime releases"""
        return """
        query ($startDate: Int, $endDate: Int, $year: Int, $page: Int, $sort: [MediaSort]) {
            Page(page: $page, perPage: 50) {
                pageInfo {
                    total
                    currentPage
                    lastPage
                    hasNextPage
                }
                media(
                    type: ANIME
                    format_in: [TV, TV_SHORT, ONA, OVA, MOVIE, SPECIAL]
                    startDate_greater: $startDate
                    startDate_lesser: $endDate
                    seasonYear: $year
                    sort: $sort
                    isAdult: false
                    status_in: [NOT_YET_RELEASED, RELEASING]
                ) {
                    id
                    idMal
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
                    endDate {
                        year
                        month
                        day
                    }
                    episodes
                    duration
                    status
                    season
                    seasonYear
                    format
                    genres
                    popularity
                    averageScore
                    favourites
                    trending
                    coverImage {
                        large
                        medium
                        extraLarge
                    }
                    bannerImage
                    description(asHtml: false)
                    countryOfOrigin
                    source
                    hashtag
                    trailer {
                        id
                        site
                        thumbnail
                    }
                    studios {
                        nodes {
                            name
                            isAnimationStudio
                        }
                    }
                    staff {
                        edges {
                            role
                            node {
                                name {
                                    full
                                }
                            }
                        }
                    }
                    nextAiringEpisode {
                        airingAt
                        timeUntilAiring
                        episode
                    }
                }
            }
        }
        """
    
    async def get_upcoming_anime(
        self,
        start_date: datetime,
        end_date: datetime,
        sort: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Get upcoming anime releases with enhanced data"""
        if sort is None:
            sort = ["POPULARITY_DESC", "START_DATE_DESC"]
        
        # Convert dates to integers (YYYYMMDD format)
        start_int = int(start_date.strftime("%Y%m%d"))
        end_int = int(end_date.strftime("%Y%m%d"))
        
        variables = {
            "startDate": start_int,
            "endDate": end_int,
            "year": start_date.year,
            "page": 1,
            "sort": sort
        }
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            
            all_results = []
            
            # Fetch multiple pages
            for page in range(1, 3):
                variables["page"] = page
                
                response = await self.http_client.post(
                    self.BASE_URL,
                    json={
                        "query": self._build_upcoming_query(),
                        "variables": variables
                    },
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
                
                if "errors" in data:
                    logger.error(f"AniList GraphQL errors: {data['errors']}")
                    break
                
                results = data.get("data", {}).get("Page", {}).get("media", [])
                all_results.extend(results)
                
                page_info = data.get("data", {}).get("Page", {}).get("pageInfo", {})
                if not page_info.get("hasNextPage", False):
                    break
            
            return all_results
            
        except Exception as e:
            logger.error(f"AniList upcoming query error: {e}")
            return []


class UpcomingContentService:
    """
    Advanced production-grade service for fetching upcoming releases with 
    region/timezone awareness and strict Telugu-first language priority.
    """
    
    def __init__(
        self,
        tmdb_api_key: str,
        cache_backend: Optional[Any] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        enable_analytics: bool = True
    ):
        self.tmdb_client = TMDbClient(tmdb_api_key, http_client)
        self.anilist_client = AniListClient(http_client)
        self.cache = cache_backend
        self.http_client = http_client or httpx.AsyncClient(timeout=10.0)
        self.anticipation_analyzer = AnticipationAnalyzer()
        self.enable_analytics = enable_analytics
        
        # Telugu content detection patterns
        self.telugu_patterns = {
            'languages': ['te', 'telugu'],
            'keywords': ['tollywood', 'telugu cinema', 'andhra', 'telangana'],
            'studios': ['AVM Productions', 'Vyjayanthi Movies', 'Mythri Movie Makers', 
                       'Geetha Arts', 'Sri Venkateswara Creations', 'Konidela Production Company']
        }
    
    def _get_release_windows(self, user_timezone: str) -> List[ReleaseWindow]:
        """
        Get release windows (current month + 2 months) in user's timezone.
        """
        try:
            tz = pytz.timezone(user_timezone)
        except pytz.UnknownTimeZoneError:
            logger.warning(f"Unknown timezone {user_timezone}, using UTC")
            tz = pytz.UTC
        
        now = datetime.now(tz)
        current_year = now.year
        
        windows = []
        
        # Current month
        current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        current_month_end = (current_month_start + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
        
        if current_month_start.year == current_year:
            windows.append(ReleaseWindow(
                start_date=current_month_start,
                end_date=current_month_end,
                month_label="current",
                priority=1,
                is_current=True
            ))
        
        # Next two months
        for i in range(1, 3):
            month_start = (current_month_start + timedelta(days=32 * i)).replace(day=1)
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
            
            if month_start.year == current_year:
                windows.append(ReleaseWindow(
                    start_date=month_start,
                    end_date=month_end,
                    month_label=f"month_{i}",
                    priority=i + 1,
                    is_current=False
                ))
        
        return windows
    
    def _is_telugu_content(self, item: Dict[str, Any]) -> bool:
        """Enhanced Telugu content detection"""
        # Check language
        original_language = item.get("original_language", "").lower()
        if original_language in self.telugu_patterns['languages']:
            return True
        
        # Check title for Telugu words/patterns
        title = (item.get("title", "") + " " + item.get("original_title", "")).lower()
        for keyword in self.telugu_patterns['keywords']:
            if keyword in title:
                return True
        
        # Check production companies (would need additional API call for detailed info)
        # This is a placeholder for future enhancement
        
        return False
    
    def _parse_tmdb_movie(self, item: Dict[str, Any], region: str) -> UpcomingRelease:
        """Parse TMDb movie data into UpcomingRelease object with enhancements"""
        # Parse release date
        release_date_str = item.get("release_date", "")
        try:
            release_date = datetime.strptime(release_date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            release_date = datetime.now() + timedelta(days=30)  # Default to 30 days from now
        
        # Detect language
        original_lang = item.get("original_language", "")
        
        # Enhanced Telugu detection
        is_telugu = self._is_telugu_content(item)
        if is_telugu:
            languages = ["te", original_lang] if original_lang != "te" else ["te"]
        else:
            languages = [original_lang] if original_lang else []
        
        # Get poster and backdrop paths
        poster_path = item.get("poster_path")
        if poster_path and not poster_path.startswith("http"):
            poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        backdrop_path = item.get("backdrop_path")
        if backdrop_path and not backdrop_path.startswith("http"):
            backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        
        # Determine marketing level based on popularity
        marketing_level = "minimal"
        if item.get("popularity", 0) > 100:
            marketing_level = "blockbuster"
        elif item.get("popularity", 0) > 50:
            marketing_level = "heavy"
        elif item.get("popularity", 0) > 20:
            marketing_level = "standard"
        
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
            marketing_level=marketing_level
        )
        
        # Calculate anticipation score
        if self.enable_analytics:
            release.anticipation_score = self.anticipation_analyzer.calculate_anticipation_score(release)
            release.buzz_level = self.anticipation_analyzer.determine_buzz_level(release)
        
        return release
    
    def _parse_tmdb_tv(self, item: Dict[str, Any], region: str) -> UpcomingRelease:
        """Parse TMDb TV data into UpcomingRelease object with enhancements"""
        # Parse first air date
        air_date_str = item.get("first_air_date", "")
        try:
            air_date = datetime.strptime(air_date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            air_date = datetime.now() + timedelta(days=30)
        
        # Detect language
        original_lang = item.get("original_language", "")
        
        # Enhanced Telugu detection
        is_telugu = self._is_telugu_content(item)
        if is_telugu:
            languages = ["te", original_lang] if original_lang != "te" else ["te"]
        else:
            languages = [original_lang] if original_lang else []
        
        # Get poster and backdrop paths
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
            release_strategy="streaming" if item.get("popularity", 0) < 20 else "wide"
        )
        
        # Calculate anticipation score
        if self.enable_analytics:
            release.anticipation_score = self.anticipation_analyzer.calculate_anticipation_score(release)
            release.buzz_level = self.anticipation_analyzer.determine_buzz_level(release)
        
        return release
    
    def _parse_anilist_anime(self, item: Dict[str, Any]) -> Optional[UpcomingRelease]:
        """Parse AniList anime data into UpcomingRelease object with enhancements"""
        # Parse start date
        start_date_info = item.get("startDate", {})
        if not start_date_info:
            return None
        
        year = start_date_info.get("year")
        month = start_date_info.get("month", 1)
        day = start_date_info.get("day", 1)
        
        if not year:
            return None
        
        try:
            release_date = datetime(year, month, day)
        except (ValueError, TypeError):
            return None
        
        # Get title
        title_info = item.get("title", {})
        title = title_info.get("english") or title_info.get("romaji", "")
        
        if not title:
            return None
        
        # Get cover image
        cover_image = item.get("coverImage", {})
        poster_url = cover_image.get("extraLarge") or cover_image.get("large") or cover_image.get("medium")
        
        # Get studio info
        studios = item.get("studios", {}).get("nodes", [])
        studio_name = studios[0].get("name") if studios else None
        
        # Get director info
        director = None
        staff = item.get("staff", {}).get("edges", [])
        for staff_member in staff:
            if staff_member.get("role", "").lower() == "director":
                director = staff_member.get("node", {}).get("name", {}).get("full")
                break
        
        release = UpcomingRelease(
            id=f"anilist_{item['id']}",
            title=title,
            original_title=title_info.get("native"),
            content_type=ContentType.ANIME,
            release_date=release_date,
            languages=["ja"],  # Japanese by default
            genres=item.get("genres", []),
            popularity=item.get("popularity", 0),
            vote_count=item.get("favourites", 0),
            vote_average=(item.get("averageScore", 0) / 10) if item.get("averageScore") else 0,
            poster_path=poster_url,
            backdrop_path=item.get("bannerImage"),
            overview=item.get("description"),
            source="anilist",
            studio=studio_name,
            director=director,
            is_franchise=bool(item.get("hashtag")),
            franchise_name=item.get("hashtag"),
            marketing_level="heavy" if item.get("trending", 0) > 10 else "standard"
        )
        
        # Calculate anticipation score
        if self.enable_analytics:
            release.anticipation_score = self.anticipation_analyzer.calculate_anticipation_score(release)
            release.buzz_level = self.anticipation_analyzer.determine_buzz_level(release)
        
        return release
    
    def _get_genre_names(self, genre_ids: List[int]) -> List[str]:
        """Convert TMDb genre IDs to names"""
        genre_map = {
            28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
            80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
            14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
            9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
            10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
        }
        return [genre_map.get(gid, "Unknown") for gid in genre_ids if gid in genre_map]
    
    async def _fetch_movies_for_window(
        self,
        window: ReleaseWindow,
        region: str,
        languages: List[LanguagePriority]
    ) -> List[UpcomingRelease]:
        """Fetch movies with Telugu priority"""
        releases = []
        seen_ids = set()
        
        # STRICT PRIORITY: Fetch Telugu content FIRST
        telugu_lang = languages[0]  # Telugu is always first
        telugu_items = await self.tmdb_client.discover_movies(
            region=region,
            start_date=window.start_date,
            end_date=window.end_date,
            language=telugu_lang.iso_code,
            sort_by="popularity.desc,primary_release_date.desc"
        )
        
        for item in telugu_items:
            movie_id = f"tmdb_movie_{item['id']}"
            if movie_id not in seen_ids:
                release = self._parse_tmdb_movie(item, region)
                release.is_telugu_priority = True
                releases.append(release)
                seen_ids.add(movie_id)
        
        # Then fetch other priority languages
        for lang in languages[1:]:
            items = await self.tmdb_client.discover_movies(
                region=region,
                start_date=window.start_date,
                end_date=window.end_date,
                language=lang.iso_code
            )
            
            for item in items[:20]:  # Limit per language
                movie_id = f"tmdb_movie_{item['id']}"
                if movie_id not in seen_ids:
                    release = self._parse_tmdb_movie(item, region)
                    releases.append(release)
                    seen_ids.add(movie_id)
        
        # Finally, add general popular releases
        general_items = await self.tmdb_client.discover_movies(
            region=region,
            start_date=window.start_date,
            end_date=window.end_date,
            sort_by="popularity.desc"
        )
        
        for item in general_items[:30]:
            movie_id = f"tmdb_movie_{item['id']}"
            if movie_id not in seen_ids:
                release = self._parse_tmdb_movie(item, region)
                # Check if it might be Telugu content
                if self._is_telugu_content(item):
                    release.is_telugu_priority = True
                releases.append(release)
                seen_ids.add(movie_id)
        
        return releases
    
    async def _fetch_tv_for_window(
        self,
        window: ReleaseWindow,
        region: str,
        languages: List[LanguagePriority]
    ) -> List[UpcomingRelease]:
        """Fetch TV series with Telugu priority"""
        releases = []
        seen_ids = set()
        
        # STRICT PRIORITY: Fetch Telugu content FIRST
        telugu_lang = languages[0]
        telugu_items = await self.tmdb_client.discover_tv(
            region=region,
            start_date=window.start_date,
            end_date=window.end_date,
            language=telugu_lang.iso_code,
            sort_by="popularity.desc,first_air_date.desc"
        )
        
        for item in telugu_items:
            tv_id = f"tmdb_tv_{item['id']}"
            if tv_id not in seen_ids:
                release = self._parse_tmdb_tv(item, region)
                release.is_telugu_priority = True
                releases.append(release)
                seen_ids.add(tv_id)
        
        # Then other languages
        for lang in languages[1:]:
            items = await self.tmdb_client.discover_tv(
                region=region,
                start_date=window.start_date,
                end_date=window.end_date,
                language=lang.iso_code
            )
            
            for item in items[:15]:
                tv_id = f"tmdb_tv_{item['id']}"
                if tv_id not in seen_ids:
                    release = self._parse_tmdb_tv(item, region)
                    releases.append(release)
                    seen_ids.add(tv_id)
        
        return releases
    
    async def _fetch_anime_for_window(
        self,
        window: ReleaseWindow
    ) -> List[UpcomingRelease]:
        """Fetch upcoming anime"""
        releases = []
        
        items = await self.anilist_client.get_upcoming_anime(
            start_date=window.start_date,
            end_date=window.end_date,
            sort=["TRENDING_DESC", "POPULARITY_DESC"]
        )
        
        for item in items:
            release = self._parse_anilist_anime(item)
            if release:
                releases.append(release)
        
        return releases
    
    def _sort_releases_with_telugu_priority(self, releases: List[UpcomingRelease]) -> List[UpcomingRelease]:
        """
        Sort releases with STRICT Telugu priority
        1. Telugu content ALWAYS comes first
        2. Within Telugu, sort by date and popularity
        3. Then other languages by priority
        """
        # Separate Telugu content
        telugu_releases = [r for r in releases if r.is_telugu_priority or r.language_priority == 1]
        other_releases = [r for r in releases if not (r.is_telugu_priority or r.language_priority == 1)]
        
        # Sort Telugu releases by date and anticipation
        telugu_sorted = sorted(
            telugu_releases,
            key=lambda r: (
                -r.anticipation_score,
                r.days_until_release,
                -r.popularity
            )
        )
        
        # Sort other releases
        other_sorted = sorted(
            other_releases,
            key=lambda r: (
                r.language_priority,  # Language priority
                r.days_until_release,  # Days until release
                -r.anticipation_score,  # Anticipation score
                -r.popularity  # Popularity
            )
        )
        
        # Telugu ALWAYS comes first
        return telugu_sorted + other_sorted
    
    def _generate_cache_key(
        self,
        region: str,
        timezone_name: str,
        categories: List[str]
    ) -> str:
        """Generate cache key for the request"""
        components = [
            "upcoming",
            region,
            timezone_name,
            "_".join(sorted(categories)),
            datetime.now().strftime("%Y%m%d%H")  # Hourly cache
        ]
        key_string = ":".join(components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_upcoming_releases(
        self,
        region: str = "IN",
        timezone_name: str = "Asia/Kolkata",
        categories: Optional[List[str]] = None,
        use_cache: bool = True,
        include_analytics: bool = True
    ) -> Dict[str, Any]:
        """
        Get upcoming releases with strict Telugu priority and advanced features.
        
        Args:
            region: ISO 3166-1 alpha-2 country code
            timezone_name: Timezone name (e.g., "Asia/Kolkata")
            categories: List of categories to fetch ["movies", "tv", "anime"]
            use_cache: Whether to use caching
            include_analytics: Include anticipation scores and analytics
        
        Returns:
            Dictionary with categorized upcoming releases
        """
        if categories is None:
            categories = ["movies", "tv", "anime"]
        
        self.enable_analytics = include_analytics
        
        # Check cache
        if use_cache and self.cache:
            cache_key = self._generate_cache_key(region, timezone_name, categories)
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.info(f"Cache hit for upcoming releases: {cache_key}")
                return cached_data
        
        # Get release windows
        windows = self._get_release_windows(timezone_name)
        if not windows:
            return {"error": "No valid release windows"}
        
        # Get languages in STRICT priority order
        priority_languages = LanguagePriority.get_priority_order()
        
        # Initialize results
        results = {
            "movies": [],
            "tv_series": [],
            "anime": [],
            "metadata": {
                "region": region,
                "timezone": timezone_name,
                "windows": [
                    {
                        "label": w.month_label,
                        "start": w.start_date.isoformat(),
                        "end": w.end_date.isoformat(),
                        "is_current": w.is_current
                    }
                    for w in windows
                ],
                "language_priority": [lang.language_name for lang in priority_languages],
                "telugu_priority_enabled": True,
                "fetched_at": datetime.now(pytz.timezone(timezone_name)).isoformat()
            }
        }
        
        # Fetch data for each window
        for window in windows:
            tasks = []
            
            if "movies" in categories:
                tasks.append(("movies", self._fetch_movies_for_window(
                    window, region, priority_languages
                )))
            
            if "tv" in categories:
                tasks.append(("tv_series", self._fetch_tv_for_window(
                    window, region, priority_languages
                )))
            
            if "anime" in categories:
                tasks.append(("anime", self._fetch_anime_for_window(window)))
            
            # Execute tasks
            for category, task in tasks:
                try:
                    releases = await task
                    results[category].extend(releases)
                except Exception as e:
                    logger.error(f"Error fetching {category}: {e}")
        
        # Sort with Telugu priority
        results["movies"] = self._sort_releases_with_telugu_priority(results["movies"])
        results["tv_series"] = self._sort_releases_with_telugu_priority(results["tv_series"])
        results["anime"] = self._sort_releases_with_telugu_priority(results["anime"])
        
        # Add advanced statistics
        telugu_movies = sum(1 for r in results["movies"] if r.is_telugu_priority)
        telugu_tv = sum(1 for r in results["tv_series"] if r.is_telugu_priority)
        
        results["metadata"]["statistics"] = {
            "total_movies": len(results["movies"]),
            "total_tv_series": len(results["tv_series"]),
            "total_anime": len(results["anime"]),
            "telugu_movies": telugu_movies,
            "telugu_tv_series": telugu_tv,
            "telugu_percentage": round(
                (telugu_movies + telugu_tv) / max(1, len(results["movies"]) + len(results["tv_series"])) * 100, 1
            ),
            "high_anticipation_count": sum(
                1 for r in results["movies"] + results["tv_series"] + results["anime"] 
                if r.anticipation_score >= 70
            ),
            "viral_buzz_count": sum(
                1 for r in results["movies"] + results["tv_series"] + results["anime"] 
                if r.buzz_level == "viral"
            ),
            "cache_used": False
        }
        
        # Convert to dict for JSON serialization
        results["movies"] = [r.to_dict() for r in results["movies"]]
        results["tv_series"] = [r.to_dict() for r in results["tv_series"]]
        results["anime"] = [r.to_dict() for r in results["anime"]]
        
        # Cache results
        if use_cache and self.cache:
            cache_key = self._generate_cache_key(region, timezone_name, categories)
            await self._save_to_cache(cache_key, results, ttl=3600)
        
        return results
    
    async def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache"""
        if not self.cache:
            return None
        
        try:
            cached = self.cache.get(f"upcoming:{key}")
            if cached:
                data = json.loads(cached) if isinstance(cached, str) else cached
                if data and "metadata" in data:
                    data["metadata"]["statistics"]["cache_used"] = True
                return data
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        
        return None
    
    async def _save_to_cache(self, key: str, data: Dict[str, Any], ttl: int = 3600):
        """Save data to cache"""
        if not self.cache:
            return
        
        try:
            self.cache.set(
                f"upcoming:{key}",
                json.dumps(data, default=str),
                timeout=ttl
            )
            logger.info(f"Cached upcoming releases: upcoming:{key}")
        except Exception as e:
            logger.error(f"Cache save error: {e}")
    
    async def close(self):
        """Clean up resources"""
        if self.http_client:
            await self.http_client.aclose()