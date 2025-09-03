"""
New Releases Service - Production-grade, region and timezone-aware service
for fetching latest releases across Movies, TV/Series, and Anime.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
from functools import lru_cache
import hashlib
import json

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content type enumeration"""
    MOVIE = "movie"
    TV_SERIES = "tv"
    ANIME = "anime"


class LanguagePriority(Enum):
    """Language priority with ISO-639-1 codes"""
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


@dataclass
class Release:
    """Unified release data structure"""
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
    
    def __post_init__(self):
        """Post-initialization to set language priority"""
        self._set_language_priority()
    
    def _set_language_priority(self):
        """Set language priority based on content languages"""
        for lang in self.languages:
            for lang_enum in LanguagePriority:
                if lang.lower() in [lang_enum.iso_code, lang_enum.language_name]:
                    self.language_priority = min(self.language_priority, lang_enum.priority)
                    break


class TMDbClient:
    """TMDb API client with region awareness"""
    
    BASE_URL = "https://api.themoviedb.org/3"
    
    def __init__(self, api_key: str, http_client: Optional[httpx.AsyncClient] = None):
        self.api_key = api_key
        self.http_client = http_client or httpx.AsyncClient(timeout=10.0)
    
    async def discover_movies(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Discover movies with region-aware release dates"""
        params = {
            "api_key": self.api_key,
            "region": region,
            "primary_release_year": start_date.year,
            "primary_release_date.gte": start_date.strftime("%Y-%m-%d"),
            "primary_release_date.lte": end_date.strftime("%Y-%m-%d"),
            "with_release_type": "2|3|4|5",  # Theatrical, digital, physical, TV
            "sort_by": "primary_release_date.desc,popularity.desc",
            "include_adult": False
        }
        
        if language:
            params["with_original_language"] = language
        
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/discover/movie",
                params=params
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            logger.error(f"TMDb movie discovery error: {e}")
            return []
    
    async def discover_tv(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Discover TV shows with region awareness"""
        params = {
            "api_key": self.api_key,
            "watch_region": region,
            "first_air_date_year": start_date.year,
            "first_air_date.gte": start_date.strftime("%Y-%m-%d"),
            "first_air_date.lte": end_date.strftime("%Y-%m-%d"),
            "sort_by": "first_air_date.desc,popularity.desc",
            "include_adult": False
        }
        
        if language:
            params["with_original_language"] = language
        
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/discover/tv",
                params=params
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            logger.error(f"TMDb TV discovery error: {e}")
            return []
    
    async def get_movie_release_dates(self, movie_id: int) -> Dict[str, Any]:
        """Get region-specific release dates for a movie"""
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/movie/{movie_id}/release_dates",
                params={"api_key": self.api_key}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"TMDb release dates error: {e}")
            return {}


class AniListClient:
    """AniList GraphQL client for anime"""
    
    BASE_URL = "https://graphql.anilist.co"
    
    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self.http_client = http_client or httpx.AsyncClient(timeout=10.0)
    
    def _build_query(self) -> str:
        """Build GraphQL query for anime releases"""
        return """
        query ($startDate: Int, $endDate: Int, $year: Int, $page: Int) {
            Page(page: $page, perPage: 50) {
                pageInfo {
                    total
                    currentPage
                    lastPage
                    hasNextPage
                }
                media(
                    type: ANIME
                    format_in: [TV, TV_SHORT, ONA, OVA, MOVIE]
                    startDate_greater: $startDate
                    startDate_lesser: $endDate
                    seasonYear: $year
                    sort: [START_DATE_DESC, POPULARITY_DESC]
                    isAdult: false
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
                    coverImage {
                        large
                        medium
                        extraLarge
                    }
                    bannerImage
                    description(asHtml: false)
                    countryOfOrigin
                    isAdult
                    nextAiringEpisode {
                        airingAt
                        timeUntilAiring
                        episode
                    }
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
    
    async def get_seasonal_anime(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get anime releases within date range"""
        # Convert dates to integers (YYYYMMDD format)
        start_int = int(start_date.strftime("%Y%m%d"))
        end_int = int(end_date.strftime("%Y%m%d"))
        
        variables = {
            "startDate": start_int,
            "endDate": end_int,
            "year": start_date.year,
            "page": 1
        }
        
        try:
            # Add proper headers for AniList
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            
            response = await self.http_client.post(
                self.BASE_URL,
                json={
                    "query": self._build_query(),
                    "variables": variables
                },
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            # Check for GraphQL errors
            if "errors" in data:
                logger.error(f"AniList GraphQL errors: {data['errors']}")
                return []
            
            return data.get("data", {}).get("Page", {}).get("media", [])
        except httpx.HTTPStatusError as e:
            logger.error(f"AniList HTTP error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            logger.error(f"AniList query error: {e}")
            return []
    
    async def get_current_season_anime(self) -> List[Dict[str, Any]]:
        """Alternative method to get current season anime"""
        query = """
        query ($season: MediaSeason, $year: Int, $page: Int) {
            Page(page: $page, perPage: 50) {
                pageInfo {
                    total
                    hasNextPage
                }
                media(
                    type: ANIME
                    season: $season
                    seasonYear: $year
                    sort: [POPULARITY_DESC, START_DATE_DESC]
                    isAdult: false
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
                    episodes
                    status
                    format
                    genres
                    popularity
                    averageScore
                    favourites
                    coverImage {
                        large
                        medium
                        extraLarge
                    }
                    bannerImage
                    description(asHtml: false)
                    nextAiringEpisode {
                        airingAt
                        episode
                    }
                    studios {
                        nodes {
                            name
                        }
                    }
                }
            }
        }
        """
        
        # Determine current season
        now = datetime.now()
        month = now.month
        if month in [12, 1, 2]:
            season = "WINTER"
        elif month in [3, 4, 5]:
            season = "SPRING"
        elif month in [6, 7, 8]:
            season = "SUMMER"
        else:
            season = "FALL"
        
        variables = {
            "season": season,
            "year": now.year,
            "page": 1
        }
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            
            response = await self.http_client.post(
                self.BASE_URL,
                json={"query": query, "variables": variables},
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            if "errors" in data:
                logger.error(f"AniList GraphQL errors: {data['errors']}")
                return []
            
            return data.get("data", {}).get("Page", {}).get("media", [])
        except Exception as e:
            logger.error(f"AniList current season error: {e}")
            return []


class NewReleasesService:
    """
    Production-grade service for fetching new releases with region and timezone awareness.
    """
    
    def __init__(
        self,
        tmdb_api_key: str,
        cache_backend: Optional[Any] = None,
        http_client: Optional[httpx.AsyncClient] = None
    ):
        self.tmdb_client = TMDbClient(tmdb_api_key, http_client)
        self.anilist_client = AniListClient(http_client)
        self.cache = cache_backend
        self.http_client = http_client or httpx.AsyncClient(timeout=10.0)
    
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
        
        # Only include if in current year
        if current_month_start.year == current_year:
            windows.append(ReleaseWindow(
                start_date=current_month_start,
                end_date=current_month_end,
                month_label="current",
                priority=1
            ))
        
        # Next two months
        for i in range(1, 3):
            month_start = (current_month_start + timedelta(days=32 * i)).replace(day=1)
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
            
            # Only include if in current year
            if month_start.year == current_year:
                windows.append(ReleaseWindow(
                    start_date=month_start,
                    end_date=month_end,
                    month_label=f"month_{i}",
                    priority=i + 1
                ))
        
        return windows
    
    def _parse_tmdb_movie(self, item: Dict[str, Any], region: str) -> Release:
        """Parse TMDb movie data into Release object"""
        # Parse release date
        release_date_str = item.get("release_date", "")
        try:
            release_date = datetime.strptime(release_date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            release_date = datetime.now()
        
        # Detect language from original_language field
        original_lang = item.get("original_language", "")
        languages = [original_lang] if original_lang else []
        
        # Get poster and backdrop paths
        poster_path = item.get("poster_path")
        if poster_path and not poster_path.startswith("http"):
            poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        backdrop_path = item.get("backdrop_path")
        if backdrop_path and not backdrop_path.startswith("http"):
            backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        
        return Release(
            id=f"tmdb_movie_{item['id']}",
            title=item.get("title", ""),
            original_title=item.get("original_title"),
            content_type=ContentType.MOVIE,
            release_date=release_date,
            languages=languages,
            genres=[],  # Would need genre mapping
            popularity=item.get("popularity", 0),
            vote_count=item.get("vote_count", 0),
            vote_average=item.get("vote_average", 0),
            poster_path=poster_path,
            backdrop_path=backdrop_path,
            overview=item.get("overview"),
            region=region,
            source="tmdb"
        )
    
    def _parse_tmdb_tv(self, item: Dict[str, Any], region: str) -> Release:
        """Parse TMDb TV data into Release object"""
        # Parse first air date
        air_date_str = item.get("first_air_date", "")
        try:
            air_date = datetime.strptime(air_date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            air_date = datetime.now()
        
        # Detect language
        original_lang = item.get("original_language", "")
        languages = [original_lang] if original_lang else []
        
        # Get poster and backdrop paths
        poster_path = item.get("poster_path")
        if poster_path and not poster_path.startswith("http"):
            poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        backdrop_path = item.get("backdrop_path")
        if backdrop_path and not backdrop_path.startswith("http"):
            backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        
        return Release(
            id=f"tmdb_tv_{item['id']}",
            title=item.get("name", ""),
            original_title=item.get("original_name"),
            content_type=ContentType.TV_SERIES,
            release_date=air_date,
            languages=languages,
            genres=[],
            popularity=item.get("popularity", 0),
            vote_count=item.get("vote_count", 0),
            vote_average=item.get("vote_average", 0),
            poster_path=poster_path,
            backdrop_path=backdrop_path,
            overview=item.get("overview"),
            region=region,
            source="tmdb"
        )
    
    def _parse_anilist_anime(self, item: Dict[str, Any]) -> Optional[Release]:
        """Parse AniList anime data into Release object"""
        # Parse start date
        start_date_info = item.get("startDate", {})
        if not start_date_info:
            return None
        
        # Check if we have valid date components
        year = start_date_info.get("year")
        month = start_date_info.get("month")
        day = start_date_info.get("day")
        
        # Skip if no year
        if not year:
            return None
        
        # Use defaults for missing month/day
        month = month or 1
        day = day or 1
        
        try:
            release_date = datetime(year, month, day)
        except (ValueError, TypeError):
            return None
        
        # Get title (prefer English, fallback to Romaji)
        title_info = item.get("title", {})
        title = title_info.get("english") or title_info.get("romaji", "")
        
        if not title:
            return None
        
        # Get cover image
        cover_image = item.get("coverImage", {})
        poster_url = cover_image.get("extraLarge") or cover_image.get("large") or cover_image.get("medium")
        
        return Release(
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
            source="anilist"
        )
    
    async def _fetch_movies_for_window(
        self,
        window: ReleaseWindow,
        region: str,
        languages: List[LanguagePriority]
    ) -> List[Release]:
        """Fetch movies for a specific release window"""
        releases = []
        
        # Fetch for each priority language
        for lang in languages:
            items = await self.tmdb_client.discover_movies(
                region=region,
                start_date=window.start_date,
                end_date=window.end_date,
                language=lang.iso_code
            )
            
            for item in items:
                release = self._parse_tmdb_movie(item, region)
                releases.append(release)
        
        # Also fetch general releases without language filter for the region
        general_items = await self.tmdb_client.discover_movies(
            region=region,
            start_date=window.start_date,
            end_date=window.end_date
        )
        
        for item in general_items[:20]:  # Limit to avoid too many items
            release = self._parse_tmdb_movie(item, region)
            # Avoid duplicates
            if not any(r.id == release.id for r in releases):
                releases.append(release)
        
        return releases
    
    async def _fetch_tv_for_window(
        self,
        window: ReleaseWindow,
        region: str,
        languages: List[LanguagePriority]
    ) -> List[Release]:
        """Fetch TV series for a specific release window"""
        releases = []
        
        # Fetch for each priority language
        for lang in languages:
            items = await self.tmdb_client.discover_tv(
                region=region,
                start_date=window.start_date,
                end_date=window.end_date,
                language=lang.iso_code
            )
            
            for item in items:
                release = self._parse_tmdb_tv(item, region)
                releases.append(release)
        
        # Also fetch general releases
        general_items = await self.tmdb_client.discover_tv(
            region=region,
            start_date=window.start_date,
            end_date=window.end_date
        )
        
        for item in general_items[:20]:
            release = self._parse_tmdb_tv(item, region)
            if not any(r.id == release.id for r in releases):
                releases.append(release)
        
        return releases
    
    async def _fetch_anime_for_window(
        self,
        window: ReleaseWindow
    ) -> List[Release]:
        """Fetch anime for a specific release window"""
        releases = []
        
        # Try the date range query first
        items = await self.anilist_client.get_seasonal_anime(
            start_date=window.start_date,
            end_date=window.end_date
        )
        
        # If no results or error, try current season as fallback for current window
        if not items and window.priority == 1:  # Only for current month window
            logger.info("Falling back to current season anime query")
            items = await self.anilist_client.get_current_season_anime()
        
        for item in items:
            release = self._parse_anilist_anime(item)
            if release:
                # Only include if release date is within our window or if it's currently airing
                if window.priority == 1:  # Current month - include currently airing
                    releases.append(release)
                elif release.release_date >= window.start_date and release.release_date <= window.end_date:
                    releases.append(release)
        
        return releases
    
    def _sort_releases(self, releases: List[Release]) -> List[Release]:
        """
        Sort releases according to specifications:
        1. Date (newest first)
        2. Popularity/vote_count
        3. Language priority as tiebreaker
        """
        return sorted(
            releases,
            key=lambda r: (
                -r.release_date.timestamp(),  # Newest first
                -(r.popularity * 0.7 + r.vote_count * 0.3),  # Weighted popularity
                r.language_priority  # Language priority (lower is better)
            )
        )
    
    def _deduplicate_releases(self, releases: List[Release]) -> List[Release]:
        """Remove duplicate releases based on ID"""
        seen_ids = set()
        unique_releases = []
        
        for release in releases:
            if release.id not in seen_ids:
                seen_ids.add(release.id)
                unique_releases.append(release)
        
        return unique_releases
    
    def _generate_cache_key(
        self,
        region: str,
        timezone_name: str,
        categories: List[str]
    ) -> str:
        """Generate cache key for the request"""
        components = [
            "new_releases",
            region,
            timezone_name,
            "_".join(sorted(categories)),
            datetime.now().strftime("%Y%m%d%H")  # Hourly cache
        ]
        key_string = ":".join(components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_new_releases(
        self,
        region: str = "US",
        timezone_name: str = "UTC",
        categories: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get new releases with region and timezone awareness.
        
        Args:
            region: ISO 3166-1 alpha-2 country code
            timezone_name: Timezone name (e.g., "America/New_York", "Asia/Kolkata")
            categories: List of categories to fetch ["movies", "tv", "anime"]
            use_cache: Whether to use caching
        
        Returns:
            Dictionary with categorized new releases
        """
        if categories is None:
            categories = ["movies", "tv", "anime"]
        
        # Check cache
        if use_cache and self.cache:
            cache_key = self._generate_cache_key(region, timezone_name, categories)
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                logger.info(f"Cache hit for new releases: {cache_key}")
                return cached_data
        
        # Get release windows in user's timezone
        windows = self._get_release_windows(timezone_name)
        if not windows:
            logger.warning(f"No valid release windows for timezone {timezone_name}")
            return {"error": "No valid release windows"}
        
        # Get languages in priority order
        priority_languages = LanguagePriority.get_priority_order()
        
        # Fetch data for each category and window
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
                        "end": w.end_date.isoformat()
                    }
                    for w in windows
                ],
                "language_priority": [lang.language_name for lang in priority_languages],
                "fetched_at": datetime.now(pytz.timezone(timezone_name)).isoformat()
            }
        }
        
        # Fetch data concurrently for each window
        for window in windows:
            window_releases = {
                "movies": [],
                "tv_series": [],
                "anime": []
            }
            
            # Create async tasks for parallel fetching
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
            
            # Execute tasks concurrently
            for category, task in tasks:
                try:
                    releases = await task
                    window_releases[category].extend(releases)
                except Exception as e:
                    logger.error(f"Error fetching {category} for window {window.month_label}: {e}")
            
            # Add window results to main results
            for category in ["movies", "tv_series", "anime"]:
                results[category].extend(window_releases[category])
        
        # Deduplicate and sort each category
        for category in ["movies", "tv_series", "anime"]:
            results[category] = self._deduplicate_releases(results[category])
            results[category] = self._sort_releases(results[category])
        
        # Add statistics to metadata
        results["metadata"]["statistics"] = {
            "total_movies": len(results["movies"]),
            "total_tv_series": len(results["tv_series"]),
            "total_anime": len(results["anime"]),
            "telugu_content": sum(1 for r in results["movies"] + results["tv_series"] if r.language_priority == 1),
            "english_content": sum(1 for r in results["movies"] + results["tv_series"] if r.language_priority == 2),
            "cache_used": False
        }
        
        # Cache the results
        if use_cache and self.cache:
            cache_key = self._generate_cache_key(region, timezone_name, categories)
            await self._save_to_cache(cache_key, results, ttl=3600)  # 1 hour TTL
        
        return results
    
    async def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache"""
        if not self.cache:
            return None
        
        try:
            cached = self.cache.get(f"new_releases:{key}")
            if cached:
                data = json.loads(cached) if isinstance(cached, str) else cached
                # Mark that cache was used
                if data and "metadata" in data:
                    data["metadata"]["statistics"] = data["metadata"].get("statistics", {})
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
                f"new_releases:{key}",
                json.dumps(data, default=str),
                timeout=ttl
            )
            logger.info(f"Cached new releases with key: new_releases:{key}")
        except Exception as e:
            logger.error(f"Cache save error: {e}")
    
    async def close(self):
        """Clean up resources"""
        if self.http_client:
            await self.http_client.aclose()