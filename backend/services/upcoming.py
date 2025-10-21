#backend/service/upcoming.py
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Set, Union
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
import re
import unicodedata

logger = logging.getLogger(__name__)


class ContentType(Enum):
    MOVIE = "movie"
    TV_SERIES = "tv"
    ANIME = "anime"
    ALL = "all"


class LanguagePriority(Enum):
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
        iso_lower = iso_code.lower()
        for lang in cls:
            if lang.iso_code == iso_lower:
                return lang
        return cls.OTHER
    
    @classmethod
    def get_by_name(cls, name: str) -> Optional['LanguagePriority']:
        name_lower = name.lower()
        for lang in cls:
            if lang.language_name == name_lower:
                return lang
        return cls.OTHER
    
    @classmethod
    def get_priority_order(cls) -> List['LanguagePriority']:
        return sorted([lang for lang in cls if lang != cls.OTHER], key=lambda x: x.priority)


@dataclass
class ReleaseWindow:
    start_date: datetime
    end_date: datetime
    month_label: str
    month_number: int
    priority: int
    is_current: bool = False
    total_days: int = 0
    
    def __post_init__(self):
        self.total_days = (self.end_date - self.start_date).days


@dataclass
class CinebrainUpcomingRelease:
    id: str
    slug: str
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
    
    tmdb_id: Optional[int] = None
    imdb_id: Optional[str] = None
    mal_id: Optional[int] = None
    runtime: Optional[int] = None
    certification: Optional[str] = None
    production_companies: List[str] = field(default_factory=list)
    production_countries: List[str] = field(default_factory=list)
    spoken_languages: List[str] = field(default_factory=list)
    
    days_until_release: int = 0
    weeks_until_release: int = 0
    months_until_release: int = 0
    release_status: str = "announced"
    release_type: str = "theatrical"
    
    is_telugu_content: bool = False
    is_telugu_priority: bool = False
    telugu_confidence_score: float = 0.0
    
    cinebrain_anticipation_score: float = 0.0
    cinebrain_buzz_level: str = "normal"
    cinebrain_trending_score: float = 0.0
    
    director: Optional[str] = None
    cast: List[str] = field(default_factory=list)
    trailer_url: Optional[str] = None
    official_site: Optional[str] = None
    
    def __post_init__(self):
        self._set_language_priority()
        self._calculate_time_until_release()
        self._determine_telugu_status()
        self._generate_cinebrain_slug()
        self._calculate_cinebrain_scores()
    
    def _set_language_priority(self):
        min_priority = 999
        for lang in self.languages:
            lang_enum = LanguagePriority.get_by_iso(lang) or LanguagePriority.get_by_name(lang)
            if lang_enum:
                min_priority = min(min_priority, lang_enum.priority)
        self.language_priority = min_priority
    
    def _calculate_time_until_release(self):
        if self.release_date:
            now = datetime.now()
            delta = self.release_date - now
            self.days_until_release = max(0, delta.days)
            self.weeks_until_release = max(0, delta.days // 7)
            self.months_until_release = max(0, delta.days // 30)
    
    def _determine_telugu_status(self):
        telugu_indicators = ['te', 'telugu', 'tollywood']
        
        for lang in self.languages:
            if lang.lower() in telugu_indicators:
                self.is_telugu_content = True
                self.is_telugu_priority = True
                self.telugu_confidence_score = 1.0
                return
        
        if self.title or self.original_title:
            combined_title = f"{self.title or ''} {self.original_title or ''}".lower()
            telugu_keywords = ['tollywood', 'telugu', 'andhra', 'telangana', 'hyderabad']
            
            matches = sum(1 for keyword in telugu_keywords if keyword in combined_title)
            if matches > 0:
                self.telugu_confidence_score = min(1.0, matches * 0.3)
                if self.telugu_confidence_score >= 0.5:
                    self.is_telugu_content = True
                    self.is_telugu_priority = True
        
        cinebrain_telugu_studios = [
            'geetha arts', 'mythri movie makers', 'avm productions',
            'vyjayanthi movies', 'sri venkateswara creations', 
            'konidela production company', 'uv creations',
            'haarika hassine creations', 'dil raju productions'
        ]
        
        for company in self.production_companies:
            if any(studio in company.lower() for studio in cinebrain_telugu_studios):
                self.is_telugu_content = True
                self.is_telugu_priority = True
                self.telugu_confidence_score = max(self.telugu_confidence_score, 0.8)
                break
    
    def _generate_cinebrain_slug(self):
        if not self.slug and self.title:
            clean_title = re.sub(r'[^\w\s-]', '', self.title.lower())
            clean_title = re.sub(r'[-\s]+', '-', clean_title)
            
            year = ""
            if self.release_date:
                year = f"-{self.release_date.year}"
            
            type_suffix = f"-{self.content_type.value}"
            
            self.slug = f"cinebrain-{clean_title}{year}{type_suffix}".strip('-')
    
    def _calculate_cinebrain_scores(self):
        score = 0.0
        
        if self.popularity > 0:
            score += min(30, self.popularity / 10)
        
        if self.vote_average > 0:
            score += (self.vote_average / 10) * 20
        
        if self.language_priority == 1:
            score += 25
        elif self.language_priority == 2:
            score += 20
        elif self.language_priority == 3:
            score += 15
        elif self.language_priority <= 6:
            score += 10
        
        if 0 <= self.days_until_release <= 7:
            score += 15
        elif 8 <= self.days_until_release <= 30:
            score += 10
        elif 31 <= self.days_until_release <= 60:
            score += 5
        
        self.cinebrain_anticipation_score = min(100, score)
        
        if self.cinebrain_anticipation_score >= 80:
            self.cinebrain_buzz_level = "viral"
        elif self.cinebrain_anticipation_score >= 60:
            self.cinebrain_buzz_level = "high"
        elif self.cinebrain_anticipation_score >= 30:
            self.cinebrain_buzz_level = "normal"
        else:
            self.cinebrain_buzz_level = "low"
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        
        if self.release_date:
            data['release_date'] = self.release_date.isoformat()
        
        data['content_type'] = self.content_type.value
        data['cinebrain_service'] = 'upcoming'
        
        return data


class CinebrainTeluguContentDetector:
    CINEBRAIN_TELUGU_PATTERNS = {
        'languages': ['te', 'telugu'],
        'keywords': [
            'tollywood', 'telugu cinema', 'andhra pradesh', 'telangana',
            'hyderabad', 'vijayawada', 'vishakhapatnam', 'warangal'
        ],
        'studios': [
            'geetha arts', 'mythri movie makers', 'avm productions',
            'vyjayanthi movies', 'sri venkateswara creations',
            'konidela production company', 'uv creations',
            'haarika hassine creations', 'dil raju productions',
            'sitara entertainments', '14 reels plus', 'suresh productions'
        ]
    }
    
    @classmethod
    def detect_cinebrain_telugu_content(cls, item: Dict[str, Any]) -> Tuple[bool, float]:
        confidence = 0.0
        
        original_lang = item.get('original_language', '').lower()
        if original_lang in cls.CINEBRAIN_TELUGU_PATTERNS['languages']:
            return True, 1.0
        
        title_text = f"{item.get('title', '')} {item.get('original_title', '')} {item.get('name', '')} {item.get('original_name', '')}".lower()
        
        for keyword in cls.CINEBRAIN_TELUGU_PATTERNS['keywords']:
            if keyword in title_text:
                confidence += 0.3
        
        is_telugu = confidence >= 0.3
        return is_telugu, min(1.0, confidence)


class CinebrainTMDbClient:
    BASE_URL = "https://api.themoviedb.org/3"
    
    def __init__(self, api_key: str, http_client: Optional[httpx.AsyncClient] = None):
        self.api_key = api_key
        self.http_client = http_client or httpx.AsyncClient(timeout=5.0, limits=httpx.Limits(max_connections=10))
    
    async def discover_cinebrain_upcoming_movies(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime,
        language: Optional[str] = None,
        with_genres: Optional[List[int]] = None,
        sort_by: str = "primary_release_date.asc,popularity.desc",
        page: int = 1
    ) -> Dict[str, Any]:
        params = {
            "api_key": self.api_key,
            "region": region,
            "primary_release_date.gte": start_date.strftime("%Y-%m-%d"),
            "primary_release_date.lte": end_date.strftime("%Y-%m-%d"),
            "with_release_type": "2|3",
            "sort_by": sort_by,
            "include_adult": False,
            "page": page
        }
        
        if language and language != 'all':
            params["with_original_language"] = language
        
        if with_genres:
            params["with_genres"] = ",".join(map(str, with_genres))
        
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/discover/movie",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"CineBrain TMDb movie discovery error: {e}")
            return {"results": [], "total_pages": 0, "total_results": 0}
    
    async def discover_cinebrain_upcoming_tv(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime,
        language: Optional[str] = None,
        with_genres: Optional[List[int]] = None,
        sort_by: str = "first_air_date.asc,popularity.desc",
        page: int = 1
    ) -> Dict[str, Any]:
        params = {
            "api_key": self.api_key,
            "watch_region": region,
            "first_air_date.gte": start_date.strftime("%Y-%m-%d"),
            "first_air_date.lte": end_date.strftime("%Y-%m-%d"),
            "sort_by": sort_by,
            "include_adult": False,
            "page": page
        }
        
        if language and language != 'all':
            params["with_original_language"] = language
        
        if with_genres:
            params["with_genres"] = ",".join(map(str, with_genres))
        
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/discover/tv",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"CineBrain TMDb TV discovery error: {e}")
            return {"results": [], "total_pages": 0, "total_results": 0}


class CinebrainAniListClient:
    BASE_URL = "https://graphql.anilist.co"
    
    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self.http_client = http_client or httpx.AsyncClient(timeout=5.0)
    
    def _build_cinebrain_upcoming_query(self) -> str:
        return """
        query ($season: MediaSeason, $seasonYear: Int, $page: Int, $sort: [MediaSort]) {
            Page(page: $page, perPage: 25) {
                pageInfo {
                    total
                    currentPage
                    lastPage
                    hasNextPage
                }
                media(
                    type: ANIME
                    format_in: [TV, TV_SHORT, ONA, OVA, MOVIE, SPECIAL]
                    season: $season
                    seasonYear: $seasonYear
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
                        userPreferred
                    }
                    startDate {
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
                    siteUrl
                    trailer {
                        id
                        site
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
    
    async def get_cinebrain_upcoming_anime(
        self,
        start_date: datetime,
        end_date: datetime,
        page: int = 1,
        sort: List[str] = None
    ) -> List[Dict[str, Any]]:
        if sort is None:
            sort = ["POPULARITY_DESC", "START_DATE_DESC"]
        
        month = start_date.month
        year = start_date.year
        
        if month in [1, 2, 3]:
            season = "WINTER"
        elif month in [4, 5, 6]:
            season = "SPRING"
        elif month in [7, 8, 9]:
            season = "SUMMER"
        else:
            season = "FALL"
        
        all_results = []
        
        seasons_to_check = [
            (season, year),
            (self._get_next_season(season), year if season != "FALL" else year + 1)
        ]
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            
            for check_season, check_year in seasons_to_check[:1]:  # Only check current season for performance
                variables = {
                    "season": check_season,
                    "seasonYear": check_year,
                    "page": page,
                    "sort": sort
                }
                
                response = await self.http_client.post(
                    self.BASE_URL,
                    json={
                        "query": self._build_cinebrain_upcoming_query(),
                        "variables": variables
                    },
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "errors" in data:
                        logger.error(f"CineBrain AniList GraphQL errors: {data['errors']}")
                        continue
                    
                    results = data.get("data", {}).get("Page", {}).get("media", [])
                    
                    for anime in results:
                        start_date_info = anime.get("startDate", {})
                        if start_date_info and start_date_info.get("year"):
                            try:
                                anime_date = datetime(
                                    start_date_info.get("year"),
                                    start_date_info.get("month", 1),
                                    start_date_info.get("day", 1)
                                )
                                if start_date <= anime_date <= end_date:
                                    all_results.append(anime)
                            except:
                                pass
            
            return all_results
            
        except Exception as e:
            logger.error(f"CineBrain AniList upcoming query error: {e}")
            return []
    
    def _get_next_season(self, current_season: str) -> str:
        seasons = ["WINTER", "SPRING", "SUMMER", "FALL"]
        current_idx = seasons.index(current_season)
        return seasons[(current_idx + 1) % 4]


class CinebrainUpcomingContentService:
    CINEBRAIN_TMDB_GENRES = {
        28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
        80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
        14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
        9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
        10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
        10759: "Action & Adventure", 10762: "Kids", 10763: "News",
        10764: "Reality", 10765: "Sci-Fi & Fantasy", 10766: "Soap",
        10767: "Talk", 10768: "War & Politics"
    }
    
    def __init__(
        self,
        tmdb_api_key: str,
        cache_backend: Optional[Any] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        enable_detailed_info: bool = False
    ):
        self.tmdb_client = CinebrainTMDbClient(tmdb_api_key, http_client)
        self.anilist_client = CinebrainAniListClient(http_client)
        self.cache = cache_backend
        self.http_client = http_client or httpx.AsyncClient(timeout=5.0)
        self.detector = CinebrainTeluguContentDetector()
        self.enable_detailed_info = enable_detailed_info
        self._all_releases_cache = {}
    
    def _get_cinebrain_flexible_release_windows(
        self, 
        user_timezone: str, 
        months: int = 3
    ) -> List[ReleaseWindow]:
        try:
            tz = pytz.timezone(user_timezone)
        except pytz.UnknownTimeZoneError:
            logger.warning(f"CineBrain unknown timezone {user_timezone}, using UTC")
            tz = pytz.UTC
        
        now = datetime.now(tz)
        windows = []
        
        for month_offset in range(months):
            if month_offset == 0:
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
                label = "current"
                is_current = True
            else:
                month_start = (now.replace(day=1) + timedelta(days=32 * month_offset)).replace(day=1)
                start_date = month_start.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
                label = f"month_{month_offset + 1}"
                is_current = False
            
            windows.append(ReleaseWindow(
                start_date=start_date,
                end_date=end_date,
                month_label=label,
                month_number=month_offset + 1,
                priority=month_offset + 1,
                is_current=is_current
            ))
        
        return windows
    
    def _get_cinebrain_genre_names(self, genre_ids: List[int]) -> List[str]:
        return [self.CINEBRAIN_TMDB_GENRES.get(gid, "Unknown") for gid in genre_ids if gid in self.CINEBRAIN_TMDB_GENRES]
    
    def _filter_cinebrain_by_genre(self, genre_filter: str) -> Optional[List[int]]:
        if not genre_filter or genre_filter.lower() == 'all':
            return None
        
        genre_map = {name.lower(): gid for gid, name in self.CINEBRAIN_TMDB_GENRES.items()}
        
        if genre_filter.lower() in genre_map:
            return [genre_map[genre_filter.lower()]]
        
        return None
    
    async def _parse_cinebrain_tmdb_movie(self, item: Dict[str, Any], region: str) -> CinebrainUpcomingRelease:
        release_date_str = item.get("release_date", "")
        try:
            release_date = datetime.strptime(release_date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            release_date = datetime.now() + timedelta(days=30)
        
        is_telugu, confidence = self.detector.detect_cinebrain_telugu_content(item)
        
        original_lang = item.get("original_language", "")
        languages = [original_lang] if original_lang else []
        
        if is_telugu and 'te' not in languages:
            languages.insert(0, 'te')
        
        poster_path = item.get("poster_path")
        if poster_path and not poster_path.startswith("http"):
            poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        backdrop_path = item.get("backdrop_path")
        if backdrop_path and not backdrop_path.startswith("http"):
            backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        
        release = CinebrainUpcomingRelease(
            id=f"cinebrain_movie_{item['id']}",
            slug="",
            title=item.get("title", ""),
            original_title=item.get("original_title"),
            content_type=ContentType.MOVIE,
            release_date=release_date,
            languages=languages,
            genres=self._get_cinebrain_genre_names(item.get("genre_ids", [])),
            popularity=item.get("popularity", 0),
            vote_count=item.get("vote_count", 0),
            vote_average=item.get("vote_average", 0),
            poster_path=poster_path,
            backdrop_path=backdrop_path,
            overview=item.get("overview"),
            region=region,
            source="cinebrain_tmdb",
            tmdb_id=item.get('id'),
            spoken_languages=languages,
            release_type="theatrical",
            is_telugu_content=is_telugu,
            telugu_confidence_score=confidence
        )
        
        return release
    
    async def _parse_cinebrain_tmdb_tv(self, item: Dict[str, Any], region: str) -> CinebrainUpcomingRelease:
        air_date_str = item.get("first_air_date", "")
        try:
            air_date = datetime.strptime(air_date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            air_date = datetime.now() + timedelta(days=30)
        
        is_telugu, confidence = self.detector.detect_cinebrain_telugu_content(item)
        
        original_lang = item.get("original_language", "")
        languages = [original_lang] if original_lang else []
        
        if is_telugu and 'te' not in languages:
            languages.insert(0, 'te')
        
        poster_path = item.get("poster_path")
        if poster_path and not poster_path.startswith("http"):
            poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        backdrop_path = item.get("backdrop_path")
        if backdrop_path and not backdrop_path.startswith("http"):
            backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        
        release = CinebrainUpcomingRelease(
            id=f"cinebrain_tv_{item['id']}",
            slug="",
            title=item.get("name", ""),
            original_title=item.get("original_name"),
            content_type=ContentType.TV_SERIES,
            release_date=air_date,
            languages=languages,
            genres=self._get_cinebrain_genre_names(item.get("genre_ids", [])),
            popularity=item.get("popularity", 0),
            vote_count=item.get("vote_count", 0),
            vote_average=item.get("vote_average", 0),
            poster_path=poster_path,
            backdrop_path=backdrop_path,
            overview=item.get("overview"),
            region=region,
            source="cinebrain_tmdb",
            tmdb_id=item.get('id'),
            spoken_languages=languages,
            release_type="tv",
            is_telugu_content=is_telugu,
            telugu_confidence_score=confidence
        )
        
        return release
    
    async def _parse_cinebrain_anilist_anime(self, item: Dict[str, Any]) -> Optional[CinebrainUpcomingRelease]:
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
        
        title_info = item.get("title", {})
        title = (title_info.get("english") or 
                title_info.get("userPreferred") or 
                title_info.get("romaji", ""))
        
        if not title:
            return None
        
        cover_image = item.get("coverImage", {})
        poster_url = (cover_image.get("extraLarge") or 
                     cover_image.get("large") or 
                     cover_image.get("medium"))
        
        release = CinebrainUpcomingRelease(
            id=f"cinebrain_anime_{item['id']}",
            slug="",
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
            source="cinebrain_anilist",
            mal_id=item.get("idMal"),
            runtime=item.get("duration"),
            official_site=item.get("siteUrl"),
            release_type="streaming",
            cinebrain_trending_score=item.get("trending", 0)
        )
        
        return release
    
    async def _fetch_initial_data(
        self,
        windows: List[ReleaseWindow],
        region: str,
        genre_ids: Optional[List[int]] = None
    ) -> Dict[str, List[CinebrainUpcomingRelease]]:
        """Fetch initial data for all content types with all languages"""
        all_releases = {
            "movies": [],
            "tv_series": [],
            "anime": []
        }
        
        # Only fetch first window for initial load
        first_window = windows[0]
        
        # Fetch all languages in parallel for initial display
        async def fetch_movies():
            tasks = []
            # Fetch movies without language filter to get all languages
            data = await self.tmdb_client.discover_cinebrain_upcoming_movies(
                region=region,
                start_date=first_window.start_date,
                end_date=first_window.end_date,
                language=None,  # No language filter to get all
                with_genres=genre_ids,
                page=1
            )
            
            releases = []
            for item in data.get("results", [])[:20]:
                release = await self._parse_cinebrain_tmdb_movie(item, region)
                releases.append(release)
            
            return releases
        
        async def fetch_tv():
            data = await self.tmdb_client.discover_cinebrain_upcoming_tv(
                region=region,
                start_date=first_window.start_date,
                end_date=first_window.end_date,
                language=None,  # No language filter to get all
                with_genres=genre_ids,
                page=1
            )
            
            releases = []
            for item in data.get("results", [])[:20]:
                release = await self._parse_cinebrain_tmdb_tv(item, region)
                releases.append(release)
            
            return releases
        
        async def fetch_anime():
            items = await self.anilist_client.get_cinebrain_upcoming_anime(
                start_date=first_window.start_date,
                end_date=first_window.end_date,
                page=1
            )
            
            releases = []
            for item in items[:20]:
                release = await self._parse_cinebrain_anilist_anime(item)
                if release:
                    releases.append(release)
            
            return releases
        
        # Execute all fetches in parallel
        movies, tv_series, anime = await asyncio.gather(
            fetch_movies(),
            fetch_tv(),
            fetch_anime()
        )
        
        all_releases["movies"] = movies
        all_releases["tv_series"] = tv_series
        all_releases["anime"] = anime
        
        return all_releases
    
    def _sort_cinebrain_releases_telugu_first(self, releases: List[CinebrainUpcomingRelease]) -> List[CinebrainUpcomingRelease]:
        return sorted(releases, key=lambda r: (
            not r.is_telugu_priority,
            r.language_priority,
            r.release_date,
            -r.popularity,
            -r.vote_average
        ))
    
    def _filter_cinebrain_releases(
        self,
        releases: List[CinebrainUpcomingRelease],
        language_filter: Optional[str] = None,
        genre_filter: Optional[str] = None,
        months_filter: Optional[int] = None
    ) -> List[CinebrainUpcomingRelease]:
        filtered = releases
        
        if language_filter and language_filter.lower() != 'all':
            if language_filter.lower() == 'telugu':
                filtered = [r for r in filtered if r.is_telugu_content]
            else:
                lang_enum = LanguagePriority.get_by_name(language_filter)
                if lang_enum:
                    filtered = [r for r in filtered if lang_enum.iso_code in r.languages]
        
        if genre_filter and genre_filter.lower() not in ['all', 'any']:
            filtered = [r for r in filtered if genre_filter.lower() in [g.lower() for g in r.genres]]
        
        if months_filter:
            cutoff_date = datetime.now() + timedelta(days=months_filter * 30)
            filtered = [r for r in filtered if r.release_date <= cutoff_date]
        
        return filtered
    
    def _generate_cinebrain_cache_key(
        self,
        region: str,
        timezone_name: str,
        content_type: str,
        language: str,
        genre: str,
        months: int,
        page: int,
        limit: int
    ) -> str:
        components = [
            "cinebrain_upcoming_lazy_v1",
            region,
            timezone_name,
            content_type,
            language,
            genre,
            str(months),
            str(page),
            str(limit),
            datetime.now().strftime("%Y%m%d%H")
        ]
        key_string = ":".join(components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_cinebrain_upcoming_releases_lazy(
        self,
        region: str = "IN",
        timezone_name: str = "Asia/Kolkata",
        content_type: str = "all",
        language: str = "all",
        genre: str = "all",
        months: int = 3,
        page: int = 1,
        limit: int = 20,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        months = max(1, min(3, months))
        
        if use_cache and self.cache:
            cache_key = self._generate_cinebrain_cache_key(
                region, timezone_name, content_type, language, genre, months, page, limit
            )
            cached_data = await self._get_cinebrain_from_cache(cache_key)
            if cached_data:
                logger.info(f"CineBrain cache hit for upcoming releases: {cache_key}")
                return cached_data
        
        windows = self._get_cinebrain_flexible_release_windows(timezone_name, months)
        genre_ids = self._filter_cinebrain_by_genre(genre)
        priority_languages = LanguagePriority.get_priority_order()
        
        # For first page, fetch initial data
        if page == 1:
            all_releases = await self._fetch_initial_data(windows, region, genre_ids)
        else:
            # For subsequent pages, load from cache or fetch more
            all_releases = await self._fetch_additional_data(
                windows, region, genre_ids, page, content_type
            )
        
        # Sort all releases with Telugu priority
        for key in all_releases:
            all_releases[key] = self._sort_cinebrain_releases_telugu_first(all_releases[key])
        
        # Apply filters
        for key in all_releases:
            all_releases[key] = self._filter_cinebrain_releases(
                all_releases[key], language, genre, months
            )
        
        # Paginate results
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        
        paginated_releases = {
            "movies": all_releases["movies"][start_idx:end_idx],
            "tv_series": all_releases["tv_series"][start_idx:end_idx],
            "anime": all_releases["anime"][start_idx:end_idx]
        }
        
        # Calculate totals
        total_movies = len(all_releases["movies"])
        total_tv = len(all_releases["tv_series"])
        total_anime = len(all_releases["anime"])
        
        total_content = total_movies + total_tv + total_anime
        telugu_content = sum(
            1 for releases in all_releases.values() 
            for release in releases 
            if release.is_telugu_content
        )
        
        response = {
            "movies": [r.to_dict() for r in paginated_releases["movies"]],
            "tv_series": [r.to_dict() for r in paginated_releases["tv_series"]],
            "anime": [r.to_dict() for r in paginated_releases["anime"]],
            "pagination": {
                "current_page": page,
                "per_page": limit,
                "total_movies": total_movies,
                "total_tv_series": total_tv,
                "total_anime": total_anime,
                "total_items": total_content,
                "has_more_movies": end_idx < total_movies,
                "has_more_tv": end_idx < total_tv,
                "has_more_anime": end_idx < total_anime,
                "has_more": end_idx < total_content
            },
            "metadata": {
                "cinebrain_service": "upcoming",
                "cinebrain_brand": "CineBrain Entertainment Platform",
                "region": region,
                "timezone": timezone_name,
                "content_type": content_type,
                "language_filter": language,
                "genre_filter": genre,
                "months_ahead": months,
                "windows": [
                    {
                        "label": w.month_label,
                        "month_number": w.month_number,
                        "start": w.start_date.isoformat(),
                        "end": w.end_date.isoformat(),
                        "is_current": w.is_current,
                        "total_days": w.total_days
                    }
                    for w in windows
                ],
                "cinebrain_language_priority_order": [lang.language_name for lang in priority_languages],
                "cinebrain_telugu_priority_enabled": True,
                "generated_at": datetime.now(pytz.timezone(timezone_name)).isoformat()
            },
            "statistics": {
                "total_movies": total_movies,
                "total_tv_series": total_tv, 
                "total_anime": total_anime,
                "total_content": total_content,
                "cinebrain_telugu_content_count": telugu_content,
                "cinebrain_telugu_percentage": round((telugu_content / max(1, total_content)) * 100, 1),
                "cache_used": False,
                "cinebrain_fetch_time": datetime.now().isoformat()
            },
            "filters_applied": {
                "content_type": content_type,
                "language": language,
                "genre": genre,
                "months": months,
                "region": region
            },
            "cinebrain_success": True
        }
        
        if use_cache and self.cache:
            cache_key = self._generate_cinebrain_cache_key(
                region, timezone_name, content_type, language, genre, months, page, limit
            )
            await self._save_cinebrain_to_cache(cache_key, response, ttl=900)  # 15 min cache
        
        return response
    
    async def _fetch_additional_data(
        self,
        windows: List[ReleaseWindow],
        region: str,
        genre_ids: Optional[List[int]],
        page: int,
        content_type: str
    ) -> Dict[str, List[CinebrainUpcomingRelease]]:
        """Fetch additional data for pagination"""
        all_releases = {
            "movies": [],
            "tv_series": [],
            "anime": []
        }
        
        # Calculate which TMDB page to fetch based on our page
        tmdb_page = ((page - 1) // 2) + 1
        
        window = windows[0]  # Use first window for now
        
        if content_type in ['all', 'movies']:
            data = await self.tmdb_client.discover_cinebrain_upcoming_movies(
                region=region,
                start_date=window.start_date,
                end_date=window.end_date,
                language=None,
                with_genres=genre_ids,
                page=tmdb_page
            )
            
            for item in data.get("results", []):
                release = await self._parse_cinebrain_tmdb_movie(item, region)
                all_releases["movies"].append(release)
        
        if content_type in ['all', 'tv']:
            data = await self.tmdb_client.discover_cinebrain_upcoming_tv(
                region=region,
                start_date=window.start_date,
                end_date=window.end_date,
                language=None,
                with_genres=genre_ids,
                page=tmdb_page
            )
            
            for item in data.get("results", []):
                release = await self._parse_cinebrain_tmdb_tv(item, region)
                all_releases["tv_series"].append(release)
        
        if content_type in ['all', 'anime'] and tmdb_page <= 2:
            items = await self.anilist_client.get_cinebrain_upcoming_anime(
                start_date=window.start_date,
                end_date=window.end_date,
                page=tmdb_page
            )
            
            for item in items:
                release = await self._parse_cinebrain_anilist_anime(item)
                if release:
                    all_releases["anime"].append(release)
        
        return all_releases
    
    async def _get_cinebrain_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.cache:
            return None
        
        try:
            cached = self.cache.get(f"cinebrain_upcoming_lazy:{key}")
            if cached:
                data = json.loads(cached) if isinstance(cached, str) else cached
                if data and "statistics" in data:
                    data["statistics"]["cache_used"] = True
                return data
        except Exception as e:
            logger.error(f"CineBrain cache retrieval error: {e}")
        
        return None
    
    async def _save_cinebrain_to_cache(self, key: str, data: Dict[str, Any], ttl: int = 900):
        if not self.cache:
            return
        
        try:
            self.cache.set(
                f"cinebrain_upcoming_lazy:{key}",
                json.dumps(data, default=str),
                timeout=ttl
            )
            logger.info(f"CineBrain cached upcoming releases: cinebrain_upcoming_lazy:{key}")
        except Exception as e:
            logger.error(f"CineBrain cache save error: {e}")
    
    async def close(self):
        if self.http_client:
            await self.http_client.aclose()