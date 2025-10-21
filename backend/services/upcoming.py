#backend/services/upcoming.py
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
    MOVIE = "movie"
    TV_SERIES = "tv"
    ANIME = "anime"


class CineBrainLanguagePriority(Enum):
    TELUGU = ("telugu", "te", 1)
    ENGLISH = ("english", "en", 2)
    HINDI = ("hindi", "hi", 3)
    MALAYALAM = ("malayalam", "ml", 4)
    KANNADA = ("kannada", "kn", 5)
    TAMIL = ("tamil", "ta", 6)
    OTHER = ("other", "other", 7)
    
    def __init__(self, name: str, iso_code: str, priority: int):
        self.language_name = name
        self.iso_code = iso_code
        self.priority = priority
    
    @classmethod
    def get_by_iso(cls, iso_code: str) -> Optional['CineBrainLanguagePriority']:
        for lang in cls:
            if lang.iso_code == iso_code:
                return lang
        return cls.OTHER
    
    @classmethod
    def get_priority_order(cls) -> List['CineBrainLanguagePriority']:
        return sorted(cls, key=lambda x: x.priority)


@dataclass
class CineBrainReleaseWindow:
    start_date: datetime
    end_date: datetime
    month_label: str
    priority: int
    is_current: bool = False


@dataclass
class CineBrainUpcomingRelease:
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
    
    anticipation_score: float = 0.0
    buzz_level: str = "normal"
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
    marketing_level: str = "standard"
    release_strategy: str = "wide"
    days_until_release: int = 0
    is_telugu_priority: bool = False
    exact_release_date: Optional[str] = None
    
    def __post_init__(self):
        self._set_language_priority()
        self._calculate_days_until_release()
        self._determine_telugu_priority()
        self._set_exact_release_date()
    
    def _set_language_priority(self):
        for lang in self.languages:
            for lang_enum in CineBrainLanguagePriority:
                if lang.lower() in [lang_enum.iso_code, lang_enum.language_name]:
                    self.language_priority = min(self.language_priority, lang_enum.priority)
                    break
    
    def _calculate_days_until_release(self):
        if self.release_date:
            delta = self.release_date - datetime.now()
            self.days_until_release = delta.days
    
    def _determine_telugu_priority(self):
        telugu_identifiers = ['te', 'telugu', 'tollywood']
        for lang in self.languages:
            if lang.lower() in telugu_identifiers:
                self.is_telugu_priority = True
                break
    
    def _set_exact_release_date(self):
        if self.release_date:
            self.exact_release_date = self.release_date.strftime("%Y-%m-%d")
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.release_date:
            data['release_date'] = self.release_date.isoformat()
        data['content_type'] = self.content_type.value
        return data


class CineBrainAnticipationAnalyzer:
    
    @staticmethod
    def calculate_anticipation_score(release: CineBrainUpcomingRelease) -> float:
        score = 0.0
        
        if release.popularity > 0:
            score += min(30, release.popularity / 10)
        
        if release.vote_average > 0:
            score += (release.vote_average / 10) * 20
        
        if release.language_priority == 1:
            score += 25
        elif release.language_priority == 2:
            score += 18
        elif release.language_priority == 3:
            score += 12
        elif release.language_priority <= 6:
            score += 8
        
        if release.is_franchise:
            score += 12
        
        if 0 <= release.days_until_release <= 7:
            score += 15
        elif 8 <= release.days_until_release <= 30:
            score += 10
        elif 31 <= release.days_until_release <= 60:
            score += 7
        elif 61 <= release.days_until_release <= 90:
            score += 5
        
        marketing_scores = {
            "blockbuster": 15,
            "heavy": 10,
            "standard": 5,
            "minimal": 2
        }
        score += marketing_scores.get(release.marketing_level, 5)
        
        return min(100, score)
    
    @staticmethod
    def determine_buzz_level(release: CineBrainUpcomingRelease) -> str:
        if release.anticipation_score >= 85:
            return "viral"
        elif release.anticipation_score >= 65:
            return "high"
        elif release.anticipation_score >= 35:
            return "normal"
        else:
            return "low"


class CineBrainTMDbClient:
    
    BASE_URL = "https://api.themoviedb.org/3"
    
    def __init__(self, api_key: str, http_client: Optional[httpx.AsyncClient] = None):
        self.api_key = api_key
        self.http_client = http_client or httpx.AsyncClient(timeout=15.0)
    
    async def discover_upcoming_movies(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime,
        language: Optional[str] = None,
        with_genres: Optional[List[int]] = None,
        sort_by: str = "primary_release_date.asc,popularity.desc"
    ) -> List[Dict[str, Any]]:
        params = {
            "api_key": self.api_key,
            "region": region,
            "primary_release_date.gte": start_date.strftime("%Y-%m-%d"),
            "primary_release_date.lte": end_date.strftime("%Y-%m-%d"),
            "with_release_type": "2|3|4|5|6",
            "sort_by": sort_by,
            "include_adult": False,
            "page": 1,
            "vote_count.gte": 1
        }
        
        if language:
            params["with_original_language"] = language
        
        if with_genres:
            params["with_genres"] = ",".join(map(str, with_genres))
        
        all_results = []
        
        try:
            for page in range(1, 6):
                params["page"] = page
                response = await self.http_client.get(
                    f"{self.BASE_URL}/discover/movie",
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])
                
                for result in results:
                    release_date = result.get("release_date")
                    if release_date:
                        try:
                            result_date = datetime.strptime(release_date, "%Y-%m-%d")
                            if start_date <= result_date <= end_date:
                                all_results.append(result)
                        except ValueError:
                            continue
                
                if not results or page >= data.get("total_pages", 1):
                    break
            
            return all_results
        except Exception as e:
            logger.error(f"CineBrain TMDb movie discovery error: {e}")
            return []
    
    async def discover_upcoming_tv(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime,
        language: Optional[str] = None,
        with_genres: Optional[List[int]] = None,
        sort_by: str = "first_air_date.asc,popularity.desc"
    ) -> List[Dict[str, Any]]:
        params = {
            "api_key": self.api_key,
            "watch_region": region,
            "first_air_date.gte": start_date.strftime("%Y-%m-%d"),
            "first_air_date.lte": end_date.strftime("%Y-%m-%d"),
            "sort_by": sort_by,
            "include_adult": False,
            "page": 1,
            "vote_count.gte": 1
        }
        
        if language:
            params["with_original_language"] = language
        
        if with_genres:
            params["with_genres"] = ",".join(map(str, with_genres))
        
        all_results = []
        
        try:
            for page in range(1, 5):
                params["page"] = page
                response = await self.http_client.get(
                    f"{self.BASE_URL}/discover/tv",
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])
                
                for result in results:
                    air_date = result.get("first_air_date")
                    if air_date:
                        try:
                            result_date = datetime.strptime(air_date, "%Y-%m-%d")
                            if start_date <= result_date <= end_date:
                                all_results.append(result)
                        except ValueError:
                            continue
                
                if not results or page >= data.get("total_pages", 1):
                    break
            
            return all_results
        except Exception as e:
            logger.error(f"CineBrain TMDb TV discovery error: {e}")
            return []
    
    async def get_movie_details(self, movie_id: int) -> Dict[str, Any]:
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
            logger.error(f"CineBrain TMDb movie details error: {e}")
            return {}
    
    async def get_tv_details(self, tv_id: int) -> Dict[str, Any]:
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
            logger.error(f"CineBrain TMDb TV details error: {e}")
            return {}


class CineBrainAniListClient:
    
    BASE_URL = "https://graphql.anilist.co"
    
    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self.http_client = http_client or httpx.AsyncClient(timeout=15.0)
    
    def _build_upcoming_query(self) -> str:
        return """
        query ($startDate: Int, $endDate: Int, $page: Int, $sort: [MediaSort]) {
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
        if sort is None:
            sort = ["START_DATE", "POPULARITY_DESC"]
        
        start_int = int(start_date.strftime("%Y%m%d"))
        end_int = int(end_date.strftime("%Y%m%d"))
        
        variables = {
            "startDate": start_int,
            "endDate": end_int,
            "page": 1,
            "sort": sort
        }
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            
            all_results = []
            
            for page in range(1, 4):
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
                    logger.error(f"CineBrain AniList GraphQL errors: {data['errors']}")
                    break
                
                results = data.get("data", {}).get("Page", {}).get("media", [])
                
                for result in results:
                    start_date_info = result.get("startDate", {})
                    if start_date_info and start_date_info.get("year"):
                        try:
                            result_date = datetime(
                                start_date_info.get("year"),
                                start_date_info.get("month", 1),
                                start_date_info.get("day", 1)
                            )
                            if start_date <= result_date <= end_date:
                                all_results.append(result)
                        except ValueError:
                            continue
                
                page_info = data.get("data", {}).get("Page", {}).get("pageInfo", {})
                if not page_info.get("hasNextPage", False):
                    break
            
            return all_results
            
        except Exception as e:
            logger.error(f"CineBrain AniList upcoming query error: {e}")
            return []


class CineBrainUpcomingContentService:
    
    def __init__(
        self,
        tmdb_api_key: str,
        cache_backend: Optional[Any] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        enable_analytics: bool = True
    ):
        self.tmdb_client = CineBrainTMDbClient(tmdb_api_key, http_client)
        self.anilist_client = CineBrainAniListClient(http_client)
        self.cache = cache_backend
        self.http_client = http_client or httpx.AsyncClient(timeout=15.0)
        self.anticipation_analyzer = CineBrainAnticipationAnalyzer()
        self.enable_analytics = enable_analytics
        
        self.cinebrain_telugu_patterns = {
            'languages': ['te', 'telugu'],
            'keywords': ['tollywood', 'telugu cinema', 'andhra', 'telangana', 'hyderabad', 'vijayawada'],
            'studios': [
                'AVM Productions', 'Vyjayanthi Movies', 'Mythri Movie Makers',
                'Geetha Arts', 'Sri Venkateswara Creations', 'Konidela Production Company',
                'Haarika & Hassine Creations', 'UV Creations', 'GA2 Pictures',
                'Sithara Entertainments', 'People Media Factory', 'Asian Cinemas'
            ]
        }
    
    def _get_cinebrain_release_windows(self, user_timezone: str) -> List[CineBrainReleaseWindow]:
        try:
            tz = pytz.timezone(user_timezone)
        except pytz.UnknownTimeZoneError:
            logger.warning(f"CineBrain: Unknown timezone {user_timezone}, using UTC")
            tz = pytz.UTC
        
        now = datetime.now(tz)
        
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=90)
        
        windows = []
        
        current_month_start = start_date
        for i in range(3):
            month_start = current_month_start + timedelta(days=30 * i)
            month_end = month_start + timedelta(days=29)
            
            if month_end > end_date:
                month_end = end_date
            
            windows.append(CineBrainReleaseWindow(
                start_date=month_start,
                end_date=month_end,
                month_label=f"month_{i+1}" if i > 0 else "current",
                priority=i + 1,
                is_current=(i == 0)
            ))
            
            if month_end >= end_date:
                break
        
        return windows
    
    def _is_cinebrain_telugu_content(self, item: Dict[str, Any]) -> bool:
        original_language = item.get("original_language", "").lower()
        if original_language in self.cinebrain_telugu_patterns['languages']:
            return True
        
        title = (item.get("title", "") + " " + item.get("original_title", "") + " " + item.get("name", "") + " " + item.get("original_name", "")).lower()
        for keyword in self.cinebrain_telugu_patterns['keywords']:
            if keyword in title:
                return True
        
        return False
    
    def _parse_cinebrain_movie(self, item: Dict[str, Any], region: str) -> CineBrainUpcomingRelease:
        release_date_str = item.get("release_date", "")
        try:
            release_date = datetime.strptime(release_date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            release_date = datetime.now() + timedelta(days=30)
        
        original_lang = item.get("original_language", "")
        
        is_telugu = self._is_cinebrain_telugu_content(item)
        if is_telugu:
            languages = ["te", original_lang] if original_lang != "te" else ["te"]
        else:
            languages = [original_lang] if original_lang else []
        
        poster_path = item.get("poster_path")
        if poster_path and not poster_path.startswith("http"):
            poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        backdrop_path = item.get("backdrop_path")
        if backdrop_path and not backdrop_path.startswith("http"):
            backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        
        marketing_level = "minimal"
        popularity = item.get("popularity", 0)
        if popularity > 150:
            marketing_level = "blockbuster"
        elif popularity > 75:
            marketing_level = "heavy"
        elif popularity > 25:
            marketing_level = "standard"
        
        release = CineBrainUpcomingRelease(
            id=f"cinebrain_movie_{item['id']}",
            title=item.get("title", ""),
            original_title=item.get("original_title"),
            content_type=ContentType.MOVIE,
            release_date=release_date,
            languages=languages,
            genres=self._get_cinebrain_genre_names(item.get("genre_ids", [])),
            popularity=popularity,
            vote_count=item.get("vote_count", 0),
            vote_average=item.get("vote_average", 0),
            poster_path=poster_path,
            backdrop_path=backdrop_path,
            overview=item.get("overview"),
            region=region,
            source="cinebrain_tmdb",
            marketing_level=marketing_level
        )
        
        if self.enable_analytics:
            release.anticipation_score = self.anticipation_analyzer.calculate_anticipation_score(release)
            release.buzz_level = self.anticipation_analyzer.determine_buzz_level(release)
        
        return release
    
    def _parse_cinebrain_tv(self, item: Dict[str, Any], region: str) -> CineBrainUpcomingRelease:
        air_date_str = item.get("first_air_date", "")
        try:
            air_date = datetime.strptime(air_date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            air_date = datetime.now() + timedelta(days=30)
        
        original_lang = item.get("original_language", "")
        
        is_telugu = self._is_cinebrain_telugu_content(item)
        if is_telugu:
            languages = ["te", original_lang] if original_lang != "te" else ["te"]
        else:
            languages = [original_lang] if original_lang else []
        
        poster_path = item.get("poster_path")
        if poster_path and not poster_path.startswith("http"):
            poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        backdrop_path = item.get("backdrop_path")
        if backdrop_path and not backdrop_path.startswith("http"):
            backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        
        release = CineBrainUpcomingRelease(
            id=f"cinebrain_tv_{item['id']}",
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
            release_strategy="streaming" if item.get("popularity", 0) < 30 else "wide"
        )
        
        if self.enable_analytics:
            release.anticipation_score = self.anticipation_analyzer.calculate_anticipation_score(release)
            release.buzz_level = self.anticipation_analyzer.determine_buzz_level(release)
        
        return release
    
    def _parse_cinebrain_anime(self, item: Dict[str, Any]) -> Optional[CineBrainUpcomingRelease]:
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
        title = title_info.get("english") or title_info.get("romaji", "")
        
        if not title:
            return None
        
        cover_image = item.get("coverImage", {})
        poster_url = cover_image.get("extraLarge") or cover_image.get("large") or cover_image.get("medium")
        
        studios = item.get("studios", {}).get("nodes", [])
        studio_name = studios[0].get("name") if studios else None
        
        director = None
        staff = item.get("staff", {}).get("edges", [])
        for staff_member in staff:
            if staff_member.get("role", "").lower() == "director":
                director = staff_member.get("node", {}).get("name", {}).get("full")
                break
        
        release = CineBrainUpcomingRelease(
            id=f"cinebrain_anime_{item['id']}",
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
            studio=studio_name,
            director=director,
            is_franchise=bool(item.get("hashtag")),
            franchise_name=item.get("hashtag"),
            marketing_level="heavy" if item.get("trending", 0) > 15 else "standard"
        )
        
        if self.enable_analytics:
            release.anticipation_score = self.anticipation_analyzer.calculate_anticipation_score(release)
            release.buzz_level = self.anticipation_analyzer.determine_buzz_level(release)
        
        return release
    
    def _get_cinebrain_genre_names(self, genre_ids: List[int]) -> List[str]:
        cinebrain_genre_map = {
            28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
            80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
            14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
            9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
            10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
        }
        return [cinebrain_genre_map.get(gid, "Unknown") for gid in genre_ids if gid in cinebrain_genre_map]
    
    async def _fetch_cinebrain_movies_for_window(
        self,
        window: CineBrainReleaseWindow,
        region: str,
        languages: List[CineBrainLanguagePriority]
    ) -> List[CineBrainUpcomingRelease]:
        releases = []
        seen_ids = set()
        
        telugu_lang = languages[0]
        telugu_items = await self.tmdb_client.discover_upcoming_movies(
            region=region,
            start_date=window.start_date,
            end_date=window.end_date,
            language=telugu_lang.iso_code,
            sort_by="primary_release_date.asc,popularity.desc"
        )
        
        for item in telugu_items:
            movie_id = f"cinebrain_movie_{item['id']}"
            if movie_id not in seen_ids:
                release = self._parse_cinebrain_movie(item, region)
                release.is_telugu_priority = True
                releases.append(release)
                seen_ids.add(movie_id)
        
        for lang in languages[1:]:
            items = await self.tmdb_client.discover_upcoming_movies(
                region=region,
                start_date=window.start_date,
                end_date=window.end_date,
                language=lang.iso_code,
                sort_by="primary_release_date.asc,popularity.desc"
            )
            
            for item in items[:25]:
                movie_id = f"cinebrain_movie_{item['id']}"
                if movie_id not in seen_ids:
                    release = self._parse_cinebrain_movie(item, region)
                    releases.append(release)
                    seen_ids.add(movie_id)
        
        general_items = await self.tmdb_client.discover_upcoming_movies(
            region=region,
            start_date=window.start_date,
            end_date=window.end_date,
            sort_by="primary_release_date.asc,popularity.desc"
        )
        
        for item in general_items[:40]:
            movie_id = f"cinebrain_movie_{item['id']}"
            if movie_id not in seen_ids:
                release = self._parse_cinebrain_movie(item, region)
                if self._is_cinebrain_telugu_content(item):
                    release.is_telugu_priority = True
                releases.append(release)
                seen_ids.add(movie_id)
        
        return releases
    
    async def _fetch_cinebrain_tv_for_window(
        self,
        window: CineBrainReleaseWindow,
        region: str,
        languages: List[CineBrainLanguagePriority]
    ) -> List[CineBrainUpcomingRelease]:
        releases = []
        seen_ids = set()
        
        telugu_lang = languages[0]
        telugu_items = await self.tmdb_client.discover_upcoming_tv(
            region=region,
            start_date=window.start_date,
            end_date=window.end_date,
            language=telugu_lang.iso_code,
            sort_by="first_air_date.asc,popularity.desc"
        )
        
        for item in telugu_items:
            tv_id = f"cinebrain_tv_{item['id']}"
            if tv_id not in seen_ids:
                release = self._parse_cinebrain_tv(item, region)
                release.is_telugu_priority = True
                releases.append(release)
                seen_ids.add(tv_id)
        
        for lang in languages[1:]:
            items = await self.tmdb_client.discover_upcoming_tv(
                region=region,
                start_date=window.start_date,
                end_date=window.end_date,
                language=lang.iso_code,
                sort_by="first_air_date.asc,popularity.desc"
            )
            
            for item in items[:20]:
                tv_id = f"cinebrain_tv_{item['id']}"
                if tv_id not in seen_ids:
                    release = self._parse_cinebrain_tv(item, region)
                    releases.append(release)
                    seen_ids.add(tv_id)
        
        return releases
    
    async def _fetch_cinebrain_anime_for_window(
        self,
        window: CineBrainReleaseWindow
    ) -> List[CineBrainUpcomingRelease]:
        releases = []
        
        items = await self.anilist_client.get_upcoming_anime(
            start_date=window.start_date,
            end_date=window.end_date,
            sort=["START_DATE", "POPULARITY_DESC"]
        )
        
        for item in items:
            release = self._parse_cinebrain_anime(item)
            if release:
                releases.append(release)
        
        return releases
    
    def _sort_cinebrain_releases_by_date_and_priority(self, releases: List[CineBrainUpcomingRelease]) -> List[CineBrainUpcomingRelease]:
        telugu_releases = [r for r in releases if r.is_telugu_priority or r.language_priority == 1]
        other_releases = [r for r in releases if not (r.is_telugu_priority or r.language_priority == 1)]
        
        telugu_sorted = sorted(
            telugu_releases,
            key=lambda r: (
                r.release_date,
                -r.anticipation_score,
                -r.popularity
            )
        )
        
        other_sorted = sorted(
            other_releases,
            key=lambda r: (
                r.release_date,
                r.language_priority,
                -r.anticipation_score,
                -r.popularity
            )
        )
        
        return telugu_sorted + other_sorted
    
    def _generate_cinebrain_cache_key(
        self,
        region: str,
        timezone_name: str,
        categories: List[str]
    ) -> str:
        components = [
            "cinebrain_upcoming",
            region,
            timezone_name,
            "_".join(sorted(categories)),
            datetime.now().strftime("%Y%m%d%H")
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
        if categories is None:
            categories = ["movies", "tv", "anime"]
        
        self.enable_analytics = include_analytics
        
        if use_cache and self.cache:
            cache_key = self._generate_cinebrain_cache_key(region, timezone_name, categories)
            cached_data = await self._get_cinebrain_from_cache(cache_key)
            if cached_data:
                logger.info(f"CineBrain cache hit for upcoming releases: {cache_key}")
                return cached_data
        
        windows = self._get_cinebrain_release_windows(timezone_name)
        if not windows:
            return {"error": "CineBrain: No valid release windows"}
        
        priority_languages = CineBrainLanguagePriority.get_priority_order()
        
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
                "cinebrain_telugu_priority_enabled": True,
                "fetched_at": datetime.now(pytz.timezone(timezone_name)).isoformat(),
                "cinebrain_service": "upcoming_releases"
            }
        }
        
        for window in windows:
            tasks = []
            
            if "movies" in categories:
                tasks.append(("movies", self._fetch_cinebrain_movies_for_window(
                    window, region, priority_languages
                )))
            
            if "tv" in categories:
                tasks.append(("tv_series", self._fetch_cinebrain_tv_for_window(
                    window, region, priority_languages
                )))
            
            if "anime" in categories:
                tasks.append(("anime", self._fetch_cinebrain_anime_for_window(window)))
            
            for category, task in tasks:
                try:
                    releases = await task
                    results[category].extend(releases)
                except Exception as e:
                    logger.error(f"CineBrain error fetching {category}: {e}")
        
        results["movies"] = self._sort_cinebrain_releases_by_date_and_priority(results["movies"])
        results["tv_series"] = self._sort_cinebrain_releases_by_date_and_priority(results["tv_series"])
        results["anime"] = self._sort_cinebrain_releases_by_date_and_priority(results["anime"])
        
        telugu_movies = sum(1 for r in results["movies"] if r.is_telugu_priority)
        telugu_tv = sum(1 for r in results["tv_series"] if r.is_telugu_priority)
        
        results["metadata"]["cinebrain_statistics"] = {
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
                if r.anticipation_score >= 75
            ),
            "viral_buzz_count": sum(
                1 for r in results["movies"] + results["tv_series"] + results["anime"] 
                if r.buzz_level == "viral"
            ),
            "cache_used": False,
            "cinebrain_brand": "CineBrain Entertainment Platform"
        }
        
        results["movies"] = [r.to_dict() for r in results["movies"]]
        results["tv_series"] = [r.to_dict() for r in results["tv_series"]]
        results["anime"] = [r.to_dict() for r in results["anime"]]
        
        if use_cache and self.cache:
            cache_key = self._generate_cinebrain_cache_key(region, timezone_name, categories)
            await self._save_cinebrain_to_cache(cache_key, results, ttl=3600)
        
        return results
    
    async def _get_cinebrain_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.cache:
            return None
        
        try:
            cached = self.cache.get(f"cinebrain_upcoming:{key}")
            if cached:
                data = json.loads(cached) if isinstance(cached, str) else cached
                if data and "metadata" in data:
                    data["metadata"]["cinebrain_statistics"]["cache_used"] = True
                return data
        except Exception as e:
            logger.error(f"CineBrain cache retrieval error: {e}")
        
        return None
    
    async def _save_cinebrain_to_cache(self, key: str, data: Dict[str, Any], ttl: int = 3600):
        if not self.cache:
            return
        
        try:
            self.cache.set(
                f"cinebrain_upcoming:{key}",
                json.dumps(data, default=str),
                timeout=ttl
            )
            logger.info(f"CineBrain cached upcoming releases: cinebrain_upcoming:{key}")
        except Exception as e:
            logger.error(f"CineBrain cache save error: {e}")
    
    async def close(self):
        if self.http_client:
            await self.http_client.aclose()


UpcomingContentService = CineBrainUpcomingContentService