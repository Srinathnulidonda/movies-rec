# backend/services/new_releases.py
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import pytz
import httpx
import hashlib
import json
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class NewRelease:
    """Data structure for newly released content"""
    id: str
    title: str
    original_title: Optional[str]
    content_type: str
    release_date: datetime
    languages: List[str]
    genres: List[str]
    popularity: float
    vote_count: int
    vote_average: float
    poster_path: Optional[str]
    backdrop_path: Optional[str]
    overview: Optional[str]
    runtime: Optional[int] = None
    youtube_trailer_id: Optional[str] = None
    
    # Release tracking
    days_since_release: int = 0
    hours_since_release: int = 0
    minutes_since_release: int = 0
    is_today: bool = False
    is_yesterday: bool = False
    is_this_week: bool = False
    release_timestamp: Optional[float] = None
    
    # Language priority
    is_telugu: bool = False
    language_priority: int = 999
    language_name: str = "Other"
    
    # Scoring
    freshness_score: float = 0.0
    quality_score: float = 0.0
    combined_score: float = 0.0
    priority_boost: float = 0.0
    
    def __post_init__(self):
        """Calculate all metrics post initialization"""
        self._set_release_timestamp()
        self._detect_language()
        self._calculate_scores()
    
    def calculate_time_metrics(self, user_timezone: str = 'UTC'):
        """Calculate time-based metrics in user's timezone"""
        try:
            tz = pytz.timezone(user_timezone)
        except:
            tz = pytz.UTC
        
        now = datetime.now(tz)
        
        # Convert release date to user's timezone
        if self.release_date.tzinfo is None:
            release_dt = pytz.UTC.localize(self.release_date)
        else:
            release_dt = self.release_date
        
        release_in_user_tz = release_dt.astimezone(tz)
        
        # Calculate time differences
        delta = now - release_in_user_tz
        self.days_since_release = max(0, delta.days)
        self.hours_since_release = max(0, int(delta.total_seconds() / 3600))
        self.minutes_since_release = max(0, int(delta.total_seconds() / 60))
        
        # Check if released today in user's timezone
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        yesterday_start = today_start - timedelta(days=1)
        
        self.is_today = today_start <= release_in_user_tz < today_end
        self.is_yesterday = yesterday_start <= release_in_user_tz < today_start
        self.is_this_week = self.days_since_release <= 7
        
        # Recalculate scores after updating time metrics
        self._calculate_scores()
    
    def _set_release_timestamp(self):
        """Set Unix timestamp for release date"""
        if self.release_date:
            self.release_timestamp = self.release_date.timestamp()
    
    def _detect_language(self):
        """Detect language and set priority"""
        language_priority_map = {
            ('telugu', 'te'): (1, 'Telugu'),
            ('english', 'en'): (2, 'English'),
            ('hindi', 'hi'): (3, 'Hindi'),
            ('malayalam', 'ml'): (4, 'Malayalam'),
            ('kannada', 'kn'): (5, 'Kannada'),
            ('tamil', 'ta'): (6, 'Tamil')
        }
        
        for lang in self.languages:
            lang_lower = lang.lower() if lang else ''
            
            for lang_keys, (priority, name) in language_priority_map.items():
                if lang_lower in lang_keys:
                    self.language_priority = priority
                    self.language_name = name
                    self.is_telugu = (priority == 1)
                    return
        
        self.language_priority = 999
        self.language_name = "Other"
        self.is_telugu = False
    
    def _calculate_scores(self):
        """Calculate all scoring metrics"""
        if self.is_today:
            self.freshness_score = 100.0
            self.priority_boost = 50.0
        elif self.is_yesterday:
            self.freshness_score = 80.0
            self.priority_boost = 20.0
        elif self.days_since_release <= 3:
            self.freshness_score = 70.0 - (self.days_since_release * 5)
            self.priority_boost = 10.0
        elif self.is_this_week:
            self.freshness_score = 50.0 - (self.days_since_release * 3)
            self.priority_boost = 5.0
        else:
            self.freshness_score = max(10, 40 - (self.days_since_release * 1))
            self.priority_boost = 0.0
        
        rating_score = (self.vote_average / 10) * 40 if self.vote_average > 0 else 20
        popularity_score = min(30, self.popularity / 10) if self.popularity > 0 else 10
        vote_score = min(30, self.vote_count / 100) if self.vote_count > 0 else 10
        self.quality_score = rating_score + popularity_score + vote_score
        
        base_score = (self.freshness_score * 0.7) + (self.quality_score * 0.3) + self.priority_boost
        
        if self.is_telugu:
            base_score *= 2.0
        elif self.language_priority == 2:
            base_score *= 1.5
        elif self.language_priority == 3:
            base_score *= 1.3
        elif self.language_priority <= 6:
            base_score *= 1.1
        
        self.combined_score = min(200, base_score)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if self.release_date:
            data['release_date'] = self.release_date.isoformat()
        return data


class RealTimeNewReleasesService:
    """Service for real-time new releases with continuous updates"""
    
    def __init__(
        self,
        tmdb_api_key: str,
        cache_backend: Optional[Any] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        refresh_interval_minutes: int = 5
    ):
        self.tmdb_api_key = tmdb_api_key
        self.cache = cache_backend
        self.http_client = http_client or httpx.AsyncClient(timeout=10.0)
        self.refresh_interval = refresh_interval_minutes
        self.base_url = "https://api.themoviedb.org/3"
        
        self.telugu_patterns = {
            'languages': ['te', 'telugu'],
            'keywords': [
                'tollywood', 'telugu cinema', 'telugu movie', 'telugu film',
                'andhra', 'telangana', 'hyderabad'
            ],
            'production_companies': [
                'AVM Productions', 'Vyjayanthi Movies', 'Mythri Movie Makers',
                'Geetha Arts', 'Sri Venkateswara Creations', 'Konidela Production Company',
                'UV Creations', 'Haarika & Hassine Creations', 'Sithara Entertainments'
            ]
        }
        
        self.last_fetch_time = None
        self.fetch_lock = asyncio.Lock()
    
    def _is_telugu_content(self, item: Dict[str, Any]) -> bool:
        """Enhanced Telugu content detection"""
        if item.get('original_language', '').lower() in self.telugu_patterns['languages']:
            return True
        
        text_to_check = ' '.join([
            item.get('title', ''),
            item.get('original_title', ''),
            item.get('overview', '')
        ]).lower()
        
        for keyword in self.telugu_patterns['keywords']:
            if keyword in text_to_check:
                return True
        
        return False
    
    async def fetch_by_language(
        self,
        language_code: str,
        days_back: int,
        content_type: str = 'movie'
    ) -> List[Dict[str, Any]]:
        """Fetch releases for a specific language"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        url = f"{self.base_url}/discover/{content_type}"
        params = {
            'api_key': self.tmdb_api_key,
            'with_original_language': language_code,
            'sort_by': 'release_date.desc' if content_type == 'movie' else 'first_air_date.desc',
            'page': 1,
            'include_adult': False
        }
        
        if content_type == 'movie':
            params['primary_release_date.gte'] = start_date.strftime('%Y-%m-%d')
            params['primary_release_date.lte'] = end_date.strftime('%Y-%m-%d')
        else:
            params['first_air_date.gte'] = start_date.strftime('%Y-%m-%d')
            params['first_air_date.lte'] = end_date.strftime('%Y-%m-%d')
        
        all_results = []
        
        try:
            for page in range(1, 4):
                params['page'] = page
                response = await self.http_client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    all_results.extend(results)
                    
                    if not results or page >= data.get('total_pages', 1):
                        break
                else:
                    logger.warning(f"API returned status {response.status_code} for {language_code}")
                    break
            
            return all_results
        except Exception as e:
            logger.error(f"Error fetching {language_code} releases: {e}")
            return []
    
    async def fetch_all_priority_languages(
        self,
        days_back: int,
        content_type: str = 'movie'
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch releases for all priority languages in order"""
        languages = [
            ('telugu', 'te'),
            ('english', 'en'),
            ('hindi', 'hi'),
            ('malayalam', 'ml'),
            ('kannada', 'kn'),
            ('tamil', 'ta')
        ]
        
        results = {}
        
        for lang_name, lang_code in languages:
            logger.info(f"Fetching {lang_name} releases...")
            lang_results = await self.fetch_by_language(lang_code, days_back, content_type)
            results[lang_name] = lang_results
            await asyncio.sleep(0.5)
        
        general_results = await self.fetch_general_releases(days_back, content_type)
        
        telugu_from_general = []
        other_from_general = []
        
        for item in general_results:
            if self._is_telugu_content(item):
                telugu_from_general.append(item)
            else:
                other_from_general.append(item)
        
        results['telugu'].extend(telugu_from_general)
        results['other'] = other_from_general
        
        return results
    
    async def fetch_general_releases(
        self,
        days_back: int,
        content_type: str = 'movie'
    ) -> List[Dict[str, Any]]:
        """Fetch general/popular releases"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        url = f"{self.base_url}/discover/{content_type}"
        params = {
            'api_key': self.tmdb_api_key,
            'sort_by': 'popularity.desc',
            'page': 1,
            'include_adult': False
        }
        
        if content_type == 'movie':
            params['primary_release_date.gte'] = start_date.strftime('%Y-%m-%d')
            params['primary_release_date.lte'] = end_date.strftime('%Y-%m-%d')
        else:
            params['first_air_date.gte'] = start_date.strftime('%Y-%m-%d')
            params['first_air_date.lte'] = end_date.strftime('%Y-%m-%d')
        
        try:
            response = await self.http_client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])[:50]
        except Exception as e:
            logger.error(f"Error fetching general releases: {e}")
        
        return []
    
    def parse_release(
        self,
        item: Dict[str, Any],
        content_type: str,
        detected_language: Optional[str] = None
    ) -> NewRelease:
        """Parse API response into NewRelease object"""
        if content_type == 'movie':
            release_date_str = item.get('release_date', '')
        else:
            release_date_str = item.get('first_air_date', '')
        
        try:
            release_date = datetime.strptime(release_date_str, '%Y-%m-%d')
            release_date = pytz.UTC.localize(release_date)
        except:
            release_date = datetime.now(pytz.UTC)
        
        orig_lang = item.get('original_language', '')
        is_telugu = self._is_telugu_content(item) or detected_language == 'telugu'
        
        if is_telugu:
            languages = ['te', orig_lang] if orig_lang != 'te' else ['te']
        elif detected_language:
            lang_code_map = {
                'english': 'en', 'hindi': 'hi', 'malayalam': 'ml',
                'kannada': 'kn', 'tamil': 'ta'
            }
            lang_code = lang_code_map.get(detected_language, orig_lang)
            languages = [lang_code, orig_lang] if lang_code != orig_lang else [lang_code]
        else:
            languages = [orig_lang] if orig_lang else []
        
        poster_path = item.get('poster_path')
        if poster_path and not poster_path.startswith('http'):
            poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        backdrop_path = item.get('backdrop_path')
        if backdrop_path and not backdrop_path.startswith('http'):
            backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        
        genre_map = {
            28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
            80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
            14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
            9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
            10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
        }
        genres = [genre_map.get(gid, "Unknown") for gid in item.get('genre_ids', [])]
        
        return NewRelease(
            id=f"tmdb_{content_type}_{item['id']}",
            title=item.get('title') or item.get('name', ''),
            original_title=item.get('original_title') or item.get('original_name'),
            content_type=content_type,
            release_date=release_date,
            languages=languages,
            genres=genres,
            popularity=item.get('popularity', 0),
            vote_count=item.get('vote_count', 0),
            vote_average=item.get('vote_average', 0),
            poster_path=poster_path,
            backdrop_path=backdrop_path,
            overview=item.get('overview'),
            runtime=item.get('runtime')
        )
    
    async def get_new_releases(
        self,
        content_type: str = 'movie',
        days_back: int = 60,
        user_timezone: str = 'UTC',
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Get new releases with real-time updates and strict language priority"""
        async with self.fetch_lock:
            current_minute = datetime.now().replace(second=0, microsecond=0)
            cache_window = int(current_minute.timestamp() // (self.refresh_interval * 60))
            cache_key = f"realtime_releases:{content_type}:{days_back}:{user_timezone}:{cache_window}"
            
            if use_cache and not force_refresh and self.cache:
                cached_data = self.cache.get(cache_key)
                if cached_data:
                    logger.info(f"Using cached new releases")
                    try:
                        return json.loads(cached_data) if isinstance(cached_data, str) else cached_data
                    except Exception as e:
                        logger.error(f"Cache deserialization error: {e}")
            
            logger.info(f"Fetching real-time new releases for {content_type}")
            start_time = datetime.now()
            
            all_language_results = await self.fetch_all_priority_languages(days_back, content_type)
            
            all_releases = []
            seen_ids = set()
            
            language_order = ['telugu', 'english', 'hindi', 'malayalam', 'kannada', 'tamil', 'other']
            
            for lang in language_order:
                lang_items = all_language_results.get(lang, [])
                for item in lang_items:
                    release = self.parse_release(item, content_type, detected_language=lang)
                    release.calculate_time_metrics(user_timezone)
                    
                    if release.id not in seen_ids:
                        all_releases.append(release)
                        seen_ids.add(release.id)
            
            def sort_key(release):
                return (
                    not release.is_today,
                    release.language_priority,
                    -release.combined_score,
                    release.days_since_release,
                    -release.popularity
                )
            
            all_releases.sort(key=sort_key)
            
            todays_releases = [r for r in all_releases if r.is_today]
            
            stats = self._calculate_statistics(all_releases, todays_releases, user_timezone)
            
            response = {
                'releases': [r.to_dict() for r in all_releases],
                'todays_releases': [r.to_dict() for r in todays_releases],
                'metadata': {
                    'total_count': len(all_releases),
                    'todays_count': len(todays_releases),
                    'user_timezone': user_timezone,
                    'refresh_interval_minutes': self.refresh_interval,
                    'last_updated': datetime.now(pytz.UTC).isoformat(),
                    'next_update': (datetime.now(pytz.UTC) + timedelta(minutes=self.refresh_interval)).isoformat(),
                    'fetch_duration_seconds': (datetime.now() - start_time).total_seconds(),
                    'language_priority': ['Telugu', 'English', 'Hindi', 'Malayalam', 'Kannada', 'Tamil']
                },
                'statistics': stats
            }
            
            if use_cache and self.cache:
                cache_data = json.dumps(response, default=str)
                self.cache.set(
                    cache_key,
                    cache_data,
                    timeout=self.refresh_interval * 60
                )
                logger.info(f"Cached new releases")
            
            self.last_fetch_time = datetime.now()
            
            return response
    
    def _calculate_statistics(
        self,
        all_releases: List[NewRelease],
        todays_releases: List[NewRelease],
        user_timezone: str
    ) -> Dict[str, Any]:
        """Calculate detailed statistics"""
        lang_dist = defaultdict(int)
        for release in all_releases:
            lang_dist[release.language_name] += 1
        
        time_dist = {
            'today': len([r for r in all_releases if r.is_today]),
            'yesterday': len([r for r in all_releases if r.is_yesterday]),
            'last_3_days': len([r for r in all_releases if r.days_since_release <= 3]),
            'this_week': len([r for r in all_releases if r.is_this_week]),
            'older': len([r for r in all_releases if r.days_since_release > 7])
        }
        
        telugu_releases = [r for r in all_releases if r.is_telugu]
        telugu_today = [r for r in todays_releases if r.is_telugu]
        
        genre_dist = defaultdict(int)
        for release in todays_releases:
            for genre in release.genres:
                genre_dist[genre] += 1
        
        return {
            'language_distribution': dict(lang_dist),
            'time_distribution': time_dist,
            'telugu_stats': {
                'total_count': len(telugu_releases),
                'percentage': round(len(telugu_releases) / len(all_releases) * 100, 1) if all_releases else 0,
                'todays_count': len(telugu_today),
                'todays_percentage': round(len(telugu_today) / len(todays_releases) * 100, 1) if todays_releases else 0
            },
            'todays_genre_distribution': dict(genre_dist),
            'average_rating': round(
                sum(r.vote_average for r in all_releases if r.vote_average > 0) / 
                max(1, len([r for r in all_releases if r.vote_average > 0])), 1
            ) if all_releases else 0
        }
    
    async def close(self):
        """Clean up resources"""
        if self.http_client:
            await self.http_client.aclose()