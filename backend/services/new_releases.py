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
    is_brand_new: bool = False  # Released in last 7 days
    is_fresh: bool = False  # Released in last 3 days
    is_today: bool = False  # Released today
    
    # Language priority
    is_telugu: bool = False
    language_priority: int = 999
    
    # Quality metrics
    quality_score: float = 0.0
    freshness_score: float = 0.0
    combined_score: float = 0.0
    
    def __post_init__(self):
        """Calculate release metrics"""
        self._calculate_time_since_release()
        self._detect_telugu()
        self._set_language_priority()
        self._calculate_scores()
    
    def _calculate_time_since_release(self):
        """Calculate time since release"""
        now = datetime.now()
        if self.release_date:
            delta = now - self.release_date
            self.days_since_release = max(0, delta.days)
            self.hours_since_release = max(0, int(delta.total_seconds() / 3600))
            
            self.is_today = self.days_since_release == 0
            self.is_fresh = self.days_since_release <= 3
            self.is_brand_new = self.days_since_release <= 7
    
    def _detect_telugu(self):
        """Detect if content is Telugu"""
        telugu_identifiers = ['te', 'telugu', 'tollywood']
        for lang in self.languages:
            if any(identifier in lang.lower() for identifier in telugu_identifiers):
                self.is_telugu = True
                break
    
    def _set_language_priority(self):
        """Set language priority (Telugu = 1, English = 2, etc.)"""
        priority_map = {
            'telugu': 1, 'te': 1,
            'english': 2, 'en': 2,
            'hindi': 3, 'hi': 3,
            'malayalam': 4, 'ml': 4,
            'kannada': 5, 'kn': 5,
            'tamil': 6, 'ta': 6
        }
        
        for lang in self.languages:
            lang_lower = lang.lower()
            if lang_lower in priority_map:
                self.language_priority = min(self.language_priority, priority_map[lang_lower])
    
    def _calculate_scores(self):
        """Calculate quality and freshness scores"""
        # Freshness score (100 for today, decreasing by day)
        if self.is_today:
            self.freshness_score = 100.0
        elif self.is_fresh:
            self.freshness_score = 100.0 - (self.days_since_release * 10)
        elif self.is_brand_new:
            self.freshness_score = 100.0 - (self.days_since_release * 5)
        else:
            self.freshness_score = max(0, 100.0 - (self.days_since_release * 2))
        
        # Quality score based on ratings and popularity
        if self.vote_average > 0:
            rating_component = (self.vote_average / 10) * 40
        else:
            rating_component = 20  # Default for unrated
        
        if self.popularity > 0:
            popularity_component = min(30, self.popularity / 10)
        else:
            popularity_component = 10
        
        if self.vote_count > 0:
            vote_component = min(30, self.vote_count / 100)
        else:
            vote_component = 10
        
        self.quality_score = rating_component + popularity_component + vote_component
        
        # Combined score with language boost
        base_score = (self.freshness_score * 0.6) + (self.quality_score * 0.4)
        
        # Telugu boost
        if self.is_telugu:
            base_score *= 1.5
        # Other language boosts
        elif self.language_priority == 2:  # English
            base_score *= 1.2
        elif self.language_priority == 3:  # Hindi
            base_score *= 1.1
        
        self.combined_score = min(100, base_score)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if self.release_date:
            data['release_date'] = self.release_date.isoformat()
        return data


class NewReleasesService:
    """Service for fetching and managing new releases with auto-refresh"""
    
    def __init__(
        self,
        tmdb_api_key: str,
        cache_backend: Optional[Any] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        refresh_interval_hours: int = 3
    ):
        self.tmdb_api_key = tmdb_api_key
        self.cache = cache_backend
        self.http_client = http_client or httpx.AsyncClient(timeout=10.0)
        self.refresh_interval = refresh_interval_hours
        self.base_url = "https://api.themoviedb.org/3"
        
        # Telugu detection patterns
        self.telugu_patterns = {
            'languages': ['te', 'telugu'],
            'keywords': ['tollywood', 'telugu cinema', 'telugu movie', 'telugu film'],
            'production_companies': [
                'AVM Productions', 'Vyjayanthi Movies', 'Mythri Movie Makers',
                'Geetha Arts', 'Sri Venkateswara Creations', 'Konidela Production Company',
                'UV Creations', 'Haarika & Hassine Creations', 'Sithara Entertainments'
            ]
        }
    
    def _is_telugu_content(self, item: Dict[str, Any]) -> bool:
        """Enhanced Telugu content detection"""
        # Check original language
        if item.get('original_language', '').lower() in self.telugu_patterns['languages']:
            return True
        
        # Check title and overview for Telugu keywords
        text_to_check = (
            item.get('title', '') + ' ' +
            item.get('original_title', '') + ' ' +
            item.get('overview', '')
        ).lower()
        
        for keyword in self.telugu_patterns['keywords']:
            if keyword in text_to_check:
                return True
        
        return False
    
    async def fetch_new_releases_by_language(
        self,
        language_code: str,
        days_back: int = 60,
        content_type: str = 'movie'
    ) -> List[Dict[str, Any]]:
        """Fetch new releases for a specific language"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        url = f"{self.base_url}/discover/{content_type}"
        params = {
            'api_key': self.tmdb_api_key,
            'primary_release_date.gte': start_date.strftime('%Y-%m-%d') if content_type == 'movie' else None,
            'primary_release_date.lte': end_date.strftime('%Y-%m-%d') if content_type == 'movie' else None,
            'first_air_date.gte': start_date.strftime('%Y-%m-%d') if content_type == 'tv' else None,
            'first_air_date.lte': end_date.strftime('%Y-%m-%d') if content_type == 'tv' else None,
            'with_original_language': language_code,
            'sort_by': 'release_date.desc' if content_type == 'movie' else 'first_air_date.desc',
            'page': 1
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        all_results = []
        
        try:
            # Fetch multiple pages for better coverage
            for page in range(1, 4):  # Get 3 pages
                params['page'] = page
                response = await self.http_client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                results = data.get('results', [])
                all_results.extend(results)
                
                if not results or page >= data.get('total_pages', 1):
                    break
            
            return all_results
        except Exception as e:
            logger.error(f"Error fetching {language_code} releases: {e}")
            return []
    
    async def fetch_all_new_releases(
        self,
        days_back: int = 60,
        content_type: str = 'movie'
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch new releases for all priority languages"""
        languages = {
            'telugu': 'te',
            'english': 'en',
            'hindi': 'hi',
            'malayalam': 'ml',
            'kannada': 'kn',
            'tamil': 'ta'
        }
        
        results = {}
        
        # Fetch Telugu FIRST
        telugu_results = await self.fetch_new_releases_by_language('te', days_back, content_type)
        results['telugu'] = telugu_results
        
        # Then fetch other languages in parallel
        tasks = []
        for lang_name, lang_code in languages.items():
            if lang_name != 'telugu':  # Skip Telugu as we already fetched it
                tasks.append((lang_name, self.fetch_new_releases_by_language(lang_code, days_back, content_type)))
        
        # Execute parallel fetches
        for lang_name, task in tasks:
            try:
                lang_results = await task
                results[lang_name] = lang_results
            except Exception as e:
                logger.error(f"Error fetching {lang_name} releases: {e}")
                results[lang_name] = []
        
        # Also fetch general/popular releases
        general_results = await self.fetch_general_new_releases(days_back, content_type)
        
        # Check general results for Telugu content
        telugu_from_general = []
        other_from_general = []
        
        for item in general_results:
            if self._is_telugu_content(item):
                telugu_from_general.append(item)
            else:
                other_from_general.append(item)
        
        # Add detected Telugu content to Telugu results
        results['telugu'].extend(telugu_from_general)
        results['general'] = other_from_general
        
        return results
    
    async def fetch_general_new_releases(
        self,
        days_back: int = 60,
        content_type: str = 'movie'
    ) -> List[Dict[str, Any]]:
        """Fetch general new releases sorted by popularity"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        url = f"{self.base_url}/discover/{content_type}"
        params = {
            'api_key': self.tmdb_api_key,
            'primary_release_date.gte': start_date.strftime('%Y-%m-%d') if content_type == 'movie' else None,
            'primary_release_date.lte': end_date.strftime('%Y-%m-%d') if content_type == 'movie' else None,
            'first_air_date.gte': start_date.strftime('%Y-%m-%d') if content_type == 'tv' else None,
            'first_air_date.lte': end_date.strftime('%Y-%m-%d') if content_type == 'tv' else None,
            'sort_by': 'popularity.desc',
            'page': 1
        }
        
        params = {k: v for k, v in params.items() if v is not None}
        
        try:
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get('results', [])
        except Exception as e:
            logger.error(f"Error fetching general releases: {e}")
            return []
    
    def parse_release(self, item: Dict[str, Any], content_type: str) -> NewRelease:
        """Parse API response into NewRelease object"""
        # Get release date
        if content_type == 'movie':
            release_date_str = item.get('release_date', '')
        else:
            release_date_str = item.get('first_air_date', '')
        
        try:
            release_date = datetime.strptime(release_date_str, '%Y-%m-%d')
        except:
            release_date = datetime.now()
        
        # Detect languages
        orig_lang = item.get('original_language', '')
        is_telugu = self._is_telugu_content(item)
        
        if is_telugu:
            languages = ['te', orig_lang] if orig_lang != 'te' else ['te']
        else:
            languages = [orig_lang] if orig_lang else []
        
        # Get poster path
        poster_path = item.get('poster_path')
        if poster_path and not poster_path.startswith('http'):
            poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        backdrop_path = item.get('backdrop_path')
        if backdrop_path and not backdrop_path.startswith('http'):
            backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        
        # Get genre names
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
        use_cache: bool = True
    ) -> List[NewRelease]:
        """
        Get new releases with Telugu priority, auto-refreshed every 3 hours
        """
        # Generate cache key with 3-hour granularity
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        cache_window = int(current_hour.timestamp() // (self.refresh_interval * 3600))
        cache_key = f"new_releases:{content_type}:{days_back}:{cache_window}"
        
        # Check cache
        if use_cache and self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Cache hit for new releases: {cache_key}")
                try:
                    data = json.loads(cached_data) if isinstance(cached_data, str) else cached_data
                    # Reconstruct NewRelease objects
                    releases = []
                    for item in data:
                        release_date = datetime.fromisoformat(item['release_date'])
                        item['release_date'] = release_date
                        release = NewRelease(**item)
                        releases.append(release)
                    return releases
                except Exception as e:
                    logger.error(f"Cache deserialization error: {e}")
        
        # Fetch fresh data
        logger.info(f"Fetching fresh new releases for {content_type}")
        all_language_results = await self.fetch_all_new_releases(days_back, content_type)
        
        # Parse all results
        all_releases = []
        seen_ids = set()
        
        # Process Telugu first (STRICT PRIORITY)
        for item in all_language_results.get('telugu', []):
            release = self.parse_release(item, content_type)
            if release.id not in seen_ids:
                release.is_telugu = True  # Ensure Telugu flag is set
                all_releases.append(release)
                seen_ids.add(release.id)
        
        # Process other languages in priority order
        language_order = ['english', 'hindi', 'malayalam', 'kannada', 'tamil', 'general']
        
        for lang in language_order:
            for item in all_language_results.get(lang, []):
                release = self.parse_release(item, content_type)
                if release.id not in seen_ids:
                    all_releases.append(release)
                    seen_ids.add(release.id)
        
        # Sort with Telugu FIRST, then by freshness and quality
        telugu_releases = [r for r in all_releases if r.is_telugu]
        other_releases = [r for r in all_releases if not r.is_telugu]
        
        # Sort Telugu by freshness and quality
        telugu_sorted = sorted(
            telugu_releases,
            key=lambda r: (
                -r.freshness_score,  # Freshest first
                -r.combined_score,   # Best quality
                r.days_since_release # Newest first
            )
        )
        
        # Sort others by language priority, freshness, and quality
        other_sorted = sorted(
            other_releases,
            key=lambda r: (
                r.language_priority,  # Language priority
                -r.freshness_score,   # Freshness
                -r.combined_score,    # Quality
                r.days_since_release  # Newest first
            )
        )
        
        # Combine: Telugu ALWAYS first
        final_releases = telugu_sorted + other_sorted
        
        # Cache the results for 3 hours
        if use_cache and self.cache:
            cache_data = [r.to_dict() for r in final_releases]
            self.cache.set(
                cache_key,
                json.dumps(cache_data, default=str),
                timeout=self.refresh_interval * 3600  # 3 hours in seconds
            )
            logger.info(f"Cached new releases: {cache_key}")
        
        return final_releases
    
    async def close(self):
        """Clean up resources"""
        if self.http_client:
            await self.http_client.aclose()