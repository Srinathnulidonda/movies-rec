import asyncio
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from flask import current_app
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CineBrainContent:
    id: int
    slug: str
    title: str
    original_title: Optional[str]
    content_type: str
    genres: List[str]
    languages: List[str]
    release_date: str
    rating: float
    vote_count: int
    popularity: float
    overview: str
    poster_path: Optional[str]
    backdrop_path: Optional[str]
    runtime: Optional[int]
    youtube_trailer_id: Optional[str]
    tmdb_id: Optional[int] = None
    mal_id: Optional[int] = None
    imdb_id: Optional[str] = None

@dataclass
class CineBrainNewReleasesConfig:
    language_priorities: List[str]
    date_range_days: int
    refresh_interval_minutes: int
    cache_file_path: str
    max_items_per_category: int
    api_timeout_seconds: int
    max_retries: int
    
class CineBrainNewReleasesService:
    def __init__(self, app=None, db=None, models=None, services=None):
        self.app = app
        self.db = db
        self.models = models
        self.services = services
        
        self.config = CineBrainNewReleasesConfig(
            language_priorities=['telugu', 'english', 'hindi', 'malayalam', 'kannada', 'tamil', 'other'],
            date_range_days=45,  # Updated to 45 days
            refresh_interval_minutes=35,  # More frequent refresh for Telugu content
            cache_file_path='cache/cinebrain_new_releases.json',
            max_items_per_category=80,  # Increased for better Telugu coverage
            api_timeout_seconds=15,
            max_retries=3
        )
        
        # Telugu-specific configuration
        self.telugu_focus_config = {
            'telugu_quota_multiplier': 3,  # Give Telugu content 3x more quota
            'telugu_regions': ['IN', 'US'],  # Check both Indian and US Telugu releases
            'telugu_keywords': ['tollywood', 'telugu', 'andhra', 'hyderabad', 'telangana'],
            'enhanced_telugu_search': True
        }
        
        self._cache_lock = threading.RLock()
        self._refresh_thread = None
        self._should_stop = threading.Event()
        self._cache_data = None
        self._last_update = None
        
        self.tmdb_api_key = app.config.get('TMDB_API_KEY') if app else None
        self.http_session = self._create_session()
        
        self._ensure_cache_directory()
        self._load_cached_data()
        
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=15,
            pool_maxsize=30
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.headers.update({
            'User-Agent': 'CineBrain/2.0 Telugu Focus (https://cinebrain.vercel.app)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        })
        return session
        
    def _ensure_cache_directory(self):
        cache_dir = Path(self.config.cache_file_path).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        
    def start_background_refresh(self):
        if self._refresh_thread and self._refresh_thread.is_alive():
            return
            
        self._should_stop.clear()
        self._refresh_thread = threading.Thread(
            target=self._background_refresh_worker,
            daemon=True,
            name='CineBrainTeluguNewReleasesRefresh'
        )
        self._refresh_thread.start()
        logger.info("CineBrain Telugu-focused new releases background refresh started")
        
    def stop_background_refresh(self):
        if self._refresh_thread:
            self._should_stop.set()
            self._refresh_thread.join(timeout=10)
            logger.info("CineBrain new releases background refresh stopped")
            
    def _background_refresh_worker(self):
        initial_delay = 45  # Shorter initial delay for quicker Telugu content
        time.sleep(initial_delay)
        
        while not self._should_stop.is_set():
            try:
                logger.info("CineBrain: Starting scheduled Telugu-focused new releases refresh")
                self.refresh_new_releases()
                
                wait_time = self.config.refresh_interval_minutes * 60
                for _ in range(wait_time):
                    if self._should_stop.wait(1):
                        return
                        
            except Exception as e:
                logger.error(f"CineBrain new releases background refresh error: {e}")
                if self._should_stop.wait(180):  # Shorter retry delay
                    return
                    
    def get_new_releases(self, force_refresh: bool = False) -> Dict[str, Any]:
        with self._cache_lock:
            if force_refresh or self._should_refresh():
                logger.info("CineBrain: Triggering Telugu-focused new releases refresh")
                self.refresh_new_releases()
                
            return self._cache_data or self._get_empty_response()
            
    def _should_refresh(self) -> bool:
        if not self._cache_data or not self._last_update:
            return True
            
        age_minutes = (datetime.now() - self._last_update).total_seconds() / 60
        return age_minutes >= self.config.refresh_interval_minutes
        
    def refresh_new_releases(self):
        try:
            logger.info("CineBrain: Starting comprehensive Telugu-focused new releases data collection")
            
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=self.config.date_range_days)
            
            logger.info(f"CineBrain: Fetching releases from {start_date} to {end_date} (45 days)")
            
            all_content = []
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                
                # Enhanced Telugu content fetching
                futures.append(executor.submit(
                    self._fetch_enhanced_telugu_releases, start_date, end_date
                ))
                
                # Standard TMDB releases
                futures.append(executor.submit(
                    self._fetch_tmdb_releases, 'movie', start_date, end_date
                ))
                futures.append(executor.submit(
                    self._fetch_tmdb_releases, 'tv', start_date, end_date
                ))
                
                # Regional language specific releases
                for language_code in ['te', 'hi', 'ta', 'kn', 'ml']:
                    futures.append(executor.submit(
                        self._fetch_enhanced_language_specific_releases, language_code, start_date, end_date
                    ))
                
                # Anime releases
                futures.append(executor.submit(
                    self._fetch_anime_releases, start_date, end_date
                ))
                
                # Regional-specific searches
                for region in ['IN', 'US']:
                    futures.append(executor.submit(
                        self._fetch_regional_releases, region, start_date, end_date
                    ))
                
                for future in as_completed(futures, timeout=300):
                    try:
                        content_batch = future.result()
                        if content_batch:
                            all_content.extend(content_batch)
                            logger.info(f"CineBrain: Collected {len(content_batch)} items from batch")
                    except Exception as e:
                        logger.error(f"CineBrain: Content batch fetch error: {e}")
                        continue
                        
            if not all_content:
                logger.warning("CineBrain: No new releases found, keeping existing cache")
                return
                
            unique_content = self._deduplicate_content(all_content)
            logger.info(f"CineBrain: Collected {len(unique_content)} unique releases after deduplication")
            
            # Enhanced processing with Telugu priority
            processed_data = self._process_and_sort_content_with_telugu_priority(unique_content)
            
            with self._cache_lock:
                self._cache_data = processed_data
                self._last_update = datetime.now()
                self._save_cache_to_disk()
                
            telugu_count = sum(1 for item in unique_content if 'telugu' in [lang.lower() for lang in item.languages])
            logger.info(f"CineBrain: Successfully refreshed {len(unique_content)} unique new releases ({telugu_count} Telugu)")
            
        except Exception as e:
            logger.error(f"CineBrain: New releases refresh failed: {e}")
            
    def _fetch_enhanced_telugu_releases(self, start_date, end_date) -> List[CineBrainContent]:
        """Enhanced Telugu content fetching with multiple strategies"""
        if not self.tmdb_api_key:
            logger.warning("CineBrain: TMDB API key not available for Telugu search")
            return []
            
        content_list = []
        
        try:
            # Strategy 1: Direct Telugu language search
            for content_type in ['movie', 'tv']:
                for page in range(1, 6):  # More pages for Telugu
                    url = f"https://api.themoviedb.org/3/discover/{content_type}"
                    params = {
                        'api_key': self.tmdb_api_key,
                        'with_original_language': 'te',
                        'primary_release_date.gte': start_date.isoformat(),
                        'primary_release_date.lte': end_date.isoformat(),
                        'sort_by': 'primary_release_date.desc',
                        'page': page,
                        'vote_count.gte': 1,
                        'region': 'IN'
                    }
                    
                    response = self._make_request(url, params)
                    if not response:
                        break
                        
                    results = response.get('results', [])
                    for item in results:
                        content = self._convert_tmdb_to_cinebrain(item, content_type)
                        if content and self._is_telugu_content(content):
                            content_list.append(content)
                            
                    if not results:
                        break
                        
                    time.sleep(0.3)
            
            # Strategy 2: Search with Telugu keywords
            telugu_keywords = ['tollywood', 'telugu cinema', 'andhra pradesh', 'telangana']
            for keyword in telugu_keywords:
                for content_type in ['movie', 'tv']:
                    url = f"https://api.themoviedb.org/3/search/{content_type}"
                    params = {
                        'api_key': self.tmdb_api_key,
                        'query': keyword,
                        'primary_release_date.gte': start_date.isoformat(),
                        'primary_release_date.lte': end_date.isoformat(),
                        'region': 'IN'
                    }
                    
                    response = self._make_request(url, params)
                    if response:
                        results = response.get('results', [])
                        for item in results:
                            content = self._convert_tmdb_to_cinebrain(item, content_type)
                            if content and self._is_telugu_content(content):
                                content_list.append(content)
                    
                    time.sleep(0.5)
            
            # Strategy 3: Production company search for major Telugu studios
            telugu_companies = [
                'Mythri Movie Makers', 'Geetha Arts', 'Haarika & Hassine Creations',
                'Sri Venkateswara Creations', 'UV Creations', 'GA2 Pictures'
            ]
            
            for company in telugu_companies:
                url = f"https://api.themoviedb.org/3/search/company"
                params = {
                    'api_key': self.tmdb_api_key,
                    'query': company
                }
                
                response = self._make_request(url, params)
                if response:
                    companies = response.get('results', [])
                    for comp in companies[:2]:  # Limit to first 2 matches
                        company_id = comp.get('id')
                        if company_id:
                            # Search movies by this company
                            url = f"https://api.themoviedb.org/3/discover/movie"
                            params = {
                                'api_key': self.tmdb_api_key,
                                'with_companies': company_id,
                                'primary_release_date.gte': start_date.isoformat(),
                                'primary_release_date.lte': end_date.isoformat(),
                                'sort_by': 'primary_release_date.desc'
                            }
                            
                            comp_response = self._make_request(url, params)
                            if comp_response:
                                results = comp_response.get('results', [])
                                for item in results:
                                    content = self._convert_tmdb_to_cinebrain(item, 'movie')
                                    if content:
                                        content_list.append(content)
                
                time.sleep(1.0)
                
        except Exception as e:
            logger.error(f"CineBrain enhanced Telugu fetch error: {e}")
            
        logger.info(f"CineBrain: Enhanced Telugu search found {len(content_list)} items")
        return content_list
        
    def _fetch_enhanced_language_specific_releases(self, language_code: str, start_date, end_date) -> List[CineBrainContent]:
        """Enhanced language-specific fetching with better coverage"""
        if not self.tmdb_api_key:
            return []
            
        content_list = []
        
        try:
            # Primary language search
            for content_type in ['movie', 'tv']:
                pages_to_fetch = 6 if language_code == 'te' else 4  # More pages for Telugu
                
                for page in range(1, pages_to_fetch + 1):
                    url = f"https://api.themoviedb.org/3/discover/{content_type}"
                    params = {
                        'api_key': self.tmdb_api_key,
                        'with_original_language': language_code,
                        'primary_release_date.gte': start_date.isoformat(),
                        'primary_release_date.lte': end_date.isoformat(),
                        'sort_by': 'primary_release_date.desc',
                        'page': page,
                        'vote_count.gte': 1,
                        'region': 'IN' if language_code in ['te', 'hi', 'ta', 'kn', 'ml'] else None
                    }
                    
                    response = self._make_request(url, params)
                    if not response:
                        break
                        
                    results = response.get('results', [])
                    for item in results:
                        content = self._convert_tmdb_to_cinebrain(item, content_type)
                        if content:
                            content_list.append(content)
                            
                    if not results:
                        break
                        
                    time.sleep(0.3)
            
            # Secondary search by popularity for this language
            for content_type in ['movie', 'tv']:
                url = f"https://api.themoviedb.org/3/discover/{content_type}"
                params = {
                    'api_key': self.tmdb_api_key,
                    'with_original_language': language_code,
                    'primary_release_date.gte': start_date.isoformat(),
                    'primary_release_date.lte': end_date.isoformat(),
                    'sort_by': 'popularity.desc',
                    'page': 1,
                    'vote_count.gte': 5
                }
                
                response = self._make_request(url, params)
                if response:
                    results = response.get('results', [])
                    for item in results:
                        content = self._convert_tmdb_to_cinebrain(item, content_type)
                        if content:
                            content_list.append(content)
                
                time.sleep(0.4)
                    
        except Exception as e:
            logger.error(f"CineBrain enhanced language-specific {language_code} fetch error: {e}")
            
        return content_list
        
    def _fetch_regional_releases(self, region: str, start_date, end_date) -> List[CineBrainContent]:
        """Fetch releases by region"""
        if not self.tmdb_api_key:
            return []
            
        content_list = []
        
        try:
            for content_type in ['movie', 'tv']:
                for page in range(1, 4):
                    url = f"https://api.themoviedb.org/3/discover/{content_type}"
                    params = {
                        'api_key': self.tmdb_api_key,
                        'region': region,
                        'primary_release_date.gte': start_date.isoformat(),
                        'primary_release_date.lte': end_date.isoformat(),
                        'sort_by': 'primary_release_date.desc',
                        'page': page,
                        'vote_count.gte': 2
                    }
                    
                    response = self._make_request(url, params)
                    if not response:
                        break
                        
                    results = response.get('results', [])
                    for item in results:
                        content = self._convert_tmdb_to_cinebrain(item, content_type)
                        if content:
                            content_list.append(content)
                            
                    if not results:
                        break
                        
                    time.sleep(0.4)
                    
        except Exception as e:
            logger.error(f"CineBrain regional {region} fetch error: {e}")
            
        return content_list
        
    def _is_telugu_content(self, content: CineBrainContent) -> bool:
        """Check if content is Telugu"""
        for lang in content.languages:
            if lang.lower() in ['telugu', 'te']:
                return True
        
        # Check title for Telugu indicators
        title_lower = content.title.lower()
        telugu_indicators = ['telugu', 'tollywood', 'andhra', 'telangana']
        return any(indicator in title_lower for indicator in telugu_indicators)
        
    def _deduplicate_content(self, content_list: List[CineBrainContent]) -> List[CineBrainContent]:
        seen_titles = set()
        seen_ids = set()
        unique_content = []
        
        # Sort to prioritize higher quality content
        content_list.sort(key=lambda x: (-x.popularity, -x.rating, -x.vote_count))
        
        for content in content_list:
            # Create composite key for deduplication
            title_key = f"{content.title.lower().strip()}_{content.content_type}"
            id_key = f"{content.tmdb_id or content.mal_id or 0}_{content.content_type}"
            
            # More flexible deduplication for Telugu content
            is_telugu = self._is_telugu_content(content)
            
            if title_key not in seen_titles and id_key not in seen_ids:
                seen_titles.add(title_key)
                seen_ids.add(id_key)
                unique_content.append(content)
            elif is_telugu and len(unique_content) < 100:  # Allow some duplicates for Telugu
                unique_content.append(content)
                
        return unique_content
        
    def _fetch_tmdb_releases(self, content_type: str, start_date, end_date) -> List[CineBrainContent]:
        if not self.tmdb_api_key:
            logger.warning("CineBrain: TMDB API key not available")
            return []
            
        content_list = []
        
        try:
            for page in range(1, 10):  # More pages for comprehensive coverage
                url = f"https://api.themoviedb.org/3/discover/{content_type}"
                params = {
                    'api_key': self.tmdb_api_key,
                    'primary_release_date.gte': start_date.isoformat(),
                    'primary_release_date.lte': end_date.isoformat(),
                    'sort_by': 'primary_release_date.desc',
                    'page': page,
                    'vote_count.gte': 2,
                    'include_adult': False
                }
                
                response = self._make_request(url, params)
                if not response:
                    break
                    
                results = response.get('results', [])
                if not results:
                    break
                    
                for item in results:
                    content = self._convert_tmdb_to_cinebrain(item, content_type)
                    if content:
                        content_list.append(content)
                        
                if len(content_list) >= self.config.max_items_per_category:
                    break
                    
                time.sleep(0.2)
                
        except Exception as e:
            logger.error(f"CineBrain TMDB {content_type} fetch error: {e}")
            
        return content_list[:self.config.max_items_per_category]
        
    def _fetch_anime_releases(self, start_date, end_date) -> List[CineBrainContent]:
        content_list = []
        
        try:
            for page in range(1, 6):
                url = "https://api.jikan.moe/v4/anime"
                params = {
                    'status': 'airing',
                    'order_by': 'start_date',
                    'sort': 'desc',
                    'page': page,
                    'limit': 25,
                    'min_score': 5.5
                }
                
                response = self._make_request(url, params)
                if not response:
                    break
                    
                results = response.get('data', [])
                if not results:
                    break
                    
                for item in results:
                    content = self._convert_anime_to_cinebrain(item)
                    if content and self._is_within_date_range(content.release_date, start_date, end_date):
                        content_list.append(content)
                        
                if len(content_list) >= self.config.max_items_per_category:
                    break
                    
                time.sleep(1.2)
                
        except Exception as e:
            logger.error(f"CineBrain anime fetch error: {e}")
            
        return content_list[:self.config.max_items_per_category]
        
    def _make_request(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for attempt in range(self.config.max_retries):
            try:
                response = self.http_session.get(
                    url, 
                    params=params, 
                    timeout=self.config.api_timeout_seconds
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = min(2 ** attempt * 4, 45)
                    logger.warning(f"CineBrain: Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                elif response.status_code in [500, 502, 503, 504]:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    logger.warning(f"CineBrain API request failed: {response.status_code}")
                    break
                    
            except Exception as e:
                logger.error(f"CineBrain API request error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
                    
        return None
        
    def _convert_tmdb_to_cinebrain(self, item: Dict[str, Any], content_type: str) -> Optional[CineBrainContent]:
        try:
            title = item.get('title') or item.get('name', '')
            if not title or len(title.strip()) == 0:
                return None
                
            release_date = item.get('release_date') or item.get('first_air_date', '')
            if not release_date:
                return None
                
            slug = self._generate_slug(title, release_date)
            poster_path = item.get('poster_path')
            if poster_path and not poster_path.startswith('http'):
                poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
                
            backdrop_path = item.get('backdrop_path')
            if backdrop_path and not backdrop_path.startswith('http'):
                backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
            
            languages = self._extract_languages(item.get('original_language', 'en'))
            
            return CineBrainContent(
                id=item.get('id', 0),
                slug=slug,
                title=title.strip(),
                original_title=item.get('original_title') or item.get('original_name'),
                content_type=content_type,
                genres=self._extract_genres(item.get('genre_ids', [])),
                languages=languages,
                release_date=release_date,
                rating=float(item.get('vote_average', 0)),
                vote_count=item.get('vote_count', 0),
                popularity=float(item.get('popularity', 0)),
                overview=item.get('overview', '').strip(),
                poster_path=poster_path,
                backdrop_path=backdrop_path,
                runtime=None,
                youtube_trailer_id=None,
                tmdb_id=item.get('id')
            )
            
        except Exception as e:
            logger.error(f"CineBrain TMDB conversion error: {e}")
            return None
            
    def _convert_anime_to_cinebrain(self, item: Dict[str, Any]) -> Optional[CineBrainContent]:
        try:
            title = item.get('title', '')
            if not title or len(title.strip()) == 0:
                return None
                
            aired = item.get('aired', {})
            release_date = ''
            if aired and aired.get('from'):
                release_date = aired['from'][:10]
                
            if not release_date:
                return None
                
            slug = self._generate_slug(title, release_date)
            
            return CineBrainContent(
                id=item.get('mal_id', 0),
                slug=slug,
                title=title.strip(),
                original_title=item.get('title_japanese'),
                content_type='anime',
                genres=[genre.get('name', '') for genre in item.get('genres', [])],
                languages=['japanese'],
                release_date=release_date,
                rating=float(item.get('score', 0) or 0),
                vote_count=item.get('scored_by', 0),
                popularity=float(item.get('popularity', 0) or 0),
                overview=item.get('synopsis', '').strip() if item.get('synopsis') else '',
                poster_path=item.get('images', {}).get('jpg', {}).get('image_url'),
                backdrop_path=item.get('images', {}).get('jpg', {}).get('large_image_url'),
                runtime=None,
                youtube_trailer_id=None,
                mal_id=item.get('mal_id')
            )
            
        except Exception as e:
            logger.error(f"CineBrain anime conversion error: {e}")
            return None
            
    def _extract_genres(self, genre_ids: List[int]) -> List[str]:
        genre_map = {
            28: 'Action', 35: 'Comedy', 18: 'Drama', 27: 'Horror',
            10749: 'Romance', 878: 'Science Fiction', 53: 'Thriller',
            9648: 'Mystery', 14: 'Fantasy', 12: 'Adventure',
            16: 'Animation', 80: 'Crime', 99: 'Documentary',
            10751: 'Family', 36: 'History', 10402: 'Music',
            10770: 'TV Movie', 37: 'Western', 10752: 'War'
        }
        return [genre_map.get(gid, f'Genre_{gid}') for gid in genre_ids if gid in genre_map]
        
    def _extract_languages(self, original_language: str) -> List[str]:
        language_map = {
            'te': 'telugu', 'hi': 'hindi', 'ta': 'tamil', 
            'kn': 'kannada', 'ml': 'malayalam', 'en': 'english',
            'ja': 'japanese', 'ko': 'korean', 'es': 'spanish',
            'fr': 'french', 'de': 'german', 'it': 'italian',
            'zh': 'chinese', 'ru': 'russian', 'pt': 'portuguese'
        }
        lang = language_map.get(original_language, original_language)
        return [lang] if lang else ['english']
        
    def _generate_slug(self, title: str, release_date: str) -> str:
        try:
            year = datetime.fromisoformat(release_date).year
            slug = title.lower()
            slug = ''.join(c if c.isalnum() or c == ' ' else '' for c in slug)
            slug = '-'.join(slug.split())
            slug = f"{slug}-{year}"
            return slug[:100] if len(slug) > 100 else slug
        except:
            slug = title.lower()
            slug = ''.join(c if c.isalnum() or c == ' ' else '' for c in slug)
            slug = '-'.join(slug.split())
            return slug[:50] if len(slug) > 50 else slug
            
    def _is_within_date_range(self, release_date: str, start_date, end_date) -> bool:
        try:
            release = datetime.fromisoformat(release_date).date()
            return start_date <= release <= end_date
        except:
            return False
            
    def _process_and_sort_content_with_telugu_priority(self, content_list: List[CineBrainContent]) -> Dict[str, Any]:
        """Enhanced processing with Telugu priority and 45-day focus"""
        all_content = {
            'movies': [],
            'tv_shows': [],
            'anime': []
        }
        
        for content in content_list:
            content_dict = asdict(content)
            
            if content.content_type == 'movie':
                all_content['movies'].append(content_dict)
            elif content.content_type == 'tv':
                all_content['tv_shows'].append(content_dict)
            elif content.content_type == 'anime':
                all_content['anime'].append(content_dict)
                
        # Sort each category with Telugu priority and recency
        for category in all_content:
            all_content[category] = self._sort_by_telugu_priority_and_date(all_content[category])
            
        priority_content = self._create_telugu_priority_content(all_content)
        
        return {
            'priority_content': priority_content,
            'all_content': all_content,
            'metadata': {
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'total_items': sum(len(items) for items in all_content.values()),
                'language_priorities': self.config.language_priorities,
                'date_range_days': self.config.date_range_days,
                'refresh_interval_minutes': self.config.refresh_interval_minutes,
                'telugu_focus': True,
                'cache_version': '2.1',
                'cinebrain_service': 'new_releases'
            }
        }
        
    def _sort_by_telugu_priority_and_date(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort by Telugu priority first, then by release date (newest first)"""
        def sort_key(item):
            try:
                # Calculate days since release (lower = more recent)
                release_date = datetime.fromisoformat(item['release_date'])
                days_ago = (datetime.now().date() - release_date.date()).days
                
                # Language priority (lower = higher priority)
                language_priority = 999
                for lang in item.get('languages', []):
                    lang_lower = lang.lower()
                    if lang_lower in self.config.language_priorities:
                        priority_index = self.config.language_priorities.index(lang_lower)
                        language_priority = min(language_priority, priority_index)
                    elif lang_lower not in ['telugu', 'english', 'hindi', 'malayalam', 'kannada', 'tamil']:
                        # Assign to 'other' category
                        other_index = self.config.language_priorities.index('other') if 'other' in self.config.language_priorities else 6
                        language_priority = min(language_priority, other_index)
                
                # Quality scores (higher = better)
                popularity_score = item.get('popularity', 0)
                rating_score = item.get('rating', 0) * max(1, item.get('vote_count', 1))
                
                # Telugu gets extra boost
                telugu_boost = -100 if any('telugu' in lang.lower() for lang in item.get('languages', [])) else 0
                
                return (
                    language_priority + telugu_boost,  # Primary: Language priority with Telugu boost
                    days_ago,                          # Secondary: Recency (newest first)
                    -popularity_score,                 # Tertiary: Popularity (descending)
                    -rating_score                      # Quaternary: Quality (descending)
                )
            except:
                return (999, 999, 0, 0)
                
        return sorted(content_list, key=sort_key)
        
    def _create_telugu_priority_content(self, all_content: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Create priority content with strong Telugu preference"""
        priority_items = []
        
        # Enhanced quotas with Telugu focus
        language_quotas = {
            'telugu': 15,     # Significantly increased Telugu quota
            'english': 8,     # Reduced other quotas to prioritize Telugu
            'hindi': 6,
            'malayalam': 4,
            'kannada': 4,
            'tamil': 4,
            'other': 3
        }
        
        for category in ['movies', 'tv_shows', 'anime']:
            category_items = all_content.get(category, [])
            
            # Separate by language
            language_buckets = {lang: [] for lang in self.config.language_priorities}
            
            for item in category_items:
                categorized = False
                for lang in item.get('languages', []):
                    lang_lower = lang.lower()
                    if lang_lower in language_buckets:
                        language_buckets[lang_lower].append(item)
                        categorized = True
                        break
                        
                if not categorized:
                    language_buckets['other'].append(item)
                    
            # Add items according to quotas
            for lang in self.config.language_priorities:
                quota = language_quotas.get(lang, 2)
                
                # Special handling for anime
                if category == 'anime':
                    if lang == 'japanese':
                        quota = 8
                    elif lang not in ['japanese', 'english']:
                        quota = 1
                
                priority_items.extend(language_buckets[lang][:quota])
                
        # Final sort to ensure Telugu content appears first
        final_priority = self._sort_by_telugu_priority_and_date(priority_items)
        
        return final_priority[:50]  # Return top 50 items
        
    def _save_cache_to_disk(self):
        try:
            temp_file = f"{self.config.cache_file_path}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache_data, f, ensure_ascii=False, indent=2)
            
            os.replace(temp_file, self.config.cache_file_path)
            logger.info("CineBrain: Telugu-focused cache saved successfully")
            
        except Exception as e:
            logger.error(f"CineBrain cache save error: {e}")
            
    def _load_cached_data(self):
        try:
            if os.path.exists(self.config.cache_file_path):
                with open(self.config.cache_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if self._validate_cache_structure(data):
                    self._cache_data = data
                    self._last_update = datetime.now()
                    logger.info("CineBrain: Loaded cached Telugu-focused new releases data successfully")
                else:
                    logger.warning("CineBrain: Invalid cache structure, will refresh")
                    
        except Exception as e:
            logger.error(f"CineBrain cache load error: {e}")
            
    def _validate_cache_structure(self, data: Dict[str, Any]) -> bool:
        required_keys = ['priority_content', 'all_content', 'metadata']
        if not all(key in data for key in required_keys):
            return False
            
        all_content = data.get('all_content', {})
        required_categories = ['movies', 'tv_shows', 'anime']
        if not all(cat in all_content for cat in required_categories):
            return False
            
        return True
            
    def _get_empty_response(self) -> Dict[str, Any]:
        return {
            'priority_content': [],
            'all_content': {
                'movies': [],
                'tv_shows': [],
                'anime': []
            },
            'metadata': {
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'total_items': 0,
                'language_priorities': self.config.language_priorities,
                'date_range_days': self.config.date_range_days,
                'refresh_interval_minutes': self.config.refresh_interval_minutes,
                'telugu_focus': True,
                'cache_version': '2.1',
                'error': 'No data available',
                'cinebrain_service': 'new_releases'
            }
        }
        
    def update_config(self, **kwargs):
        with self._cache_lock:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.info(f"CineBrain: Updated config {key} = {value}")
                
    def get_stats(self) -> Dict[str, Any]:
        with self._cache_lock:
            if not self._cache_data:
                return {'status': 'no_data', 'cinebrain_service': 'new_releases'}
                
            metadata = self._cache_data.get('metadata', {})
            
            # Calculate Telugu content stats
            telugu_stats = self._calculate_telugu_stats()
            
            return {
                'status': 'active',
                'cinebrain_service': 'new_releases',
                'telugu_focus': True,
                'last_update': self._last_update.isoformat() if self._last_update else None,
                'total_items': metadata.get('total_items', 0),
                'priority_items': len(self._cache_data.get('priority_content', [])),
                'movies': len(self._cache_data.get('all_content', {}).get('movies', [])),
                'tv_shows': len(self._cache_data.get('all_content', {}).get('tv_shows', [])),
                'anime': len(self._cache_data.get('all_content', {}).get('anime', [])),
                'telugu_content': telugu_stats,
                'background_refresh': self._refresh_thread.is_alive() if self._refresh_thread else False,
                'cache_age_minutes': (datetime.now() - self._last_update).total_seconds() / 60 if self._last_update else None,
                'language_priorities': self.config.language_priorities,
                'refresh_interval': self.config.refresh_interval_minutes,
                'date_range_days': self.config.date_range_days
            }
            
    def _calculate_telugu_stats(self) -> Dict[str, int]:
        """Calculate Telugu content statistics"""
        if not self._cache_data:
            return {'total': 0, 'priority': 0, 'movies': 0, 'tv_shows': 0}
            
        telugu_stats = {'total': 0, 'priority': 0, 'movies': 0, 'tv_shows': 0}
        
        # Count in priority content
        priority_content = self._cache_data.get('priority_content', [])
        for item in priority_content:
            if any('telugu' in lang.lower() for lang in item.get('languages', [])):
                telugu_stats['priority'] += 1
                
        # Count in all content
        all_content = self._cache_data.get('all_content', {})
        for category, items in all_content.items():
            for item in items:
                if any('telugu' in lang.lower() for lang in item.get('languages', [])):
                    telugu_stats['total'] += 1
                    if category == 'movies':
                        telugu_stats['movies'] += 1
                    elif category == 'tv_shows':
                        telugu_stats['tv_shows'] += 1
                        
        return telugu_stats

def init_cinebrain_new_releases_service(app, db, models, services):
    try:
        service = CineBrainNewReleasesService(app, db, models, services)
        service.start_background_refresh()
        
        @app.teardown_appcontext
        def cleanup_cinebrain_service(error):
            if hasattr(app, '_cinebrain_new_releases_service'):
                app._cinebrain_new_releases_service.stop_background_refresh()
                
        app._cinebrain_new_releases_service = service
        logger.info("CineBrain Telugu-focused new releases service initialized successfully")
        return service
        
    except Exception as e:
        logger.error(f"CineBrain new releases service initialization error: {e}")
        return None