#backend/services/new_releases.py
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
            date_range_days=45,
            refresh_interval_minutes=35,
            cache_file_path='cache/cinebrain_new_releases.json',
            max_items_per_category=80,
            api_timeout_seconds=15,
            max_retries=3
        )
        
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
            'User-Agent': 'CineBrain/2.0 45-Day Releases (https://cinebrain.vercel.app)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        })
        return session
        
    def _ensure_cache_directory(self):
        cache_dir = Path(self.config.cache_file_path).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_strict_date_range(self) -> Tuple[datetime.date, datetime.date]:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.config.date_range_days)
        return start_date, end_date
        
    def _is_within_strict_date_range(self, release_date: str) -> bool:
        try:
            if not release_date:
                return False
                
            release = datetime.fromisoformat(release_date.replace('Z', '+00:00')).date()
            start_date, end_date = self._get_strict_date_range()
            
            is_valid = start_date <= release <= end_date
            if not is_valid:
                days_ago = (end_date - release).days
                logger.debug(f"CineBrain: Filtered out content from {release_date} ({days_ago} days ago)")
            
            return is_valid
        except Exception as e:
            logger.warning(f"CineBrain: Invalid date format {release_date}: {e}")
            return False
            
    def start_background_refresh(self):
        if self._refresh_thread and self._refresh_thread.is_alive():
            return
            
        self._should_stop.clear()
        self._refresh_thread = threading.Thread(
            target=self._background_refresh_worker,
            daemon=True,
            name='CineBrain45DayNewReleases'
        )
        self._refresh_thread.start()
        logger.info("CineBrain 45-day new releases background refresh started")
        
    def stop_background_refresh(self):
        if self._refresh_thread:
            self._should_stop.set()
            self._refresh_thread.join(timeout=10)
            logger.info("CineBrain new releases background refresh stopped")
            
    def _background_refresh_worker(self):
        initial_delay = 45
        time.sleep(initial_delay)
        
        while not self._should_stop.is_set():
            try:
                logger.info("CineBrain: Starting scheduled 45-day new releases refresh")
                self.refresh_new_releases()
                
                wait_time = self.config.refresh_interval_minutes * 60
                for _ in range(wait_time):
                    if self._should_stop.wait(1):
                        return
                        
            except Exception as e:
                logger.error(f"CineBrain new releases background refresh error: {e}")
                if self._should_stop.wait(180):
                    return
                    
    def get_new_releases(self, force_refresh: bool = False) -> Dict[str, Any]:
        with self._cache_lock:
            if force_refresh or self._should_refresh():
                logger.info("CineBrain: Triggering 45-day new releases refresh")
                self.refresh_new_releases()
                
            return self._cache_data or self._get_empty_response()
            
    def _should_refresh(self) -> bool:
        if not self._cache_data or not self._last_update:
            return True
            
        age_minutes = (datetime.now() - self._last_update).total_seconds() / 60
        return age_minutes >= self.config.refresh_interval_minutes
        
    def refresh_new_releases(self):
        try:
            start_date, end_date = self._get_strict_date_range()
            logger.info(f"CineBrain: STRICT date range - {start_date} to {end_date} (last 45 days ONLY)")
            logger.info(f"CineBrain: Fetching ONLY releases from {start_date} to {end_date} (last 45 days)")
            logger.info(f"CineBrain: Using language priority order: {self.config.language_priorities}")
            
            all_content = []
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                
                futures.append(executor.submit(
                    self._fetch_enhanced_telugu_releases, start_date, end_date
                ))
                
                futures.append(executor.submit(
                    self._fetch_tmdb_releases, 'movie', start_date, end_date
                ))
                futures.append(executor.submit(
                    self._fetch_tmdb_releases, 'tv', start_date, end_date
                ))
                
                for language_code in ['te', 'hi', 'ta', 'kn', 'ml']:
                    futures.append(executor.submit(
                        self._fetch_enhanced_language_specific_releases, language_code, start_date, end_date
                    ))
                
                futures.append(executor.submit(
                    self._fetch_anime_releases, start_date, end_date
                ))
                
                for region in ['IN', 'US']:
                    futures.append(executor.submit(
                        self._fetch_regional_releases, region, start_date, end_date
                    ))
                
                for future in as_completed(futures, timeout=300):
                    try:
                        content_batch = future.result()
                        if content_batch:
                            valid_content = [
                                content for content in content_batch 
                                if self._is_within_strict_date_range(content.release_date)
                            ]
                            all_content.extend(valid_content)
                            filtered_count = len(content_batch) - len(valid_content)
                            if filtered_count > 0:
                                logger.info(f"CineBrain: Filtered out {filtered_count} items older than 45 days")
                            logger.info(f"CineBrain: Collected {len(valid_content)} valid items from batch")
                    except Exception as e:
                        logger.error(f"CineBrain: Content batch fetch error: {e}")
                        continue
                        
            if not all_content:
                logger.warning("CineBrain: No new releases found within last 45 days, keeping existing cache")
                return
                
            strictly_filtered_content = [
                content for content in all_content 
                if self._is_within_strict_date_range(content.release_date)
            ]
            
            if len(strictly_filtered_content) != len(all_content):
                filtered_out = len(all_content) - len(strictly_filtered_content)
                logger.info(f"CineBrain: Final filter removed {filtered_out} items older than 45 days")
            
            unique_content = self._deduplicate_content(strictly_filtered_content)
            logger.info(f"CineBrain: Processing {len(unique_content)} unique releases from last 45 days")
            
            processed_data = self._process_and_sort_content_with_language_priority(unique_content)
            
            with self._cache_lock:
                self._cache_data = processed_data
                self._last_update = datetime.now()
                self._save_cache_to_disk()
                
            telugu_count = sum(1 for item in unique_content if 'telugu' in [lang.lower() for lang in item.languages])
            logger.info(f"CineBrain: Successfully refreshed {len(unique_content)} releases from last 45 days ({telugu_count} Telugu)")
            
        except Exception as e:
            logger.error(f"CineBrain: New releases refresh failed: {e}")
            
    def _fetch_enhanced_telugu_releases(self, start_date, end_date) -> List[CineBrainContent]:
        if not self.tmdb_api_key:
            logger.warning("CineBrain: TMDB API key not available for Telugu search")
            return []
            
        content_list = []
        
        try:
            for content_type in ['movie', 'tv']:
                for page in range(1, 6):
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
                        release_date = item.get('release_date') or item.get('first_air_date', '')
                        if not self._is_within_strict_date_range(release_date):
                            continue
                            
                        content = self._convert_tmdb_to_cinebrain(item, content_type)
                        if content and self._is_telugu_content(content):
                            content_list.append(content)
                            
                    if not results:
                        break
                        
                    time.sleep(0.3)
            
            telugu_keywords = ['tollywood', 'telugu cinema']
            for keyword in telugu_keywords:
                for content_type in ['movie', 'tv']:
                    url = f"https://api.themoviedb.org/3/search/{content_type}"
                    params = {
                        'api_key': self.tmdb_api_key,
                        'query': keyword,
                        'region': 'IN'
                    }
                    
                    response = self._make_request(url, params)
                    if response:
                        results = response.get('results', [])
                        for item in results:
                            release_date = item.get('release_date') or item.get('first_air_date', '')
                            if not self._is_within_strict_date_range(release_date):
                                continue
                                
                            content = self._convert_tmdb_to_cinebrain(item, content_type)
                            if content and self._is_telugu_content(content):
                                content_list.append(content)
                    
                    time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"CineBrain enhanced Telugu fetch error: {e}")
            
        logger.info(f"CineBrain: Enhanced Telugu search found {len(content_list)} items within 45 days")
        return content_list
        
    def _fetch_enhanced_language_specific_releases(self, language_code: str, start_date, end_date) -> List[CineBrainContent]:
        if not self.tmdb_api_key:
            return []
            
        content_list = []
        
        try:
            for content_type in ['movie', 'tv']:
                pages_to_fetch = 6 if language_code == 'te' else 4
                
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
                        release_date = item.get('release_date') or item.get('first_air_date', '')
                        if not self._is_within_strict_date_range(release_date):
                            continue
                            
                        content = self._convert_tmdb_to_cinebrain(item, content_type)
                        if content:
                            content_list.append(content)
                            
                    if not results:
                        break
                        
                    time.sleep(0.3)
                    
        except Exception as e:
            logger.error(f"CineBrain enhanced language-specific {language_code} fetch error: {e}")
            
        return content_list
        
    def _fetch_regional_releases(self, region: str, start_date, end_date) -> List[CineBrainContent]:
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
                        release_date = item.get('release_date') or item.get('first_air_date', '')
                        if not self._is_within_strict_date_range(release_date):
                            continue
                            
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
        for lang in content.languages:
            if lang.lower() in ['telugu', 'te']:
                return True
        
        title_lower = content.title.lower()
        telugu_indicators = ['telugu', 'tollywood', 'andhra', 'telangana']
        return any(indicator in title_lower for indicator in telugu_indicators)
        
    def _deduplicate_content(self, content_list: List[CineBrainContent]) -> List[CineBrainContent]:
        seen_titles = set()
        seen_ids = set()
        unique_content = []
        
        content_list.sort(key=lambda x: (-x.popularity, -x.rating, -x.vote_count))
        
        for content in content_list:
            if not self._is_within_strict_date_range(content.release_date):
                continue
                
            title_key = f"{content.title.lower().strip()}_{content.content_type}"
            id_key = f"{content.tmdb_id or content.mal_id or 0}_{content.content_type}"
            
            is_telugu = self._is_telugu_content(content)
            
            if title_key not in seen_titles and id_key not in seen_ids:
                seen_titles.add(title_key)
                seen_ids.add(id_key)
                unique_content.append(content)
            elif is_telugu and len(unique_content) < 100:
                unique_content.append(content)
                
        return unique_content
        
    def _fetch_tmdb_releases(self, content_type: str, start_date, end_date) -> List[CineBrainContent]:
        if not self.tmdb_api_key:
            logger.warning("CineBrain: TMDB API key not available")
            return []
            
        content_list = []
        
        try:
            for page in range(1, 10):
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
                    release_date = item.get('release_date') or item.get('first_air_date', '')
                    if not self._is_within_strict_date_range(release_date):
                        continue
                        
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
                    if content and self._is_within_strict_date_range(content.release_date):
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
                
            if not self._is_within_strict_date_range(release_date):
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
                
            if not self._is_within_strict_date_range(release_date):
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
            year = datetime.fromisoformat(release_date.replace('Z', '+00:00')).year
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
            
    def _process_and_sort_content_with_language_priority(self, content_list: List[CineBrainContent]) -> Dict[str, Any]:
        all_content = {
            'movies': [],
            'tv_shows': [],
            'anime': []
        }
        
        valid_content_list = [
            content for content in content_list 
            if self._is_within_strict_date_range(content.release_date)
        ]
        
        filtered_out = len(content_list) - len(valid_content_list)
        if filtered_out > 0:
            logger.info(f"CineBrain: Final processing filtered out {filtered_out} items older than 45 days")
        
        for content in valid_content_list:
            content_dict = asdict(content)
            
            if content.content_type == 'movie':
                all_content['movies'].append(content_dict)
            elif content.content_type == 'tv':
                all_content['tv_shows'].append(content_dict)
            elif content.content_type == 'anime':
                all_content['anime'].append(content_dict)
                
        for category in all_content:
            all_content[category] = self._sort_by_language_priority_and_date(all_content[category])
            logger.info(f"CineBrain: Sorted {len(all_content[category])} {category} by language priority and release date")
            
        priority_content = self._create_language_priority_content(all_content)
        
        valid_priority_content = [
            item for item in priority_content 
            if self._is_within_strict_date_range(item.get('release_date', ''))
        ]
        
        logger.info(f"CineBrain: Created priority content with {len(valid_priority_content)} items sorted by language priority")
        
        return {
            'priority_content': valid_priority_content,
            'all_content': all_content,
            'metadata': {
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'total_items': sum(len(items) for items in all_content.values()),
                'language_priorities': self.config.language_priorities,
                'sorting_strategy': 'language_priority_then_release_date_newest_first',
                'date_range_days': self.config.date_range_days,
                'strict_45_day_filter': True,
                'date_range_start': self._get_strict_date_range()[0].isoformat(),
                'date_range_end': self._get_strict_date_range()[1].isoformat(),
                'refresh_interval_minutes': self.config.refresh_interval_minutes,
                'telugu_focus': True,
                'cache_version': '2.3',
                'cinebrain_service': 'new_releases'
            }
        }
        
    def _sort_by_language_priority_and_date(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def sort_key(item):
            try:
                release_date_str = item.get('release_date', '')
                if not self._is_within_strict_date_range(release_date_str):
                    return (9999, 9999, 0, 0)
                
                release_date = datetime.fromisoformat(release_date_str)
                days_ago = (datetime.now().date() - release_date.date()).days
                
                language_priority = 999
                
                item_languages = item.get('languages', [])
                for lang in item_languages:
                    lang_lower = lang.lower()
                    
                    if lang_lower in self.config.language_priorities:
                        priority_index = self.config.language_priorities.index(lang_lower)
                        language_priority = min(language_priority, priority_index)
                        break
                    else:
                        if 'other' in self.config.language_priorities:
                            other_index = self.config.language_priorities.index('other')
                            language_priority = min(language_priority, other_index)
                
                popularity_score = item.get('popularity', 0)
                rating_score = item.get('rating', 0) * max(1, item.get('vote_count', 1))
                
                if any('telugu' in lang.lower() for lang in item_languages):
                    logger.debug(f"CineBrain Telugu: '{item.get('title', 'Unknown')}' - "
                               f"Language Priority: {language_priority}, Days Ago: {days_ago}, "
                               f"Popularity: {popularity_score}")
                
                return (
                    language_priority,
                    days_ago,
                    -popularity_score,
                    -rating_score
                )
                
            except Exception as e:
                logger.warning(f"CineBrain: Sort key error for item {item.get('title', 'Unknown')}: {e}")
                return (9999, 9999, 0, 0)
                
        sorted_content = sorted(content_list, key=sort_key)
        
        if sorted_content:
            logger.info(f"CineBrain: Sorted content example - Top 3 items:")
            for i, item in enumerate(sorted_content[:3]):
                languages = item.get('languages', [])
                days_ago = (datetime.now().date() - datetime.fromisoformat(item.get('release_date', '2000-01-01')).date()).days
                logger.info(f"  {i+1}. '{item.get('title', 'Unknown')}' - "
                           f"Languages: {languages}, Days ago: {days_ago}")
                    
        return sorted_content
        
    def _create_language_priority_content(self, all_content: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        priority_items = []
        
        language_quotas = {
            'telugu': 15,
            'english': 8,     
            'hindi': 6,
            'malayalam': 4,
            'kannada': 4,
            'tamil': 4,
            'other': 3
        }
        
        logger.info(f"CineBrain: Creating priority content with language quotas: {language_quotas}")
        
        for category in ['movies', 'tv_shows', 'anime']:
            category_items = all_content.get(category, [])
            logger.info(f"CineBrain: Processing {len(category_items)} {category}")
            
            language_buckets = {lang: [] for lang in self.config.language_priorities}
            
            for item in category_items:
                if not self._is_within_strict_date_range(item.get('release_date', '')):
                    continue
                    
                item_languages = item.get('languages', [])
                categorized = False
                
                for priority_lang in self.config.language_priorities:
                    if priority_lang == 'other':
                        continue
                        
                    for lang in item_languages:
                        if lang.lower() == priority_lang:
                            language_buckets[priority_lang].append(item)
                            categorized = True
                            break
                            
                    if categorized:
                        break
                        
                if not categorized and 'other' in language_buckets:
                    language_buckets['other'].append(item)
                    
            for lang in self.config.language_priorities:
                bucket_items = language_buckets.get(lang, [])
                quota = language_quotas.get(lang, 2)
                
                if category == 'anime':
                    if lang == 'japanese':
                        quota = 8
                    elif lang not in ['japanese', 'english']:
                        quota = 1
                
                selected_items = bucket_items[:quota]
                priority_items.extend(selected_items)
                
                if selected_items:
                    logger.info(f"CineBrain: Added {len(selected_items)} {lang} {category} "
                               f"(quota: {quota}, available: {len(bucket_items)})")
                    
        final_priority = self._sort_by_language_priority_and_date(priority_items)
        
        lang_summary = {}
        for item in final_priority[:20]:
            for lang in item.get('languages', []):
                lang_lower = lang.lower()
                if lang_lower in lang_summary:
                    lang_summary[lang_lower] += 1
                else:
                    lang_summary[lang_lower] = 1
                    
        logger.info(f"CineBrain: Final priority content language distribution (top 20): {lang_summary}")
        
        return final_priority[:50]

    def _get_language_priority_index(self, language: str) -> int:
        lang_lower = language.lower()
        
        if lang_lower in self.config.language_priorities:
            return self.config.language_priorities.index(lang_lower)
        else:
            if 'other' in self.config.language_priorities:
                return self.config.language_priorities.index('other')
            else:
                return 999

    def update_language_priorities(self, new_priorities: List[str]):
        with self._cache_lock:
            old_priorities = self.config.language_priorities.copy()
            self.config.language_priorities = new_priorities
            
            logger.info(f"CineBrain: Updated language priorities from {old_priorities} to {new_priorities}")
            
            if self._cache_data:
                logger.info("CineBrain: Triggering refresh due to language priority change")
                threading.Thread(target=self.refresh_new_releases, daemon=True).start()

    def get_language_priority_stats(self) -> Dict[str, Any]:
        if not self._cache_data:
            return {}
            
        priority_content = self._cache_data.get('priority_content', [])
        all_content = self._cache_data.get('all_content', {})
        
        stats = {
            'language_priorities': self.config.language_priorities,
            'priority_content_distribution': {},
            'all_content_distribution': {},
            'total_priority_items': len(priority_content)
        }
        
        for item in priority_content:
            for lang in item.get('languages', []):
                lang_lower = lang.lower()
                if lang_lower in stats['priority_content_distribution']:
                    stats['priority_content_distribution'][lang_lower] += 1
                else:
                    stats['priority_content_distribution'][lang_lower] = 1
        
        total_all_items = 0
        for category, items in all_content.items():
            total_all_items += len(items)
            for item in items:
                for lang in item.get('languages', []):
                    lang_lower = lang.lower()
                    if lang_lower in stats['all_content_distribution']:
                        stats['all_content_distribution'][lang_lower] += 1
                    else:
                        stats['all_content_distribution'][lang_lower] = 1
        
        stats['total_all_items'] = total_all_items
        
        return stats
        
    def _save_cache_to_disk(self):
        try:
            temp_file = f"{self.config.cache_file_path}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache_data, f, ensure_ascii=False, indent=2)
            
            os.replace(temp_file, self.config.cache_file_path)
            logger.info("CineBrain: 45-day strict cache saved successfully")
            
        except Exception as e:
            logger.error(f"CineBrain cache save error: {e}")
            
    def _load_cached_data(self):
        try:
            if os.path.exists(self.config.cache_file_path):
                with open(self.config.cache_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if self._validate_cache_structure(data):
                    valid_data = self._validate_cached_content_dates(data)
                    if valid_data:
                        self._cache_data = valid_data
                        self._last_update = datetime.now()
                        logger.info("CineBrain: Loaded cached 45-day new releases data successfully")
                    else:
                        logger.warning("CineBrain: Cached data too old, will refresh")
                else:
                    logger.warning("CineBrain: Invalid cache structure, will refresh")
                    
        except Exception as e:
            logger.error(f"CineBrain cache load error: {e}")
            
    def _validate_cached_content_dates(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            all_content = data.get('all_content', {})
            priority_content = data.get('priority_content', [])
            
            valid_priority = [
                item for item in priority_content 
                if self._is_within_strict_date_range(item.get('release_date', ''))
            ]
            
            valid_all_content = {}
            for category, items in all_content.items():
                valid_items = [
                    item for item in items 
                    if self._is_within_strict_date_range(item.get('release_date', ''))
                ]
                valid_all_content[category] = valid_items
                
            total_valid = sum(len(items) for items in valid_all_content.values())
            if total_valid < 10:
                logger.info("CineBrain: Cached content too sparse after date validation, will refresh")
                return None
                
            data['priority_content'] = valid_priority
            data['all_content'] = valid_all_content
            data['metadata']['total_items'] = total_valid
            data['metadata']['cache_validated'] = datetime.now(timezone.utc).isoformat()
            
            filtered_out = len(priority_content) - len(valid_priority)
            if filtered_out > 0:
                logger.info(f"CineBrain: Filtered {filtered_out} outdated items from cache")
                
            return data
            
        except Exception as e:
            logger.error(f"CineBrain cache validation error: {e}")
            return None
            
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
        start_date, end_date = self._get_strict_date_range()
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
                'sorting_strategy': 'language_priority_then_release_date_newest_first',
                'date_range_days': self.config.date_range_days,
                'strict_45_day_filter': True,
                'date_range_start': start_date.isoformat(),
                'date_range_end': end_date.isoformat(),
                'refresh_interval_minutes': self.config.refresh_interval_minutes,
                'telugu_focus': True,
                'cache_version': '2.3',
                'error': 'No data available within last 45 days',
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
            telugu_stats = self._calculate_telugu_stats()
            start_date, end_date = self._get_strict_date_range()
            
            base_stats = {
                'status': 'active',
                'cinebrain_service': 'new_releases',
                'strict_45_day_filter': True,
                'sorting_strategy': 'language_priority_then_release_date_newest_first',
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'days': self.config.date_range_days
                },
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
                'refresh_interval': self.config.refresh_interval_minutes
            }
            
            lang_stats = self.get_language_priority_stats()
            base_stats['language_priority_stats'] = lang_stats
            
            return base_stats
            
    def _calculate_telugu_stats(self) -> Dict[str, int]:
        if not self._cache_data:
            return {'total': 0, 'priority': 0, 'movies': 0, 'tv_shows': 0}
            
        telugu_stats = {'total': 0, 'priority': 0, 'movies': 0, 'tv_shows': 0}
        
        priority_content = self._cache_data.get('priority_content', [])
        for item in priority_content:
            if any('telugu' in lang.lower() for lang in item.get('languages', [])):
                telugu_stats['priority'] += 1
                
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
        logger.info("CineBrain 45-day strict new releases service initialized successfully")
        return service
        
    except Exception as e:
        logger.error(f"CineBrain new releases service initialization error: {e}")
        return None