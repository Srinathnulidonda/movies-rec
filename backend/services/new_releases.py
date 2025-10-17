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
            language_priorities=['telugu', 'english', 'hindi', 'malayalam', 'kannada', 'tamil'],
            date_range_days=30,
            refresh_interval_minutes=45,
            cache_file_path='cache/cinebrain_new_releases.json',
            max_items_per_category=60,
            api_timeout_seconds=12,
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
            pool_connections=10,
            pool_maxsize=20
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.headers.update({
            'User-Agent': 'CineBrain/2.0 (https://cinebrain.vercel.app)',
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
            name='CineBrainNewReleasesRefresh'
        )
        self._refresh_thread.start()
        logger.info("CineBrain new releases background refresh started")
        
    def stop_background_refresh(self):
        if self._refresh_thread:
            self._should_stop.set()
            self._refresh_thread.join(timeout=10)
            logger.info("CineBrain new releases background refresh stopped")
            
    def _background_refresh_worker(self):
        initial_delay = 30
        time.sleep(initial_delay)
        
        while not self._should_stop.is_set():
            try:
                logger.info("CineBrain: Starting scheduled new releases refresh")
                self.refresh_new_releases()
                
                wait_time = self.config.refresh_interval_minutes * 60
                for _ in range(wait_time):
                    if self._should_stop.wait(1):
                        return
                        
            except Exception as e:
                logger.error(f"CineBrain new releases background refresh error: {e}")
                if self._should_stop.wait(300):
                    return
                    
    def get_new_releases(self, force_refresh: bool = False) -> Dict[str, Any]:
        with self._cache_lock:
            if force_refresh or self._should_refresh():
                logger.info("CineBrain: Triggering new releases refresh")
                self.refresh_new_releases()
                
            return self._cache_data or self._get_empty_response()
            
    def _should_refresh(self) -> bool:
        if not self._cache_data or not self._last_update:
            return True
            
        age_minutes = (datetime.now() - self._last_update).total_seconds() / 60
        return age_minutes >= self.config.refresh_interval_minutes
        
    def refresh_new_releases(self):
        try:
            logger.info("CineBrain: Starting comprehensive new releases data collection")
            
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=self.config.date_range_days)
            
            all_content = []
            
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = []
                
                futures.append(executor.submit(
                    self._fetch_tmdb_releases, 'movie', start_date, end_date
                ))
                futures.append(executor.submit(
                    self._fetch_tmdb_releases, 'tv', start_date, end_date
                ))
                futures.append(executor.submit(
                    self._fetch_anime_releases, start_date, end_date
                ))
                
                for language in ['te', 'hi', 'ta']:
                    futures.append(executor.submit(
                        self._fetch_language_specific_releases, language, start_date, end_date
                    ))
                
                for future in as_completed(futures, timeout=180):
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
            processed_data = self._process_and_sort_content(unique_content)
            
            with self._cache_lock:
                self._cache_data = processed_data
                self._last_update = datetime.now()
                self._save_cache_to_disk()
                
            logger.info(f"CineBrain: Successfully refreshed {len(unique_content)} unique new releases")
            
        except Exception as e:
            logger.error(f"CineBrain: New releases refresh failed: {e}")
            
    def _deduplicate_content(self, content_list: List[CineBrainContent]) -> List[CineBrainContent]:
        seen_titles = set()
        seen_ids = set()
        unique_content = []
        
        content_list.sort(key=lambda x: (-x.popularity, -x.rating, -x.vote_count))
        
        for content in content_list:
            title_key = f"{content.title.lower()}_{content.content_type}"
            id_key = f"{content.tmdb_id or content.mal_id}_{content.content_type}"
            
            if title_key not in seen_titles and id_key not in seen_ids:
                seen_titles.add(title_key)
                seen_ids.add(id_key)
                unique_content.append(content)
                
        return unique_content
        
    def _fetch_tmdb_releases(self, content_type: str, start_date, end_date) -> List[CineBrainContent]:
        if not self.tmdb_api_key:
            logger.warning("CineBrain: TMDB API key not available")
            return []
            
        content_list = []
        
        try:
            for page in range(1, 8):
                url = f"https://api.themoviedb.org/3/discover/{content_type}"
                params = {
                    'api_key': self.tmdb_api_key,
                    'primary_release_date.gte': start_date.isoformat(),
                    'primary_release_date.lte': end_date.isoformat(),
                    'sort_by': 'primary_release_date.desc',
                    'page': page,
                    'vote_count.gte': 3,
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
                    
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"CineBrain TMDB {content_type} fetch error: {e}")
            
        return content_list[:self.config.max_items_per_category]
        
    def _fetch_language_specific_releases(self, language_code: str, start_date, end_date) -> List[CineBrainContent]:
        if not self.tmdb_api_key:
            return []
            
        content_list = []
        
        try:
            for content_type in ['movie', 'tv']:
                for page in range(1, 4):
                    url = f"https://api.themoviedb.org/3/discover/{content_type}"
                    params = {
                        'api_key': self.tmdb_api_key,
                        'with_original_language': language_code,
                        'primary_release_date.gte': start_date.isoformat(),
                        'primary_release_date.lte': end_date.isoformat(),
                        'sort_by': 'primary_release_date.desc',
                        'page': page,
                        'vote_count.gte': 1
                    }
                    
                    response = self._make_request(url, params)
                    if not response:
                        break
                        
                    results = response.get('results', [])
                    for item in results:
                        content = self._convert_tmdb_to_cinebrain(item, content_type)
                        if content:
                            content_list.append(content)
                            
                    time.sleep(0.2)
                    
        except Exception as e:
            logger.error(f"CineBrain language-specific {language_code} fetch error: {e}")
            
        return content_list
        
    def _fetch_anime_releases(self, start_date, end_date) -> List[CineBrainContent]:
        content_list = []
        
        try:
            for page in range(1, 5):
                url = "https://api.jikan.moe/v4/anime"
                params = {
                    'status': 'airing',
                    'order_by': 'start_date',
                    'sort': 'desc',
                    'page': page,
                    'limit': 25,
                    'min_score': 6.0
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
                    
                time.sleep(1.0)
                
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
                    wait_time = min(2 ** attempt * 3, 30)
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
            
            return CineBrainContent(
                id=item.get('id', 0),
                slug=slug,
                title=title.strip(),
                original_title=item.get('original_title') or item.get('original_name'),
                content_type=content_type,
                genres=self._extract_genres(item.get('genre_ids', [])),
                languages=self._extract_languages(item.get('original_language', 'en')),
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
            
    def _process_and_sort_content(self, content_list: List[CineBrainContent]) -> Dict[str, Any]:
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
                
        for category in all_content:
            all_content[category] = self._sort_by_date_and_language(all_content[category])
            
        priority_content = self._create_priority_content(all_content)
        
        return {
            'priority_content': priority_content,
            'all_content': all_content,
            'metadata': {
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'total_items': sum(len(items) for items in all_content.values()),
                'language_priorities': self.config.language_priorities,
                'date_range_days': self.config.date_range_days,
                'refresh_interval_minutes': self.config.refresh_interval_minutes,
                'cache_version': '2.0',
                'cinebrain_service': 'new_releases'
            }
        }
        
    def _sort_by_date_and_language(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def sort_key(item):
            try:
                release_date = datetime.fromisoformat(item['release_date'])
                days_ago = (datetime.now().date() - release_date.date()).days
                
                language_priority = 999
                for lang in item.get('languages', []):
                    if lang in self.config.language_priorities:
                        language_priority = min(language_priority, self.config.language_priorities.index(lang))
                        
                popularity_score = item.get('popularity', 0)
                rating_score = item.get('rating', 0) * item.get('vote_count', 1)
                
                return (days_ago, language_priority, -popularity_score, -rating_score)
            except:
                return (999, 999, 0, 0)
                
        return sorted(content_list, key=sort_key)
        
    def _create_priority_content(self, all_content: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        priority_items = []
        
        language_quotas = {
            'telugu': 8,
            'english': 6,
            'hindi': 4,
            'malayalam': 3,
            'kannada': 3,
            'tamil': 3
        }
        
        for category in ['movies', 'tv_shows', 'anime']:
            category_items = all_content.get(category, [])
            
            language_buckets = {lang: [] for lang in self.config.language_priorities}
            other_bucket = []
            
            for item in category_items:
                categorized = False
                for lang in item.get('languages', []):
                    if lang in self.config.language_priorities:
                        language_buckets[lang].append(item)
                        categorized = True
                        break
                        
                if not categorized:
                    other_bucket.append(item)
                    
            for lang in self.config.language_priorities:
                quota = language_quotas.get(lang, 2)
                if category == 'anime' and lang != 'japanese':
                    quota = 1
                priority_items.extend(language_buckets[lang][:quota])
                
            priority_items.extend(other_bucket[:3])
            
        final_priority = self._sort_by_date_and_language(priority_items)
        return final_priority[:40]
        
    def _save_cache_to_disk(self):
        try:
            temp_file = f"{self.config.cache_file_path}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache_data, f, ensure_ascii=False, indent=2)
            
            os.replace(temp_file, self.config.cache_file_path)
            logger.info("CineBrain: Cache saved successfully")
            
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
                    logger.info("CineBrain: Loaded cached new releases data successfully")
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
                'cache_version': '2.0',
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
            
            return {
                'status': 'active',
                'cinebrain_service': 'new_releases',
                'last_update': self._last_update.isoformat() if self._last_update else None,
                'total_items': metadata.get('total_items', 0),
                'priority_items': len(self._cache_data.get('priority_content', [])),
                'movies': len(self._cache_data.get('all_content', {}).get('movies', [])),
                'tv_shows': len(self._cache_data.get('all_content', {}).get('tv_shows', [])),
                'anime': len(self._cache_data.get('all_content', {}).get('anime', [])),
                'background_refresh': self._refresh_thread.is_alive() if self._refresh_thread else False,
                'cache_age_minutes': (datetime.now() - self._last_update).total_seconds() / 60 if self._last_update else None,
                'language_priorities': self.config.language_priorities,
                'refresh_interval': self.config.refresh_interval_minutes
            }

def init_cinebrain_new_releases_service(app, db, models, services):
    try:
        service = CineBrainNewReleasesService(app, db, models, services)
        service.start_background_refresh()
        
        @app.teardown_appcontext
        def cleanup_cinebrain_service(error):
            if hasattr(app, '_cinebrain_new_releases_service'):
                app._cinebrain_new_releases_service.stop_background_refresh()
                
        app._cinebrain_new_releases_service = service
        logger.info("CineBrain new releases service initialized successfully")
        return service
        
    except Exception as e:
        logger.error(f"CineBrain new releases service initialization error: {e}")
        return None