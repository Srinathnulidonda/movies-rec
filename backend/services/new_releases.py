#backend/services/new_releases.py
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from flask import Blueprint, request, jsonify, current_app
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event, Thread
from pathlib import Path
import json
import time
import logging
import hashlib
import asyncio
import aiohttp
import pytz
import os
import atexit
from collections import defaultdict, OrderedDict

logger = logging.getLogger(__name__)

@dataclass
class NewReleaseItem:
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
    tmdb_id: Optional[int] = None
    mal_id: Optional[int] = None
    days_since_release: int = 0
    language_priority: int = 999
    primary_language: str = "Other"
    freshness_score: float = 0.0
    combined_score: float = 0.0

    def __post_init__(self):
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        now = datetime.now(timezone.utc)
        if self.release_date.tzinfo is None:
            release_dt = pytz.UTC.localize(self.release_date)
        else:
            release_dt = self.release_date
        
        delta = now - release_dt
        self.days_since_release = max(0, delta.days)
        
        self._set_language_priority()
        self._calculate_scores()
    
    def _set_language_priority(self):
        language_map = {
            ('telugu', 'te'): (1, 'Telugu'),
            ('english', 'en'): (2, 'English'),
            ('hindi', 'hi'): (3, 'Hindi'),
            ('malayalam', 'ml'): (4, 'Malayalam'),
            ('kannada', 'kn'): (5, 'Kannada'),
            ('tamil', 'ta'): (6, 'Tamil')
        }
        
        for lang in self.languages:
            lang_lower = lang.lower() if lang else ''
            for lang_keys, (priority, name) in language_map.items():
                if lang_lower in lang_keys:
                    self.language_priority = priority
                    self.primary_language = name
                    return
    
    def _calculate_scores(self):
        if self.days_since_release == 0:
            self.freshness_score = 100.0
        elif self.days_since_release == 1:
            self.freshness_score = 80.0
        elif self.days_since_release <= 3:
            self.freshness_score = 70.0 - (self.days_since_release * 5)
        elif self.days_since_release <= 7:
            self.freshness_score = 50.0 - (self.days_since_release * 3)
        else:
            self.freshness_score = max(10, 40 - self.days_since_release)
        
        quality_score = (
            (self.vote_average / 10) * 40 +
            min(30, self.popularity / 10) +
            min(30, self.vote_count / 100)
        )
        
        language_multiplier = {1: 2.0, 2: 1.5, 3: 1.3, 4: 1.1, 5: 1.1, 6: 1.1}.get(
            self.language_priority, 1.0
        )
        
        self.combined_score = (self.freshness_score * 0.7 + quality_score * 0.3) * language_multiplier

class CineBrainNewReleasesConfig:
    def __init__(self):
        self.TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '')
        self.JIKAN_BASE_URL = 'https://api.jikan.moe/v4'
        self.TMDB_BASE_URL = 'https://api.themoviedb.org/3'
        self.REFRESH_INTERVAL_MINUTES = int(os.environ.get('CINEBRAIN_NEW_RELEASES_REFRESH_MINUTES', '15'))
        self.CACHE_FILE = Path(os.environ.get('CINEBRAIN_NEW_RELEASES_CACHE_FILE', 'data/cinebrain_new_releases_cache.json'))
        self.LANGUAGE_PRIORITIES = [
            'Telugu', 'English', 'Hindi', 'Malayalam', 'Kannada', 'Tamil'
        ]
        self.DAYS_BACK = int(os.environ.get('CINEBRAIN_NEW_RELEASES_DAYS_BACK', '30'))
        self.MAX_ITEMS_PER_CATEGORY = int(os.environ.get('CINEBRAIN_NEW_RELEASES_MAX_ITEMS', '50'))
        self.REQUEST_TIMEOUT = int(os.environ.get('CINEBRAIN_NEW_RELEASES_TIMEOUT', '10'))
        self.MAX_RETRIES = int(os.environ.get('CINEBRAIN_NEW_RELEASES_MAX_RETRIES', '3'))
        self.CONCURRENT_REQUESTS = int(os.environ.get('CINEBRAIN_NEW_RELEASES_CONCURRENT', '5'))

class CineBrainNewReleasesCache:
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._data = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if self._is_cache_valid(data):
                        return data
        except Exception as e:
            logger.error(f"CineBrain cache load error: {e}")
        return self._empty_cache()
    
    def _is_cache_valid(self, data: Dict[str, Any]) -> bool:
        try:
            timestamp = data.get('metadata', {}).get('generated_at')
            if not timestamp:
                return False
            
            cache_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            age_minutes = (datetime.now(timezone.utc) - cache_time).total_seconds() / 60
            
            return age_minutes < CineBrainNewReleasesConfig().REFRESH_INTERVAL_MINUTES * 2
        except Exception:
            return False
    
    def _empty_cache(self) -> Dict[str, Any]:
        return {
            'priority_content': [],
            'all_content': [],
            'metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'total_items': 0,
                'languages_found': [],
                'content_types': [],
                'is_stale': True,
                'platform': 'CineBrain'
            }
        }
    
    def get(self) -> Dict[str, Any]:
        with self._lock:
            return self._data.copy()
    
    def set(self, data: Dict[str, Any]) -> bool:
        try:
            with self._lock:
                self._data = data
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                return True
        except Exception as e:
            logger.error(f"CineBrain cache save error: {e}")
            return False
    
    def is_stale(self) -> bool:
        with self._lock:
            return self._data.get('metadata', {}).get('is_stale', True)

class CineBrainNewReleasesAPIClient:
    def __init__(self, config: CineBrainNewReleasesConfig):
        self.config = config
        self._session_lock = Lock()
        
    async def _create_session(self) -> aiohttp.ClientSession:
        timeout = aiohttp.ClientTimeout(total=self.config.REQUEST_TIMEOUT)
        connector = aiohttp.TCPConnector(limit=self.config.CONCURRENT_REQUESTS)
        return aiohttp.ClientSession(timeout=timeout, connector=connector)
    
    async def fetch_tmdb_content(self, content_type: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        session = await self._create_session()
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.DAYS_BACK)
            
            url = f"{self.config.TMDB_BASE_URL}/discover/{content_type}"
            params = {
                'api_key': self.config.TMDB_API_KEY,
                'sort_by': 'release_date.desc' if content_type == 'movie' else 'first_air_date.desc',
                'include_adult': 'false',
                'page': '1'
            }
            
            if content_type == 'movie':
                params.update({
                    'primary_release_date.gte': start_date.strftime('%Y-%m-%d'),
                    'primary_release_date.lte': end_date.strftime('%Y-%m-%d')
                })
            else:
                params.update({
                    'first_air_date.gte': start_date.strftime('%Y-%m-%d'),
                    'first_air_date.lte': end_date.strftime('%Y-%m-%d')
                })
            
            if language:
                params['with_original_language'] = str(language)
            
            all_results = []
            max_pages = 3
            
            for page in range(1, max_pages + 1):
                params['page'] = str(page)
                
                for attempt in range(self.config.MAX_RETRIES):
                    try:
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                results = data.get('results', [])
                                all_results.extend(results)
                                
                                if not results or page >= data.get('total_pages', 1):
                                    return all_results
                                break
                            elif response.status == 429:
                                await asyncio.sleep(2 ** attempt)
                            else:
                                logger.warning(f"CineBrain TMDB API error {response.status} for {content_type}")
                                break
                    except Exception as e:
                        logger.error(f"CineBrain TMDB request error (attempt {attempt + 1}): {e}")
                        if attempt == self.config.MAX_RETRIES - 1:
                            break
                        await asyncio.sleep(1)
                
                await asyncio.sleep(0.25)
            
            return all_results
        finally:
            await session.close()
    
    async def fetch_anime_content(self) -> List[Dict[str, Any]]:
        session = await self._create_session()
        try:
            url = f"{self.config.JIKAN_BASE_URL}/seasons/now"
            
            for attempt in range(self.config.MAX_RETRIES):
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data.get('data', [])[:self.config.MAX_ITEMS_PER_CATEGORY]
                        elif response.status == 429:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            break
                except Exception as e:
                    logger.error(f"CineBrain Jikan request error (attempt {attempt + 1}): {e}")
                    if attempt < self.config.MAX_RETRIES - 1:
                        await asyncio.sleep(1)
            
            return []
        finally:
            await session.close()

class CineBrainNewReleasesProcessor:
    def __init__(self, config: CineBrainNewReleasesConfig):
        self.config = config
        self.api_client = CineBrainNewReleasesAPIClient(config)
    
    def _parse_tmdb_item(self, item: Dict[str, Any], content_type: str) -> NewReleaseItem:
        release_date_key = 'release_date' if content_type == 'movie' else 'first_air_date'
        title_key = 'title' if content_type == 'movie' else 'name'
        original_title_key = 'original_title' if content_type == 'movie' else 'original_name'
        
        try:
            release_date_str = item.get(release_date_key, '')
            release_date = datetime.strptime(release_date_str, '%Y-%m-%d') if release_date_str else datetime.now()
            release_date = pytz.UTC.localize(release_date)
        except (ValueError, TypeError):
            release_date = datetime.now(timezone.utc)
        
        languages = []
        orig_lang = item.get('original_language', '')
        if orig_lang:
            lang_name_map = {
                'te': 'Telugu', 'en': 'English', 'hi': 'Hindi',
                'ml': 'Malayalam', 'kn': 'Kannada', 'ta': 'Tamil',
                'ja': 'Japanese', 'ko': 'Korean', 'es': 'Spanish',
                'fr': 'French', 'de': 'German', 'it': 'Italian'
            }
            languages.append(lang_name_map.get(orig_lang, orig_lang.upper()))
        
        genre_map = {
            28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
            80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
            14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
            9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
            10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
        }
        genres = [genre_map.get(gid, "Unknown") for gid in item.get('genre_ids', [])]
        
        poster_path = item.get('poster_path')
        if poster_path and not poster_path.startswith('http'):
            poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        backdrop_path = item.get('backdrop_path')
        if backdrop_path and not backdrop_path.startswith('http'):
            backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        
        return NewReleaseItem(
            id=f"cinebrain_tmdb_{content_type}_{item['id']}",
            title=item.get(title_key, ''),
            original_title=item.get(original_title_key),
            content_type=content_type,
            release_date=release_date,
            languages=languages,
            genres=genres,
            popularity=float(item.get('popularity', 0)),
            vote_count=int(item.get('vote_count', 0)),
            vote_average=float(item.get('vote_average', 0)),
            poster_path=poster_path,
            backdrop_path=backdrop_path,
            overview=item.get('overview'),
            tmdb_id=item.get('id')
        )
    
    def _parse_anime_item(self, item: Dict[str, Any]) -> NewReleaseItem:
        try:
            aired = item.get('aired', {})
            from_date = aired.get('from') if aired else None
            
            if from_date:
                release_date = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
            else:
                release_date = datetime.now(timezone.utc)
        except (ValueError, TypeError):
            release_date = datetime.now(timezone.utc)
        
        genres = [g.get('name', 'Unknown') for g in item.get('genres', [])]
        
        popularity = 0.0
        vote_count = 0
        vote_average = 0.0
        
        try:
            if item.get('popularity') is not None:
                popularity = float(item.get('popularity', 0))
        except (ValueError, TypeError):
            popularity = 0.0
            
        try:
            if item.get('scored_by') is not None:
                vote_count = int(item.get('scored_by', 0))
        except (ValueError, TypeError):
            vote_count = 0
            
        try:
            if item.get('score') is not None:
                vote_average = float(item.get('score', 0))
        except (ValueError, TypeError):
            vote_average = 0.0
        
        return NewReleaseItem(
            id=f"cinebrain_mal_anime_{item.get('mal_id', 0)}",
            title=item.get('title', ''),
            original_title=item.get('title_japanese'),
            content_type='anime',
            release_date=release_date,
            languages=['Japanese'],
            genres=genres,
            popularity=popularity,
            vote_count=vote_count,
            vote_average=vote_average,
            poster_path=item.get('images', {}).get('jpg', {}).get('large_image_url'),
            backdrop_path=item.get('images', {}).get('jpg', {}).get('large_image_url'),
            overview=item.get('synopsis'),
            mal_id=item.get('mal_id')
        )
    
    async def fetch_all_content(self) -> List[NewReleaseItem]:
        all_items = []
        
        language_codes = {
            'Telugu': 'te', 'English': 'en', 'Hindi': 'hi',
            'Malayalam': 'ml', 'Kannada': 'kn', 'Tamil': 'ta'
        }
        
        tasks = []
        
        for content_type in ['movie', 'tv']:
            tasks.append(self.api_client.fetch_tmdb_content(content_type))
            
            for lang_name, lang_code in language_codes.items():
                tasks.append(self.api_client.fetch_tmdb_content(content_type, lang_code))
        
        tasks.append(self.api_client.fetch_anime_content())
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            task_index = 0
            
            for content_type in ['movie', 'tv']:
                general_result = results[task_index]
                task_index += 1
                
                if isinstance(general_result, list):
                    for item in general_result:
                        try:
                            parsed_item = self._parse_tmdb_item(item, content_type)
                            all_items.append(parsed_item)
                        except Exception as e:
                            logger.error(f"Error parsing CineBrain {content_type} item: {e}")
                
                for lang_name in language_codes.keys():
                    lang_result = results[task_index]
                    task_index += 1
                    
                    if isinstance(lang_result, list):
                        for item in lang_result:
                            try:
                                parsed_item = self._parse_tmdb_item(item, content_type)
                                all_items.append(parsed_item)
                            except Exception as e:
                                logger.error(f"Error parsing CineBrain {lang_name} {content_type} item: {e}")
            
            anime_result = results[task_index]
            if isinstance(anime_result, list):
                for item in anime_result:
                    try:
                        parsed_item = self._parse_anime_item(item)
                        all_items.append(parsed_item)
                    except Exception as e:
                        logger.error(f"Error parsing CineBrain anime item: {e}")
        
        except Exception as e:
            logger.error(f"Error in CineBrain fetch_all_content: {e}")
        
        seen_ids = set()
        unique_items = []
        for item in all_items:
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                unique_items.append(item)
        
        return unique_items
    
    def process_content(self, items: List[NewReleaseItem]) -> Dict[str, Any]:
        all_content = sorted(items, key=lambda x: (x.days_since_release, -x.combined_score))
        
        priority_content = []
        priority_seen = set()
        
        for priority in range(1, 7):
            priority_items = [
                item for item in all_content 
                if item.language_priority == priority and item.id not in priority_seen
            ]
            
            priority_items = sorted(priority_items, key=lambda x: (x.days_since_release, -x.combined_score))
            
            limit = self.config.MAX_ITEMS_PER_CATEGORY if priority == 1 else self.config.MAX_ITEMS_PER_CATEGORY // 2
            
            for item in priority_items[:limit]:
                priority_content.append(item)
                priority_seen.add(item.id)
        
        other_items = [
            item for item in all_content[:self.config.MAX_ITEMS_PER_CATEGORY] 
            if item.id not in priority_seen
        ]
        priority_content.extend(other_items)
        
        priority_content = priority_content[:self.config.MAX_ITEMS_PER_CATEGORY]
        
        languages_found = list(set(item.primary_language for item in all_content))
        content_types = list(set(item.content_type for item in all_content))
        
        return {
            'priority_content': [asdict(item) for item in priority_content],
            'all_content': [asdict(item) for item in all_content],
            'metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'total_items': len(all_content),
                'priority_items': len(priority_content),
                'languages_found': sorted(languages_found),
                'content_types': sorted(content_types),
                'language_priorities': self.config.LANGUAGE_PRIORITIES,
                'days_back': self.config.DAYS_BACK,
                'is_stale': False,
                'platform': 'CineBrain',
                'next_refresh': (datetime.now(timezone.utc) + 
                               timedelta(minutes=self.config.REFRESH_INTERVAL_MINUTES)).isoformat()
            }
        }

class CineBrainNewReleasesService:
    def __init__(self, config: CineBrainNewReleasesConfig):
        self.config = config
        self.cache = CineBrainNewReleasesCache(config.CACHE_FILE)
        self.processor = CineBrainNewReleasesProcessor(config)
        self._refresh_lock = Lock()
        self._shutdown_event = Event()
        self._background_thread = None
        self._is_refreshing = False
        
        atexit.register(self.shutdown)
    
    def start_background_refresh(self):
        if self._background_thread is None or not self._background_thread.is_alive():
            self._shutdown_event.clear()
            self._background_thread = Thread(target=self._background_refresh_loop, daemon=True)
            self._background_thread.start()
            logger.info("CineBrain background refresh thread started")
    
    def _background_refresh_loop(self):
        while not self._shutdown_event.is_set():
            try:
                if self.cache.is_stale():
                    logger.info("CineBrain cache is stale, triggering refresh")
                    asyncio.run(self._refresh_content())
                
                self._shutdown_event.wait(timeout=self.config.REFRESH_INTERVAL_MINUTES * 60)
            except Exception as e:
                logger.error(f"CineBrain background refresh error: {e}")
                self._shutdown_event.wait(timeout=300)
    
    async def _refresh_content(self) -> bool:
        if self._is_refreshing:
            return False
        
        try:
            with self._refresh_lock:
                if self._is_refreshing:
                    return False
                self._is_refreshing = True
            
            logger.info("Starting CineBrain content refresh")
            start_time = time.time()
            
            items = await self.processor.fetch_all_content()
            processed_data = self.processor.process_content(items)
            
            success = self.cache.set(processed_data)
            
            duration = time.time() - start_time
            logger.info(f"CineBrain content refresh completed in {duration:.2f}s, success: {success}")
            
            return success
            
        except Exception as e:
            logger.error(f"CineBrain content refresh failed: {e}")
            return False
        finally:
            self._is_refreshing = False
    
    def get_content(self, force_refresh: bool = False) -> Dict[str, Any]:
        if force_refresh or self.cache.is_stale():
            try:
                asyncio.run(self._refresh_content())
            except Exception as e:
                logger.error(f"CineBrain forced refresh failed: {e}")
        
        return self.cache.get()
    
    def shutdown(self):
        if self._shutdown_event:
            self._shutdown_event.set()
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=5)
            logger.info("CineBrain background refresh thread stopped")

config = CineBrainNewReleasesConfig()
cinebrain_new_releases_service = CineBrainNewReleasesService(config)

new_releases_bp = Blueprint('new_releases', __name__)

@new_releases_bp.route('/api/new-releases', methods=['GET'])
def get_new_releases():
    try:
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        content_type_filter = request.args.get('type')
        language_filter = request.args.get('language')
        limit = request.args.get('limit', type=int)
        
        data = cinebrain_new_releases_service.get_content(force_refresh=force_refresh)
        
        if content_type_filter:
            data['priority_content'] = [
                item for item in data['priority_content'] 
                if item.get('content_type') == content_type_filter
            ]
            data['all_content'] = [
                item for item in data['all_content'] 
                if item.get('content_type') == content_type_filter
            ]
        
        if language_filter:
            data['priority_content'] = [
                item for item in data['priority_content'] 
                if language_filter.lower() in [lang.lower() for lang in item.get('languages', [])]
            ]
            data['all_content'] = [
                item for item in data['all_content'] 
                if language_filter.lower() in [lang.lower() for lang in item.get('languages', [])]
            ]
        
        if limit:
            data['priority_content'] = data['priority_content'][:limit]
            data['all_content'] = data['all_content'][:limit]
        
        data['metadata']['filtered'] = bool(content_type_filter or language_filter or limit)
        data['metadata']['api_endpoint'] = '/api/new-releases'
        data['metadata']['cache_status'] = 'fresh' if not cinebrain_new_releases_service.cache.is_stale() else 'stale'
        data['metadata']['platform'] = 'CineBrain'
        
        return jsonify(data), 200
        
    except Exception as e:
        logger.error(f"CineBrain new releases endpoint error: {e}")
        
        fallback_data = {
            'priority_content': [],
            'all_content': [],
            'metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'total_items': 0,
                'error': 'CineBrain service temporarily unavailable',
                'is_stale': True,
                'cache_status': 'error',
                'platform': 'CineBrain'
            }
        }
        
        return jsonify(fallback_data), 200

@new_releases_bp.route('/api/new-releases/health', methods=['GET'])
def new_releases_health():
    try:
        cache_data = cinebrain_new_releases_service.cache.get()
        metadata = cache_data.get('metadata', {})
        
        health_info = {
            'status': 'healthy',
            'service': 'cinebrain_new_releases',
            'platform': 'CineBrain',
            'cache_status': 'fresh' if not cinebrain_new_releases_service.cache.is_stale() else 'stale',
            'last_generated': metadata.get('generated_at'),
            'total_items': metadata.get('total_items', 0),
            'priority_items': metadata.get('priority_items', 0),
            'languages_available': metadata.get('languages_found', []),
            'content_types': metadata.get('content_types', []),
            'background_refresh_active': cinebrain_new_releases_service._background_thread and cinebrain_new_releases_service._background_thread.is_alive(),
            'is_refreshing': cinebrain_new_releases_service._is_refreshing,
            'config': {
                'refresh_interval_minutes': config.REFRESH_INTERVAL_MINUTES,
                'days_back': config.DAYS_BACK,
                'max_items_per_category': config.MAX_ITEMS_PER_CATEGORY
            }
        }
        
        return jsonify(health_info), 200
        
    except Exception as e:
        logger.error(f"CineBrain health check error: {e}")
        return jsonify({
            'status': 'error',
            'service': 'cinebrain_new_releases',
            'platform': 'CineBrain',
            'error': str(e)
        }), 500

@new_releases_bp.route('/api/new-releases/refresh', methods=['POST'])
def trigger_refresh():
    try:
        force = request.json.get('force', False) if request.is_json else False
        
        if force or cinebrain_new_releases_service.cache.is_stale():
            success = asyncio.run(cinebrain_new_releases_service._refresh_content())
            
            return jsonify({
                'success': success,
                'message': 'CineBrain refresh completed' if success else 'CineBrain refresh failed',
                'platform': 'CineBrain',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 200 if success else 500
        else:
            return jsonify({
                'success': False,
                'message': 'CineBrain cache is fresh, refresh not needed',
                'platform': 'CineBrain',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 200
            
    except Exception as e:
        logger.error(f"CineBrain manual refresh error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'platform': 'CineBrain',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

def init_new_releases_service():
    try:
        cinebrain_new_releases_service.start_background_refresh()
        logger.info("CineBrain new releases service initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize CineBrain new releases service: {e}")
        return False

if __name__ == '__main__':
    init_new_releases_service()