import os
import json
import asyncio
import logging
import signal
import aiofiles
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pytz
import httpx
import hashlib
from collections import defaultdict
from pathlib import Path
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('new_releases.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ReleaseItem:
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
    runtime: Optional[int]
    tmdb_id: int
    
    days_since_release: int = 0
    hours_since_release: int = 0
    is_today: bool = False
    is_yesterday: bool = False
    is_this_week: bool = False
    
    language_priority: int = 999
    language_name: str = "Other"
    is_telugu: bool = False
    
    freshness_score: float = 0.0
    quality_score: float = 0.0
    final_score: float = 0.0

class ContinuousNewReleasesService:
    def __init__(self):
        self.tmdb_api_key = os.getenv('TMDB_API_KEY')
        self.refresh_interval = int(os.getenv('REFRESH_INTERVAL_SECONDS', '300'))
        self.output_file = os.getenv('OUTPUT_FILE', 'new_releases.json')
        self.max_age_days = int(os.getenv('MAX_AGE_DAYS', '30'))
        self.timezone = os.getenv('USER_TIMEZONE', 'Asia/Kolkata')
        self.max_results = int(os.getenv('MAX_RESULTS', '100'))
        self.rate_limit_delay = float(os.getenv('RATE_LIMIT_DELAY', '0.5'))
        
        self.base_url = "https://api.themoviedb.org/3"
        self.running = False
        self.http_client = None
        
        self.language_config = {
            ('telugu', 'te'): (1, 'Telugu'),
            ('english', 'en'): (2, 'English'), 
            ('hindi', 'hi'): (3, 'Hindi'),
            ('malayalam', 'ml'): (4, 'Malayalam'),
            ('kannada', 'kn'): (5, 'Kannada'),
            ('tamil', 'ta'): (6, 'Tamil')
        }
        
        self.telugu_patterns = {
            'languages': ['te', 'telugu'],
            'keywords': [
                'tollywood', 'telugu cinema', 'telugu movie', 'telugu film',
                'andhra pradesh', 'telangana', 'hyderabad', 'vijayawada',
                'visakhapatnam', 'warangal'
            ],
            'production_companies': [
                'AVM Productions', 'Vyjayanthi Movies', 'Mythri Movie Makers',
                'Geetha Arts', 'Sri Venkateswara Creations', 'Konidela Production',
                'UV Creations', 'Haarika & Hassine Creations', 'Sithara Entertainments',
                'DVV Entertainments', 'Suresh Productions', 'Bhavya Creations'
            ]
        }
        
        self.genre_map = {
            28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
            80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
            14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
            9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
            10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
        }
        
        self.last_success_time = None
        self.error_count = 0
        self.total_processed = 0

    async def start(self):
        if not self.tmdb_api_key:
            logger.error("TMDB_API_KEY environment variable not set")
            return
            
        self.running = True
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(15.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Starting continuous new releases service (refresh every {self.refresh_interval}s)")
        
        try:
            while self.running:
                start_time = time.time()
                
                try:
                    await self._process_releases()
                    self.last_success_time = datetime.now()
                    self.error_count = 0
                    
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Processing error (count: {self.error_count}): {e}")
                    
                    if self.error_count >= 5:
                        logger.critical("Too many consecutive errors, backing off")
                        await asyncio.sleep(min(self.refresh_interval * 2, 1800))
                
                processing_time = time.time() - start_time
                sleep_time = max(0, self.refresh_interval - processing_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
        except asyncio.CancelledError:
            logger.info("Service cancelled")
        finally:
            await self._cleanup()

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.running = False

    async def _cleanup(self):
        if self.http_client:
            await self.http_client.aclose()
        logger.info("Cleanup completed")

    async def _process_releases(self):
        logger.info("Processing new releases")
        
        all_releases = []
        seen_ids = set()
        
        now = datetime.now(pytz.timezone(self.timezone))
        cutoff_date = now - timedelta(days=self.max_age_days)
        
        languages = ['te', 'en', 'hi', 'ml', 'kn', 'ta']
        content_types = ['movie', 'tv']
        
        for content_type in content_types:
            for lang_code in languages:
                try:
                    releases = await self._fetch_language_releases(lang_code, content_type, cutoff_date)
                    for item in releases:
                        if item['id'] not in seen_ids:
                            release_item = self._parse_release(item, content_type, lang_code, now)
                            if release_item and release_item.days_since_release <= self.max_age_days:
                                all_releases.append(release_item)
                                seen_ids.add(item['id'])
                    
                    await asyncio.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch {lang_code} {content_type}: {e}")
                    continue
        
        general_releases = await self._fetch_general_releases(cutoff_date)
        for item in general_releases:
            if item['id'] not in seen_ids:
                release_item = self._parse_release(item, 'movie', None, now)
                if release_item and release_item.days_since_release <= self.max_age_days:
                    if self._is_telugu_content(item):
                        release_item.is_telugu = True
                        release_item.language_priority = 1
                        release_item.language_name = "Telugu"
                    all_releases.append(release_item)
                    seen_ids.add(item['id'])
        
        self._calculate_scores(all_releases, now)
        sorted_releases = self._sort_releases(all_releases)
        
        await self._write_output(sorted_releases[:self.max_results])
        
        self.total_processed = len(sorted_releases)
        logger.info(f"Processed {len(sorted_releases)} releases, saved top {min(len(sorted_releases), self.max_results)}")

    async def _fetch_language_releases(self, language_code: str, content_type: str, cutoff_date: datetime) -> List[Dict]:
        end_date = datetime.now()
        start_date = cutoff_date
        
        url = f"{self.base_url}/discover/{content_type}"
        params = {
            'api_key': self.tmdb_api_key,
            'with_original_language': language_code,
            'sort_by': 'release_date.desc' if content_type == 'movie' else 'first_air_date.desc',
            'include_adult': False,
            'vote_count.gte': 1
        }
        
        if content_type == 'movie':
            params['primary_release_date.gte'] = start_date.strftime('%Y-%m-%d')
            params['primary_release_date.lte'] = end_date.strftime('%Y-%m-%d')
        else:
            params['first_air_date.gte'] = start_date.strftime('%Y-%m-%d')
            params['first_air_date.lte'] = end_date.strftime('%Y-%m-%d')
        
        all_results = []
        max_pages = 5
        
        for page in range(1, max_pages + 1):
            params['page'] = page
            
            try:
                response = await self.http_client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                results = data.get('results', [])
                all_results.extend(results)
                
                if not results or page >= data.get('total_pages', 1):
                    break
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    await asyncio.sleep(2)
                    continue
                logger.warning(f"HTTP error {e.response.status_code} for {language_code} {content_type}")
                break
            except Exception as e:
                logger.warning(f"Request error for {language_code} {content_type}: {e}")
                break
        
        return all_results

    async def _fetch_general_releases(self, cutoff_date: datetime) -> List[Dict]:
        end_date = datetime.now()
        
        url = f"{self.base_url}/discover/movie"
        params = {
            'api_key': self.tmdb_api_key,
            'sort_by': 'popularity.desc',
            'primary_release_date.gte': cutoff_date.strftime('%Y-%m-%d'),
            'primary_release_date.lte': end_date.strftime('%Y-%m-%d'),
            'include_adult': False,
            'vote_count.gte': 10,
            'page': 1
        }
        
        try:
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get('results', [])[:50]
        except Exception as e:
            logger.warning(f"Failed to fetch general releases: {e}")
            return []

    def _parse_release(self, item: Dict, content_type: str, detected_language: Optional[str], now: datetime) -> Optional[ReleaseItem]:
        try:
            if content_type == 'movie':
                release_date_str = item.get('release_date', '')
            else:
                release_date_str = item.get('first_air_date', '')
            
            if not release_date_str:
                return None
                
            try:
                release_date = datetime.strptime(release_date_str, '%Y-%m-%d')
                release_date = pytz.timezone(self.timezone).localize(release_date)
            except:
                return None
            
            delta = now - release_date
            days_since = max(0, delta.days)
            hours_since = max(0, int(delta.total_seconds() / 3600))
            
            if days_since > self.max_age_days:
                return None
            
            orig_lang = item.get('original_language', '')
            languages = [orig_lang] if orig_lang else []
            
            language_priority = 999
            language_name = "Other"
            is_telugu = False
            
            if detected_language:
                lang_code_map = {
                    'telugu': 'te', 'english': 'en', 'hindi': 'hi',
                    'malayalam': 'ml', 'kannada': 'kn', 'tamil': 'ta'
                }
                lang_code = lang_code_map.get(detected_language, orig_lang)
                if lang_code not in languages:
                    languages.append(lang_code)
            
            for lang in languages:
                for lang_keys, (priority, name) in self.language_config.items():
                    if lang.lower() in lang_keys:
                        if priority < language_priority:
                            language_priority = priority
                            language_name = name
                            is_telugu = (priority == 1)
                        break
            
            if self._is_telugu_content(item):
                is_telugu = True
                language_priority = 1
                language_name = "Telugu"
                if 'te' not in languages:
                    languages.insert(0, 'te')
            
            poster_path = item.get('poster_path')
            if poster_path and not poster_path.startswith('http'):
                poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
            
            backdrop_path = item.get('backdrop_path')
            if backdrop_path and not backdrop_path.startswith('http'):
                backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
            
            genres = [self.genre_map.get(gid, "Unknown") for gid in item.get('genre_ids', [])]
            
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday_start = today_start - timedelta(days=1)
            
            return ReleaseItem(
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
                runtime=item.get('runtime'),
                tmdb_id=item['id'],
                days_since_release=days_since,
                hours_since_release=hours_since,
                is_today=today_start <= release_date < today_start + timedelta(days=1),
                is_yesterday=yesterday_start <= release_date < today_start,
                is_this_week=days_since <= 7,
                language_priority=language_priority,
                language_name=language_name,
                is_telugu=is_telugu
            )
            
        except Exception as e:
            logger.warning(f"Error parsing release item: {e}")
            return None

    def _is_telugu_content(self, item: Dict) -> bool:
        if item.get('original_language', '').lower() in self.telugu_patterns['languages']:
            return True
        
        text_fields = [
            item.get('title', ''),
            item.get('original_title', ''),
            item.get('overview', '')
        ]
        text_to_check = ' '.join(text_fields).lower()
        
        for keyword in self.telugu_patterns['keywords']:
            if keyword in text_to_check:
                return True
        
        return False

    def _calculate_scores(self, releases: List[ReleaseItem], now: datetime):
        for release in releases:
            if release.is_today:
                release.freshness_score = 100.0
            elif release.is_yesterday:
                release.freshness_score = 85.0
            elif release.days_since_release <= 3:
                release.freshness_score = 70.0 - (release.days_since_release * 5)
            elif release.is_this_week:
                release.freshness_score = 50.0 - (release.days_since_release * 3)
            else:
                release.freshness_score = max(10, 40 - (release.days_since_release * 1))
            
            rating_score = (release.vote_average / 10) * 40 if release.vote_average > 0 else 20
            popularity_score = min(30, release.popularity / 10) if release.popularity > 0 else 10
            vote_score = min(30, release.vote_count / 100) if release.vote_count > 0 else 10
            release.quality_score = rating_score + popularity_score + vote_score
            
            base_score = (release.freshness_score * 0.7) + (release.quality_score * 0.3)
            
            if release.is_telugu:
                base_score *= 3.0
            elif release.language_priority == 2:
                base_score *= 2.0
            elif release.language_priority == 3:
                base_score *= 1.5
            elif release.language_priority <= 6:
                base_score *= 1.2
            
            if release.is_today:
                base_score += 50
            elif release.is_yesterday:
                base_score += 25
            elif release.days_since_release <= 3:
                base_score += 15
            
            release.final_score = base_score

    def _sort_releases(self, releases: List[ReleaseItem]) -> List[ReleaseItem]:
        def sort_key(release):
            return (
                not release.is_today,
                not release.is_yesterday,
                release.days_since_release,
                release.language_priority,
                -release.final_score,
                -release.popularity
            )
        
        return sorted(releases, key=sort_key)

    async def _write_output(self, releases: List[ReleaseItem]):
        output_data = {
            'last_updated': datetime.now(pytz.timezone(self.timezone)).isoformat(),
            'total_count': len(releases),
            'telugu_count': len([r for r in releases if r.is_telugu]),
            'today_count': len([r for r in releases if r.is_today]),
            'yesterday_count': len([r for r in releases if r.is_yesterday]),
            'this_week_count': len([r for r in releases if r.is_this_week]),
            'metadata': {
                'timezone': self.timezone,
                'max_age_days': self.max_age_days,
                'refresh_interval_seconds': self.refresh_interval,
                'language_priority': ['Telugu', 'English', 'Hindi', 'Malayalam', 'Kannada', 'Tamil']
            },
            'releases': [asdict(release) for release in releases]
        }
        
        temp_file = f"{self.output_file}.tmp"
        
        try:
            async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(output_data, default=str, ensure_ascii=False, separators=(',', ':')))
            
            if os.path.exists(self.output_file):
                os.remove(self.output_file)
            os.rename(temp_file, self.output_file)
            
            logger.info(f"Successfully wrote {len(releases)} releases to {self.output_file}")
            
        except Exception as e:
            logger.error(f"Failed to write output file: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise

    async def health_check(self) -> Dict[str, Any]:
        return {
            'status': 'running' if self.running else 'stopped',
            'last_success': self.last_success_time.isoformat() if self.last_success_time else None,
            'error_count': self.error_count,
            'total_processed': self.total_processed,
            'output_file_exists': os.path.exists(self.output_file),
            'output_file_size': os.path.getsize(self.output_file) if os.path.exists(self.output_file) else 0
        }

async def main():
    service = ContinuousNewReleasesService()
    await service.start()

if __name__ == "__main__":
    asyncio.run(main())