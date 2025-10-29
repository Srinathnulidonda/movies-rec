# backend/services/persons_details.py
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import func, desc
from flask import current_app, has_app_context
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from dotenv import load_dotenv
from difflib import SequenceMatcher
from collections import defaultdict, Counter

load_dotenv()

logger = logging.getLogger(__name__)

TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
PROFILE_BASE = 'https://image.tmdb.org/t/p/w185'
POSTER_BASE = 'https://image.tmdb.org/t/p/w500'
BACKDROP_BASE = 'https://image.tmdb.org/t/p/w1280'
STILL_BASE = 'https://image.tmdb.org/t/p/w780'

if not TMDB_API_KEY:
    logger.warning("TMDB_API_KEY not found - person details may be limited")

def ensure_app_context(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if has_app_context():
            return func(*args, **kwargs)
        else:
            try:
                if current_app:
                    with current_app.app_context():
                        return func(*args, **kwargs)
                else:
                    logger.warning(f"No app context available for {func.__name__}")
                    return None
            except Exception as e:
                logger.error(f"Error in app context wrapper for {func.__name__}: {e}")
                return None
    return wrapper

@dataclass
class PersonDetails:
    slug: str
    id: int
    name: str
    biography: str
    birthday: Optional[str]
    deathday: Optional[str]
    place_of_birth: Optional[str]
    profile_url: Optional[str]
    popularity: float
    known_for_department: Optional[str]
    also_known_as: List[str]
    gender: Optional[int]
    filmography: Dict
    images: List[str]
    social_media: Dict
    total_works: int
    awards: List[Dict]
    personal_info: Dict
    career_highlights: Dict
    
    def to_dict(self):
        return asdict(self)

class SlugManager:
    
    @staticmethod
    def normalize_title(title: str) -> str:
        if not title or not isinstance(title, str):
            return ""
        
        try:
            clean_title = str(title).strip()
            if not clean_title:
                return ""
            
            import unicodedata
            normalized = unicodedata.normalize('NFKD', clean_title)
            normalized = normalized.encode('ascii', 'ignore').decode('ascii')
            
            import re
            normalized = re.sub(r'[^\w\s\-\']', '', normalized)
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing title '{title}': {e}")
            return str(title).strip()
    
    @staticmethod
    def generate_slug(title: str, year: Optional[int] = None, content_type: str = 'person', 
                     original_title: str = None, tmdb_id: int = None) -> str:
        try:
            if not title or not isinstance(title, str):
                fallback = f"person-{tmdb_id or int(time.time())}"
                logger.warning(f"Invalid title provided, using fallback: {fallback}")
                return fallback
            
            normalized_title = SlugManager.normalize_title(title)
            
            if not normalized_title:
                fallback = f"person-{tmdb_id or int(time.time())}"
                logger.warning(f"Title normalization failed, using fallback: {fallback}")
                return fallback
            
            try:
                from slugify import slugify
                slug = slugify(normalized_title, max_length=70, word_boundary=True, save_order=True)
            except Exception as slugify_error:
                logger.warning(f"Slugify failed for '{normalized_title}': {slugify_error}")
                slug = SlugManager._manual_slugify(normalized_title)
            
            if not slug or len(slug) < 2:
                slug = SlugManager._manual_slugify(normalized_title)
            
            if not slug:
                return f"person-{tmdb_id or int(time.time())}"
            
            if len(slug) > 120:
                parts = slug[:117].split('-')
                if len(parts) > 1:
                    slug = '-'.join(parts[:-1])
                else:
                    slug = slug[:117]
            
            return slug
            
        except Exception as e:
            logger.error(f"Critical error generating slug for title '{title}': {e}")
            return f"person-{tmdb_id or int(time.time())}"
    
    @staticmethod
    def _manual_slugify(text: str) -> str:
        try:
            import re
            slug = text.lower()
            slug = re.sub(r'[^\w\s-]', '', slug)
            slug = re.sub(r'[-\s]+', '-', slug)
            slug = slug.strip('-')
            return slug[:70] if slug else ""
        except Exception:
            return ""
    
    @staticmethod
    def generate_unique_slug(db, model, title: str, year: Optional[int] = None, 
                           content_type: str = 'person', existing_id: Optional[int] = None,
                           original_title: str = None, tmdb_id: int = None) -> str:
        try:
            base_slug = SlugManager.generate_slug(title, year, content_type, original_title, tmdb_id)
            
            if not base_slug:
                base_slug = f"person-{tmdb_id or int(time.time())}"
            
            slug = base_slug
            counter = 1
            max_attempts = 100
            
            while counter <= max_attempts:
                try:
                    query = db.session.query(model.id).filter_by(slug=slug)
                    
                    if existing_id:
                        query = query.filter(model.id != existing_id)
                    
                    exists = query.first() is not None
                    
                    if not exists:
                        break
                    
                    slug = f"{base_slug}-{counter}"
                    counter += 1
                    
                except Exception as e:
                    logger.error(f"Error checking slug uniqueness: {e}")
                    slug = f"{base_slug}-{int(time.time())}"
                    break
            
            if counter > max_attempts:
                timestamp_slug = f"{base_slug}-{int(time.time())}"
                logger.warning(f"Hit max attempts for slug generation, using timestamp: {timestamp_slug}")
                return timestamp_slug
            
            return slug
            
        except Exception as e:
            logger.error(f"Critical error generating unique slug: {e}")
            return f"person-{tmdb_id or int(time.time())}-{abs(hash(str(title)))[:6]}"
    
    @staticmethod
    def extract_info_from_slug(slug: str) -> Dict:
        try:
            if not slug:
                return {'title': 'Unknown', 'year': None, 'content_type': 'person'}
            
            content_type = 'person'
            clean_slug = slug
            
            if slug.startswith('person-'):
                clean_slug = slug[7:]  # Remove 'person-' prefix
            
            title = SlugManager._slug_to_title(clean_slug)
            
            return {
                'title': title,
                'year': None,
                'content_type': content_type
            }
            
        except Exception as e:
            logger.error(f"Error extracting info from slug '{slug}': {e}")
            return {
                'title': slug.replace('-', ' ').title() if slug else 'Unknown',
                'year': None,
                'content_type': 'person'
            }
    
    @staticmethod
    def _slug_to_title(slug: str) -> str:
        try:
            title = slug.replace('-', ' ').title()
            
            title_fixes = {
                'Dc': 'DC',
                'Mcu': 'MCU',
                'Jr': 'Jr.',
                'Sr': 'Sr.',
                'Iii': 'III',
                'Ii': 'II',
                'Iv': 'IV'
            }
            
            import re
            for wrong, correct in title_fixes.items():
                title = re.sub(f'\\b{wrong}\\b', correct, title)
            
            return title
            
        except Exception as e:
            logger.error(f"Error converting slug to title: {e}")
            return slug.replace('-', ' ').title()
    
    @staticmethod
    def update_content_slug(db, content, force_update: bool = False) -> str:
        try:
            if content.slug and not force_update:
                return content.slug
            
            name = getattr(content, 'name', '') or getattr(content, 'title', '')
            tmdb_id = getattr(content, 'tmdb_id', None)
            
            if not name:
                name = f"Person {getattr(content, 'id', 'Unknown')}"
            
            new_slug = SlugManager.generate_unique_slug(
                db, 
                content.__class__, 
                name, 
                content_type='person',
                existing_id=getattr(content, 'id', None),
                tmdb_id=tmdb_id
            )
            
            content.slug = new_slug
            
            return new_slug
            
        except Exception as e:
            logger.error(f"Error updating person slug: {e}")
            fallback = f"person-{getattr(content, 'id', int(time.time()))}"
            content.slug = fallback
            return fallback

class PersonsDetailsService:
    
    def __init__(self, db, models, cache=None):
        self.db = db
        self.Content = models.get('Content')
        self.User = models.get('User')
        self.Person = models.get('Person')
        self.ContentPerson = models.get('ContentPerson')
        self.cache = cache
        self.models = models
        
        self.session = self._create_http_session()
        self.executor = ThreadPoolExecutor(max_workers=3)
    
    def _create_http_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            read=3,
            connect=3,
            backoff_factor=0.3,
            status_forcelist=(500, 502, 504)
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    @ensure_app_context
    def get_person_details(self, person_slug: str) -> Optional[Dict]:
        try:
            cache_key = f"person:details:{person_slug}"
            if self.cache:
                try:
                    cached = self.cache.get(cache_key)
                    if cached:
                        logger.info(f"Cache hit for person slug: {person_slug}")
                        return cached
                except Exception as e:
                    logger.warning(f"Cache error for person slug {person_slug}: {e}")
            
            person = self.Person.query.filter_by(slug=person_slug).first()
            
            if not person:
                person = self._find_person_fuzzy(person_slug)
                
                if not person:
                    logger.warning(f"Person not found for slug: {person_slug}")
                    return None
            
            if not person.slug or person.slug.startswith('person-'):
                try:
                    SlugManager.update_content_slug(self.db, person, force_update=True)
                    self.db.session.commit()
                except Exception as e:
                    logger.warning(f"Failed to update person slug: {e}")
            
            tmdb_data = {}
            if person.tmdb_id and TMDB_API_KEY:
                logger.info(f"Fetching comprehensive TMDB data for {person.name} (ID: {person.tmdb_id})")
                tmdb_data = self._fetch_complete_person_details(person.tmdb_id)
                
                self._update_person_from_tmdb(person, tmdb_data)
                
                if tmdb_data:
                    self._fetch_and_save_person_credits(person, tmdb_data)
            
            filmography = self._get_enhanced_filmography(person.id, tmdb_data)
            
            person_details = self._build_comprehensive_person_details(person, tmdb_data, filmography)
            
            if self.cache and person_details:
                try:
                    self.cache.set(cache_key, person_details, timeout=3600)
                except Exception as e:
                    logger.warning(f"Cache set error for person slug {person_slug}: {e}")
            
            return person_details
            
        except Exception as e:
            logger.error(f"Error getting person details for slug {person_slug}: {e}")
            logger.exception(e)
            return None

    def _fetch_complete_person_details(self, tmdb_id: int) -> Dict:
        try:
            if not TMDB_API_KEY:
                logger.warning("TMDB_API_KEY not available")
                return {}
            
            url = f"{TMDB_BASE_URL}/person/{tmdb_id}"
            params = {
                'api_key': TMDB_API_KEY,
                'append_to_response': 'images,external_ids,combined_credits,movie_credits,tv_credits,tagged_images'
            }
            
            response = self.session.get(url, params=params, timeout=12)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched TMDB data for person {tmdb_id}, includes external_ids: {bool(data.get('external_ids'))}")
                return data
            else:
                logger.warning(f"TMDB person API returned {response.status_code} for person {tmdb_id}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching complete person details from TMDB: {e}")
            return {}
    
    def _fetch_and_save_person_credits(self, person: Any, tmdb_data: Dict):
        try:
            if not tmdb_data:
                return
            
            movie_credits = tmdb_data.get('movie_credits', {})
            self._process_person_credits(person, movie_credits.get('cast', []), 'cast', 'movie')
            self._process_person_credits(person, movie_credits.get('crew', []), 'crew', 'movie')
            
            tv_credits = tmdb_data.get('tv_credits', {})
            self._process_person_credits(person, tv_credits.get('cast', []), 'cast', 'tv')
            self._process_person_credits(person, tv_credits.get('crew', []), 'crew', 'tv')
            
            combined_credits = tmdb_data.get('combined_credits', {})
            if combined_credits:
                self._process_person_credits(person, combined_credits.get('cast', []), 'cast', None)
                self._process_person_credits(person, combined_credits.get('crew', []), 'crew', None)
            
            self.db.session.commit()
            logger.info(f"Successfully saved credits for {person.name}")
            
        except Exception as e:
            logger.error(f"Error fetching/saving person credits: {e}")
            self.db.session.rollback()

    def _process_person_credits(self, person: Any, credits: List[Dict], role_type: str, content_type: Optional[str]):
        try:
            for credit in credits:
                try:
                    content = self._get_or_create_content_from_credit(credit, content_type)
                    if not content:
                        continue
                    
                    if role_type == 'cast':
                        self._get_or_create_content_person(
                            content.id, 
                            person.id, 
                            'cast',
                            character=credit.get('character'),
                            order=credit.get('order', 999)
                        )
                    else:
                        self._get_or_create_content_person(
                            content.id,
                            person.id,
                            'crew',
                            job=credit.get('job'),
                            department=credit.get('department')
                        )
                        
                except Exception as e:
                    logger.warning(f"Error processing credit: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing person credits: {e}")

    def _get_or_create_content_from_credit(self, credit: Dict, content_type: Optional[str]) -> Any:
        try:
            tmdb_id = credit.get('id')
            if not tmdb_id:
                return None
            
            existing = self.Content.query.filter_by(tmdb_id=tmdb_id).first()
            if existing:
                return existing
            
            if not content_type:
                media_type = credit.get('media_type', 'movie')
                content_type = 'tv' if media_type == 'tv' else 'movie'
            
            title = credit.get('title') or credit.get('name') or 'Unknown Title'
            original_title = credit.get('original_title') or credit.get('original_name')
            
            release_date = None
            year = None
            date_str = credit.get('release_date') or credit.get('first_air_date')
            if date_str:
                try:
                    release_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    year = release_date.year
                except:
                    pass
            
            # Use a simple slug generation for credits
            from slugify import slugify
            slug = slugify(title)
            if year:
                slug = f"{slug}-{year}"
            
            # Make slug unique
            base_slug = slug
            counter = 1
            while self.Content.query.filter_by(slug=slug).first():
                slug = f"{base_slug}-{counter}"
                counter += 1
            
            content = self.Content(
                slug=slug,
                tmdb_id=tmdb_id,
                title=title,
                original_title=original_title,
                content_type=content_type,
                release_date=release_date,
                rating=credit.get('vote_average'),
                vote_count=credit.get('vote_count'),
                popularity=credit.get('popularity'),
                overview=credit.get('overview'),
                poster_path=credit.get('poster_path'),
                backdrop_path=credit.get('backdrop_path')
            )
            
            self.db.session.add(content)
            self.db.session.flush()
            
            return content
            
        except Exception as e:
            logger.error(f"Error creating content from credit: {e}")
            return None

    def _get_or_create_content_person(self, content_id, person_id, role_type, 
                                     character=None, job=None, department=None, order=None):
        try:
            existing = None
            
            if role_type == 'cast' and character:
                existing = self.ContentPerson.query.filter_by(
                    content_id=content_id,
                    person_id=person_id,
                    role_type=role_type,
                    character=character
                ).first()
            elif role_type == 'crew' and job:
                existing = self.ContentPerson.query.filter_by(
                    content_id=content_id,
                    person_id=person_id,
                    role_type=role_type,
                    job=job
                ).first()
            else:
                existing = self.ContentPerson.query.filter_by(
                    content_id=content_id,
                    person_id=person_id,
                    role_type=role_type
                ).first()
            
            if not existing:
                content_person = self.ContentPerson(
                    content_id=content_id,
                    person_id=person_id,
                    role_type=role_type,
                    character=character,
                    job=job,
                    department=department,
                    order=order
                )
                
                self.db.session.add(content_person)
                return content_person
            
            return existing
            
        except Exception as e:
            logger.error(f"Error creating content-person relationship: {e}")
            return None

    def _get_enhanced_filmography(self, person_id: int, tmdb_data: Dict) -> Dict:
        try:
            filmography = {
                'as_actor': [],
                'as_director': [],
                'as_writer': [],
                'as_producer': [],
                'as_other_crew': [],
                'upcoming': [],
                'by_year': {},
                'by_decade': {},
                'statistics': {
                    'total_projects': 0,
                    'years_active': 0,
                    'debut_year': None,
                    'latest_year': None,
                    'highest_rated': None,
                    'most_popular': None,
                    'average_rating': 0,
                    'total_votes': 0
                }
            }
            
            filmography_entries = self.db.session.query(
                self.ContentPerson, self.Content
            ).join(
                self.Content
            ).filter(
                self.ContentPerson.person_id == person_id
            ).order_by(
                self.Content.release_date.desc().nullslast()
            ).all()
            
            upcoming_from_tmdb = []
            if tmdb_data:
                combined_credits = tmdb_data.get('combined_credits', {})
                for credit_type in ['cast', 'crew']:
                    for credit in combined_credits.get(credit_type, []):
                        release_date_str = credit.get('release_date') or credit.get('first_air_date')
                        if release_date_str:
                            try:
                                release_date = datetime.strptime(release_date_str, '%Y-%m-%d').date()
                                if release_date >= datetime.now().date():
                                    upcoming_from_tmdb.append((credit, credit_type))
                            except:
                                pass
            
            years = []
            all_ratings = []
            all_popularity = []
            total_votes = 0
            current_date = datetime.now().date()
            
            for cp, content in filmography_entries:
                try:
                    if not content.slug or content.slug.startswith('content-'):
                        try:
                            SlugManager.update_content_slug(self.db, content, force_update=True)
                        except Exception:
                            content.slug = f"content-{content.id}"
                    
                    year = None
                    is_upcoming = False
                    if content.release_date:
                        try:
                            year = int(content.release_date.year)
                            years.append(year)
                            if content.release_date >= current_date:
                                is_upcoming = True
                        except (AttributeError, TypeError, ValueError):
                            pass
                    
                    rating = 0
                    try:
                        if content.rating and isinstance(content.rating, (int, float)):
                            rating = float(content.rating)
                            if rating > 0:
                                all_ratings.append(rating)
                    except (TypeError, ValueError):
                        pass
                    
                    popularity = 0
                    try:
                        if content.popularity and isinstance(content.popularity, (int, float)):
                            popularity = float(content.popularity)
                            if popularity > 0:
                                all_popularity.append(popularity)
                    except (TypeError, ValueError):
                        pass
                    
                    try:
                        if content.vote_count and isinstance(content.vote_count, (int, float)):
                            total_votes += int(content.vote_count)
                    except (TypeError, ValueError):
                        pass
                    
                    work = {
                        'id': content.id,
                        'slug': content.slug,
                        'title': content.title,
                        'original_title': content.original_title,
                        'year': year,
                        'content_type': content.content_type,
                        'poster_path': self._format_image_url(content.poster_path, 'poster'),
                        'backdrop_path': self._format_image_url(content.backdrop_path, 'backdrop'),
                        'rating': rating,
                        'popularity': popularity,
                        'vote_count': content.vote_count or 0,
                        'character': cp.character,
                        'job': cp.job,
                        'department': cp.department,
                        'role_type': cp.role_type,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'overview': content.overview or '',
                        'status': 'Upcoming' if is_upcoming else 'Released'
                    }
                    
                    if is_upcoming:
                        filmography['upcoming'].append(work)
                    
                    year_key = str(year) if year else 'Unknown'
                    if year_key not in filmography['by_year']:
                        filmography['by_year'][year_key] = []
                    filmography['by_year'][year_key].append(work)
                    
                    if year:
                        decade = (year // 10) * 10
                        decade_key = f"{decade}s"
                        if decade_key not in filmography['by_decade']:
                            filmography['by_decade'][decade_key] = []
                        filmography['by_decade'][decade_key].append(work)
                    
                    if cp.role_type == 'cast':
                        filmography['as_actor'].append(work)
                    elif cp.role_type == 'crew':
                        job = (cp.job or '').lower()
                        department = (cp.department or '').lower()
                        
                        if 'director' in job or department == 'directing':
                            filmography['as_director'].append(work)
                        elif 'writer' in job or 'screenplay' in job or department == 'writing':
                            filmography['as_writer'].append(work)
                        elif 'producer' in job or department == 'production':
                            filmography['as_producer'].append(work)
                        else:
                            filmography['as_other_crew'].append(work)
                            
                except Exception as e:
                    logger.warning(f"Error processing filmography entry: {e}")
                    continue
            
            for credit, credit_type in upcoming_from_tmdb:
                try:
                    tmdb_id = credit.get('id')
                    already_exists = any(
                        work.get('tmdb_id') == tmdb_id for work in filmography['upcoming']
                    )
                    
                    if not already_exists and tmdb_id:
                        release_date_str = credit.get('release_date') or credit.get('first_air_date')
                        release_date = None
                        year = None
                        
                        if release_date_str:
                            try:
                                release_date = datetime.strptime(release_date_str, '%Y-%m-%d').date()
                                year = release_date.year
                            except:
                                pass
                        
                        upcoming_work = {
                            'id': None,
                            'tmdb_id': tmdb_id,
                            'slug': None,
                            'title': credit.get('title') or credit.get('name', 'Unknown Title'),
                            'original_title': credit.get('original_title') or credit.get('original_name'),
                            'year': year,
                            'content_type': 'tv' if credit.get('media_type') == 'tv' else 'movie',
                            'poster_path': self._format_image_url(credit.get('poster_path'), 'poster'),
                            'backdrop_path': self._format_image_url(credit.get('backdrop_path'), 'backdrop'),
                            'rating': credit.get('vote_average', 0),
                            'popularity': credit.get('popularity', 0),
                            'vote_count': credit.get('vote_count', 0),
                            'character': credit.get('character') if credit_type == 'cast' else None,
                            'job': credit.get('job') if credit_type == 'crew' else None,
                            'department': credit.get('department') if credit_type == 'crew' else None,
                            'role_type': credit_type,
                            'release_date': release_date.isoformat() if release_date else None,
                            'overview': credit.get('overview', ''),
                            'status': 'Upcoming'
                        }
                        
                        filmography['upcoming'].append(upcoming_work)
                        
                except Exception as e:
                    logger.warning(f"Error processing upcoming TMDB credit: {e}")
                    continue
            
            filmography['upcoming'].sort(key=lambda x: x.get('release_date') or '9999-12-31')
            
            try:
                filmography['statistics']['total_projects'] = len(filmography_entries)
                
                if years:
                    valid_years = [y for y in years if isinstance(y, int) and 1900 <= y <= 2030]
                    if valid_years:
                        filmography['statistics']['debut_year'] = min(valid_years)
                        filmography['statistics']['latest_year'] = max(valid_years)
                        filmography['statistics']['years_active'] = max(valid_years) - min(valid_years) + 1
                
                if all_ratings:
                    filmography['statistics']['average_rating'] = round(sum(all_ratings) / len(all_ratings), 1)
                    
                    highest_rated_score = max(all_ratings)
                    for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer', 'as_other_crew']:
                        for work in filmography[role_type]:
                            if work['rating'] == highest_rated_score:
                                filmography['statistics']['highest_rated'] = work
                                break
                        if filmography['statistics']['highest_rated']:
                            break
                
                if all_popularity:
                    highest_popularity = max(all_popularity)
                    for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer', 'as_other_crew']:
                        for work in filmography[role_type]:
                            if work['popularity'] == highest_popularity:
                                filmography['statistics']['most_popular'] = work
                                break
                        if filmography['statistics']['most_popular']:
                            break
                
                filmography['statistics']['total_votes'] = total_votes
                
            except Exception as e:
                logger.warning(f"Error calculating filmography statistics: {e}")
            
            total_upcoming = len(filmography['upcoming'])
            logger.info(f"Retrieved enhanced filmography: {filmography['statistics']['total_projects']} projects, {total_upcoming} upcoming")
            return filmography
            
        except Exception as e:
            logger.error(f"Error getting enhanced filmography: {e}")
            return {
                'as_actor': [], 'as_director': [], 'as_writer': [], 'as_producer': [], 'as_other_crew': [],
                'upcoming': [], 'by_year': {}, 'by_decade': {}, 'statistics': {}
            }

    def _build_comprehensive_person_details(self, person: Any, tmdb_data: Dict, filmography: Dict) -> Dict:
        try:
            also_known_as = []
            try:
                if person.also_known_as:
                    if isinstance(person.also_known_as, str):
                        also_known_as = json.loads(person.also_known_as)
                    elif isinstance(person.also_known_as, list):
                        also_known_as = person.also_known_as
            except (json.JSONDecodeError, TypeError):
                pass
            
            if not also_known_as and tmdb_data.get('also_known_as'):
                also_known_as = tmdb_data['also_known_as']
            
            personal_info = self._build_enhanced_personal_info(person, tmdb_data, filmography)
            career_highlights = self._build_enhanced_career_highlights(filmography)
            images = self._get_person_images(tmdb_data)
            social_media = self._get_person_social_media(tmdb_data)
            
            external_ids = tmdb_data.get('external_ids', {})
            
            person_details = {
                'id': person.id,
                'slug': person.slug,
                'tmdb_id': person.tmdb_id,
                'name': person.name,
                'biography': person.biography or tmdb_data.get('biography', ''),
                'birthday': person.birthday.isoformat() if person.birthday else tmdb_data.get('birthday'),
                'deathday': person.deathday.isoformat() if person.deathday else tmdb_data.get('deathday'),
                'place_of_birth': person.place_of_birth or tmdb_data.get('place_of_birth'),
                'profile_path': self._format_image_url(person.profile_path, 'profile'),
                'popularity': person.popularity or tmdb_data.get('popularity', 0),
                'known_for_department': person.known_for_department or tmdb_data.get('known_for_department'),
                'also_known_as': also_known_as,
                'gender': person.gender,
                'filmography': filmography,
                'images': images,
                'social_media': social_media,
                'total_works': self._calculate_total_works(filmography),
                'personal_info': personal_info,
                'career_highlights': career_highlights,
                'external_ids': external_ids,
                'known_for': self._get_known_for_works(filmography),
                'awards_recognition': self._build_awards_info(tmdb_data, filmography),
                'trivia': self._extract_trivia(person, tmdb_data),
                'collaborations': self._analyze_collaborations(filmography),
                'upcoming_projects_count': len(filmography.get('upcoming', [])),
                'social_media_count': len(social_media)
            }
            
            return person_details
            
        except Exception as e:
            logger.error(f"Error building comprehensive person details: {e}")
            return self._get_minimal_person_details(person)

    def _build_enhanced_personal_info(self, person: Any, tmdb_data: Dict, filmography: Dict) -> Dict:
        try:
            personal_info = {
                'age': None,
                'age_at_death': None,
                'zodiac_sign': None,
                'nationality': None,
                'birth_name': None,
                'height': None,
                'career_span': None,
                'status': 'Active',
                'family': {},
                'education': []
            }
            
            if person.birthday:
                try:
                    birth_date = person.birthday
                    if person.deathday:
                        end_date = person.deathday
                        personal_info['status'] = 'Deceased'
                        personal_info['age_at_death'] = end_date.year - birth_date.year
                        if end_date.month < birth_date.month or (end_date.month == birth_date.month and end_date.day < birth_date.day):
                            personal_info['age_at_death'] -= 1
                    else:
                        today = datetime.now().date()
                        personal_info['age'] = today.year - birth_date.year
                        if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
                            personal_info['age'] -= 1
                    
                    personal_info['zodiac_sign'] = self._get_zodiac_sign(birth_date.month, birth_date.day)
                    
                except Exception as e:
                    logger.warning(f"Error calculating age: {e}")
            
            if person.place_of_birth:
                try:
                    place_parts = person.place_of_birth.split(',')
                    if len(place_parts) > 0:
                        country = place_parts[-1].strip()
                        personal_info['nationality'] = country
                except Exception:
                    pass
            
            stats = filmography.get('statistics', {})
            if stats.get('debut_year') and stats.get('latest_year'):
                debut = stats['debut_year']
                latest = stats['latest_year']
                if isinstance(debut, int) and isinstance(latest, int):
                    personal_info['career_span'] = f"{debut} - {latest if latest != datetime.now().year else 'Present'}"
            
            return personal_info
            
        except Exception as e:
            logger.error(f"Error building enhanced personal info: {e}")
            return {}

    def _get_zodiac_sign(self, month: int, day: int) -> str:
        try:
            zodiac_signs = [
                (120, "Capricorn"), (218, "Aquarius"), (320, "Pisces"), (420, "Aries"),
                (521, "Taurus"), (621, "Gemini"), (722, "Cancer"), (823, "Leo"),
                (923, "Virgo"), (1023, "Libra"), (1122, "Scorpio"), (1222, "Sagittarius"), (1231, "Capricorn")
            ]
            
            date_number = month * 100 + day
            for date_limit, sign in zodiac_signs:
                if date_number <= date_limit:
                    return sign
            return "Capricorn"
        except:
            return ""

    def _build_enhanced_career_highlights(self, filmography: Dict) -> Dict:
        try:
            highlights = {
                'debut_work': None,
                'breakthrough_role': None,
                'latest_work': None,
                'most_successful_decade': None,
                'genre_expertise': {},
                'collaboration_frequency': {},
                'career_phases': [],
                'notable_achievements': []
            }
            
            stats = filmography.get('statistics', {})
            
            if stats.get('debut_year'):
                debut_year = stats['debut_year']
                for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer']:
                    for work in filmography.get(role_type, []):
                        if work.get('year') == debut_year:
                            highlights['debut_work'] = work
                            break
                    if highlights['debut_work']:
                        break
            
            if stats.get('latest_year'):
                latest_year = stats['latest_year']
                for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer']:
                    for work in filmography.get(role_type, []):
                        if work.get('year') == latest_year:
                            highlights['latest_work'] = work
                            break
                    if highlights['latest_work']:
                        break
            
            decade_stats = {}
            for decade, works in filmography.get('by_decade', {}).items():
                decade_stats[decade] = {
                    'count': len(works),
                    'avg_rating': 0,
                    'total_popularity': 0
                }
                
                ratings = [w['rating'] for w in works if w.get('rating', 0) > 0]
                if ratings:
                    decade_stats[decade]['avg_rating'] = sum(ratings) / len(ratings)
                
                popularity = [w['popularity'] for w in works if w.get('popularity', 0) > 0]
                if popularity:
                    decade_stats[decade]['total_popularity'] = sum(popularity)
            
            if decade_stats:
                best_decade = max(decade_stats.items(), 
                                key=lambda x: x[1]['count'] * x[1]['avg_rating'])
                highlights['most_successful_decade'] = {
                    'decade': best_decade[0],
                    'projects': best_decade[1]['count'],
                    'avg_rating': round(best_decade[1]['avg_rating'], 1)
                }
            
            return highlights
            
        except Exception as e:
            logger.error(f"Error building enhanced career highlights: {e}")
            return {}

    def _get_known_for_works(self, filmography: Dict) -> List[Dict]:
        try:
            all_works = []
            
            for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer']:
                for work in filmography.get(role_type, []):
                    if work.get('rating', 0) > 0 or work.get('popularity', 0) > 0:
                        rating_score = work.get('rating', 0) * 10
                        popularity_score = min(work.get('popularity', 0), 100)
                        combined_score = (rating_score + popularity_score) / 2
                        
                        work_copy = work.copy()
                        work_copy['combined_score'] = combined_score
                        work_copy['role_description'] = work.get('character') or work.get('job') or 'Unknown Role'
                        all_works.append(work_copy)
            
            all_works.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            return all_works[:10]
            
        except Exception as e:
            logger.error(f"Error getting known for works: {e}")
            return []

    def _build_awards_info(self, tmdb_data: Dict, filmography: Dict) -> Dict:
        try:
            awards_info = {
                'major_awards': [],
                'nominations': [],
                'recognition': [],
                'honors': []
            }
            
            stats = filmography.get('statistics', {})
            
            if stats.get('highest_rated'):
                highest = stats['highest_rated']
                if highest.get('rating', 0) >= 8.0:
                    awards_info['recognition'].append({
                        'type': 'Critical Acclaim',
                        'description': f"Highly rated performance in '{highest.get('title', 'Unknown')}'",
                        'rating': highest.get('rating'),
                        'year': highest.get('year')
                    })
            
            if stats.get('total_projects', 0) >= 50:
                awards_info['honors'].append({
                    'type': 'Prolific Career',
                    'description': f"Over {stats['total_projects']} film and television credits"
                })
            
            return awards_info
            
        except Exception as e:
            logger.error(f"Error building awards info: {e}")
            return {}

    def _extract_trivia(self, person: Any, tmdb_data: Dict) -> List[str]:
        try:
            trivia = []
            
            if person.birthday and person.place_of_birth:
                try:
                    age = datetime.now().year - person.birthday.year
                    trivia.append(f"Born in {person.place_of_birth}")
                    if age > 0:
                        trivia.append(f"Currently {age} years old")
                except:
                    pass
            
            if person.known_for_department:
                trivia.append(f"Primarily known for work in {person.known_for_department}")
            
            return trivia
            
        except Exception as e:
            logger.error(f"Error extracting trivia: {e}")
            return []

    def _analyze_collaborations(self, filmography: Dict) -> Dict:
        try:
            return {
                'frequent_co_stars': [],
                'frequent_directors': [],
                'frequent_writers': [],
                'production_companies': []
            }
            
        except Exception as e:
            logger.error(f"Error analyzing collaborations: {e}")
            return {}

    def _get_minimal_person_details(self, person: Any) -> Dict:
        try:
            return {
                'id': person.id,
                'slug': person.slug or f"person-{person.id}",
                'name': person.name,
                'biography': person.biography or '',
                'profile_path': self._format_image_url(person.profile_path, 'profile'),
                'popularity': person.popularity or 0,
                'known_for_department': person.known_for_department,
                'filmography': {
                    'as_actor': [], 'as_director': [], 'as_writer': [], 'as_producer': [],
                    'statistics': {'total_projects': 0}
                },
                'error': 'Detailed information temporarily unavailable'
            }
        except Exception as e:
            logger.error(f"Error creating minimal person details: {e}")
            return {'error': 'Person details unavailable'}

    def _update_person_from_tmdb(self, person: Any, tmdb_data: Dict):
        try:
            if not tmdb_data:
                return
            
            updated = False
            
            if not person.biography and tmdb_data.get('biography'):
                person.biography = tmdb_data['biography']
                updated = True
            
            if not person.birthday and tmdb_data.get('birthday'):
                try:
                    person.birthday = datetime.strptime(tmdb_data['birthday'], '%Y-%m-%d').date()
                    updated = True
                except:
                    pass
            
            if not person.deathday and tmdb_data.get('deathday'):
                try:
                    person.deathday = datetime.strptime(tmdb_data['deathday'], '%Y-%m-%d').date()
                    updated = True
                except:
                    pass
            
            if not person.place_of_birth and tmdb_data.get('place_of_birth'):
                person.place_of_birth = tmdb_data['place_of_birth']
                updated = True
            
            if not person.also_known_as and tmdb_data.get('also_known_as'):
                person.also_known_as = json.dumps(tmdb_data['also_known_as'])
                updated = True
            
            if tmdb_data.get('popularity') and tmdb_data['popularity'] > (person.popularity or 0):
                person.popularity = tmdb_data['popularity']
                updated = True
            
            if updated:
                self.db.session.commit()
                logger.info(f"Updated person {person.name} with TMDB data")
                
        except Exception as e:
            logger.error(f"Error updating person from TMDB: {e}")
            self.db.session.rollback()

    def _find_person_fuzzy(self, slug: str) -> Optional[Any]:
        try:
            info = SlugManager.extract_info_from_slug(slug)
            name = info['title']
            
            # Try exact name match first
            exact_match = self.Person.query.filter(
                func.lower(self.Person.name) == name.lower()
            ).first()
            
            if exact_match:
                return exact_match
            
            # Try partial matches
            results = self.Person.query.filter(
                func.lower(self.Person.name).like(f"%{name.lower()}%")
            ).limit(10).all()
            
            if results:
                best_match = max(results, key=lambda x: self._calculate_similarity(x.name, name))
                logger.info(f"Found fuzzy match for person '{slug}': {best_match.name}")
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fuzzy person search: {e}")
            return None

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        try:
            clean_str1 = str1.lower().strip()
            clean_str2 = str2.lower().strip()
            
            if clean_str1 == clean_str2:
                return 1.0
            
            basic_sim = SequenceMatcher(None, clean_str1, clean_str2).ratio()
            
            words1 = set(clean_str1.split())
            words2 = set(clean_str2.split())
            
            if words1 and words2:
                word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                return 0.6 * basic_sim + 0.4 * word_overlap
            
            return basic_sim
            
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0

    def _get_person_images(self, tmdb_data: Dict) -> List[Dict]:
        try:
            images = []
            if tmdb_data.get('images', {}).get('profiles'):
                for img in tmdb_data['images']['profiles'][:20]:
                    images.append({
                        'url': self._format_image_url(img['file_path'], 'profile'),
                        'width': img.get('width'),
                        'height': img.get('height'),
                        'aspect_ratio': img.get('aspect_ratio')
                    })
            return images
        except Exception as e:
            logger.error(f"Error getting person images: {e}")
            return []

    def _get_person_social_media(self, tmdb_data: Dict) -> Dict:
        try:
            social = {}
            
            logger.info(f"Processing social media from TMDB data: {bool(tmdb_data.get('external_ids'))}")
            
            if tmdb_data.get('external_ids'):
                ids = tmdb_data['external_ids']
                logger.info(f"Available external IDs: {list(ids.keys())}")
                
                if ids.get('twitter_id'):
                    social['twitter'] = {
                        'url': f"https://twitter.com/{ids['twitter_id']}",
                        'handle': f"@{ids['twitter_id']}",
                        'platform': 'Twitter'
                    }
                
                if ids.get('instagram_id'):
                    social['instagram'] = {
                        'url': f"https://instagram.com/{ids['instagram_id']}",
                        'handle': f"@{ids['instagram_id']}",
                        'platform': 'Instagram'
                    }
                
                if ids.get('facebook_id'):
                    social['facebook'] = {
                        'url': f"https://facebook.com/{ids['facebook_id']}",
                        'handle': ids['facebook_id'],
                        'platform': 'Facebook'
                    }
                
                if ids.get('imdb_id'):
                    social['imdb'] = {
                        'url': f"https://www.imdb.com/name/{ids['imdb_id']}",
                        'handle': ids['imdb_id'],
                        'platform': 'IMDb'
                    }
                
                if ids.get('tiktok_id'):
                    social['tiktok'] = {
                        'url': f"https://tiktok.com/@{ids['tiktok_id']}",
                        'handle': f"@{ids['tiktok_id']}",
                        'platform': 'TikTok'
                    }
                
                if ids.get('youtube_id'):
                    social['youtube'] = {
                        'url': f"https://youtube.com/channel/{ids['youtube_id']}",
                        'handle': ids['youtube_id'],
                        'platform': 'YouTube'
                    }
                
                if ids.get('wikidata_id'):
                    social['wikidata'] = {
                        'url': f"https://www.wikidata.org/wiki/{ids['wikidata_id']}",
                        'handle': ids['wikidata_id'],
                        'platform': 'Wikidata'
                    }
                
                if ids.get('twitter_id'):
                    social['x'] = {
                        'url': f"https://x.com/{ids['twitter_id']}",
                        'handle': f"@{ids['twitter_id']}",
                        'platform': 'X (Twitter)'
                    }
            
            logger.info(f"Found social media links: {list(social.keys())}")
            return social
            
        except Exception as e:
            logger.error(f"Error getting person social media: {e}")
            return {}

    def _calculate_total_works(self, filmography: Dict) -> int:
        try:
            total = 0
            for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer']:
                total += len(filmography.get(role_type, []))
            return total
        except:
            return 0

    def _format_image_url(self, path: str, image_type: str = 'poster') -> Optional[str]:
        if not path:
            return None
        
        if path.startswith('http'):
            return path
        
        try:
            if image_type == 'poster':
                return f"{POSTER_BASE}{path}"
            elif image_type == 'backdrop':
                return f"{BACKDROP_BASE}{path}"
            elif image_type == 'profile':
                return f"{PROFILE_BASE}{path}"
            elif image_type == 'still':
                return f"{STILL_BASE}{path}"
            elif image_type == 'thumbnail':
                return f"https://image.tmdb.org/t/p/w185{path}"
            elif image_type == 'logo':
                return f"https://image.tmdb.org/t/p/w92{path}"
            else:
                return f"{POSTER_BASE}{path}"
        except Exception as e:
            logger.error(f"Error formatting image URL: {e}")
            return None

    def search_persons(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for persons by name"""
        try:
            if not query or len(query.strip()) < 2:
                return []
            
            query_lower = query.lower().strip()
            
            # Search in database first
            results = self.Person.query.filter(
                func.lower(self.Person.name).like(f"%{query_lower}%")
            ).order_by(
                self.Person.popularity.desc().nullslast()
            ).limit(limit).all()
            
            formatted_results = []
            for person in results:
                if not person.slug:
                    try:
                        SlugManager.update_content_slug(self.db, person)
                        self.db.session.commit()
                    except Exception:
                        person.slug = f"person-{person.id}"
                
                formatted_results.append({
                    'id': person.id,
                    'slug': person.slug,
                    'name': person.name,
                    'profile_path': self._format_image_url(person.profile_path, 'profile'),
                    'popularity': person.popularity or 0,
                    'known_for_department': person.known_for_department
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching persons: {e}")
            return []

    def get_person_by_id(self, person_id: int) -> Optional[Dict]:
        """Get person details by ID"""
        try:
            person = self.Person.query.get(person_id)
            if not person:
                return None
            
            if not person.slug:
                try:
                    SlugManager.update_content_slug(self.db, person)
                    self.db.session.commit()
                except Exception:
                    person.slug = f"person-{person.id}"
            
            return self.get_person_details(person.slug)
            
        except Exception as e:
            logger.error(f"Error getting person by ID: {e}")
            return None

    def update_person_slug(self, person_id: int, force_update: bool = False) -> Optional[str]:
        """Update person slug"""
        try:
            person = self.Person.query.get(person_id)
            if person:
                new_slug = SlugManager.update_content_slug(self.db, person, force_update)
                self.db.session.commit()
                
                if self.cache:
                    try:
                        cache_key = f"person:details:{new_slug}"
                        self.cache.delete(cache_key)
                    except Exception as e:
                        logger.warning(f"Cache invalidation error: {e}")
                
                return new_slug
            return None
        except Exception as e:
            logger.error(f"Error updating person slug: {e}")
            self.db.session.rollback()
            return None

    def clear_cache_for_person(self, slug: str) -> bool:
        """Clear cache for person"""
        try:
            if not self.cache:
                return False
            
            cache_key = f"person:details:{slug}"
            self.cache.delete(cache_key)
            logger.info(f"Cleared cache for person slug: {slug}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache for person slug {slug}: {e}")
            return False

def init_persons_details_service(app, db, models, cache):
    try:
        with app.app_context():
            service = PersonsDetailsService(db, models, cache)
            logger.info("CineBrain Persons Details Service initialized successfully")
            return service
    except Exception as e:
        logger.error(f"Failed to initialize CineBrain Persons Details Service: {e}")
        raise e