# backend/services/persons.py
import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from sqlalchemy import and_, or_, func, desc, text
from sqlalchemy.orm import Session
from flask import current_app, has_app_context
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

# API Configuration
TMDB_API_KEY = None
OMDB_API_KEY = None
YOUTUBE_API_KEY = None
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
OMDB_BASE_URL = 'http://www.omdbapi.com/'
YOUTUBE_BASE_URL = 'https://www.googleapis.com/youtube/v3'

# Image URLs
PROFILE_BASE = 'https://image.tmdb.org/t/p/w500'
PROFILE_LARGE = 'https://image.tmdb.org/t/p/w780'
POSTER_BASE = 'https://image.tmdb.org/t/p/w500'

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
    complete_filmography: Dict
    detailed_filmography: Dict
    career_timeline: List[Dict]
    awards_and_honors: List[Dict]
    personal_life: Dict
    trivia_and_facts: List[str]
    quotes: List[Dict]
    images: List[str]
    social_media: Dict
    external_links: Dict
    collaborations: Dict
    total_works: int
    career_highlights: Dict
    statistics: Dict
    upcoming_projects: List[Dict]
    recent_news: List[Dict]
    
    def to_dict(self):
        return asdict(self)

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

def get_empty_filmography():
    """Return empty filmography structure"""
    return {
        'as_actor': [],
        'as_director': [],
        'as_writer': [],
        'as_producer': [],
        'as_crew': [],
        'upcoming': [],
        'by_year': defaultdict(list),
        'by_decade': defaultdict(list),
        'by_genre': defaultdict(list),
        'collaborations': defaultdict(list),
        'statistics': {
            'total_projects': 0,
            'years_active': 0,
            'debut_year': None,
            'latest_year': None,
            'highest_rated': None,
            'most_popular': None,
            'most_successful': None,
            'genre_breakdown': {},
            'department_breakdown': {},
            'average_rating': 0,
            'total_box_office': 0
        }
    }

class PersonsService:
    
    def __init__(self, db, models, cache=None):
        self.db = db
        self.Content = models.get('Content')
        self.User = models.get('User')
        self.Person = models.get('Person')
        self.ContentPerson = models.get('ContentPerson')
        self.cache = cache
        self.models = models
        
        self.session = self._create_http_session()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self._init_api_keys()
    
    def _init_api_keys(self):
        global TMDB_API_KEY, OMDB_API_KEY, YOUTUBE_API_KEY
        
        try:
            if has_app_context() and current_app:
                TMDB_API_KEY = current_app.config.get('TMDB_API_KEY')
                OMDB_API_KEY = current_app.config.get('OMDB_API_KEY')
                YOUTUBE_API_KEY = current_app.config.get('YOUTUBE_API_KEY')
        except Exception as e:
            logger.warning(f"Could not initialize API keys: {e}")
    
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
    
    def get_person_details_by_slug(self, slug: str) -> Optional[Dict]:
        """Get comprehensive person details by slug"""
        try:
            if not has_app_context():
                logger.warning("No app context available for get_person_details_by_slug")
                return None
            
            # Check cache first
            cache_key = f"person:slug:{slug}"
            if self.cache:
                try:
                    cached = self.cache.get(cache_key)
                    if cached:
                        logger.info(f"Cache hit for person slug: {slug}")
                        return cached
                except Exception as e:
                    logger.warning(f"Cache error for person slug {slug}: {e}")
            
            # Find person in database
            person = self.Person.query.filter_by(slug=slug).first()
            
            if not person:
                person = self._find_person_fuzzy(slug)
                if not person:
                    logger.warning(f"Person not found for slug: {slug}")
                    return None
            
            # Build comprehensive person details
            details = self._build_comprehensive_person_details(person)
            
            # Cache the result
            if self.cache and details:
                try:
                    self.cache.set(cache_key, details, timeout=3600)  # 1 hour cache
                except Exception as e:
                    logger.warning(f"Cache set error for person slug {slug}: {e}")
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting person details for slug {slug}: {e}")
            return None
    
    def _find_person_fuzzy(self, slug: str) -> Optional[Any]:
        """Fuzzy search for person when exact slug doesn't match"""
        try:
            # Extract name from slug
            name = slug.replace('-', ' ').replace('_', ' ').strip()
            
            # Try different name variations
            name_variations = [
                name.lower(),
                name.title(),
                name.upper(),
                name.replace(' ', ''),
                ' '.join(name.split()[:2]) if len(name.split()) > 2 else name  # First two words
            ]
            
            for variation in name_variations:
                results = self.Person.query.filter(
                    func.lower(self.Person.name).like(f"%{variation.lower()}%")
                ).limit(5).all()
                
                if results:
                    # Return best match based on similarity
                    best_match = max(results, key=lambda x: self._calculate_similarity(x.name.lower(), name.lower()))
                    logger.info(f"Found fuzzy match for person '{slug}': {best_match.name}")
                    return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fuzzy person search: {e}")
            return None
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, str1, str2).ratio()
        except Exception:
            return 0.0
    
    def _build_comprehensive_person_details(self, person: Any) -> Dict:
        """Build comprehensive person details with all available information"""
        try:
            self._init_api_keys()
            
            # Use ThreadPoolExecutor for concurrent API calls
            futures = {}
            
            with ThreadPoolExecutor(max_workers=6) as executor:
                # TMDB person details
                if person.tmdb_id and TMDB_API_KEY:
                    futures['tmdb_details'] = executor.submit(self._fetch_tmdb_person_details, person.tmdb_id)
                    futures['tmdb_credits'] = executor.submit(self._fetch_tmdb_person_credits, person.tmdb_id)
                    futures['tmdb_images'] = executor.submit(self._fetch_tmdb_person_images, person.tmdb_id)
                    futures['tmdb_external_ids'] = executor.submit(self._fetch_tmdb_external_ids, person.tmdb_id)
                
                # Database filmography - Ensure proper context handling
                if has_app_context():
                    futures['db_filmography'] = executor.submit(self._get_complete_database_filmography_safe, person.id)
                else:
                    logger.warning("No app context for database filmography")
                    futures['db_filmography'] = None
                
                # Additional data
                futures['awards'] = executor.submit(self._fetch_awards_and_honors, person)
                futures['trivia'] = executor.submit(self._fetch_trivia_and_facts, person)
                futures['news'] = executor.submit(self._fetch_recent_news, person)
            
            # Collect results with timeout handling
            results = {}
            for key, future in futures.items():
                try:
                    if future is not None:
                        results[key] = future.result(timeout=15)
                    else:
                        results[key] = None
                except Exception as e:
                    logger.warning(f"Error fetching {key}: {e}")
                    results[key] = None
            
            # Ensure we have filmography even if database call failed
            if results.get('db_filmography') is None:
                results['db_filmography'] = get_empty_filmography()
                logger.warning(f"Using empty filmography for person {person.name}")
            
            # Build comprehensive details
            details = self._compile_person_details(person, results)
            
            return details
            
        except Exception as e:
            logger.error(f"Error building comprehensive person details: {e}")
            return self._get_minimal_person_details(person)
    
    def _get_complete_database_filmography_safe(self, person_id: int) -> Dict:
        """Safe wrapper for filmography retrieval"""
        try:
            return self._get_complete_database_filmography(person_id)
        except Exception as e:
            logger.error(f"Error in safe filmography retrieval: {e}")
            return get_empty_filmography()
    
    def _fetch_tmdb_person_details(self, tmdb_id: int) -> Dict:
        """Fetch detailed person information from TMDB"""
        try:
            url = f"{TMDB_BASE_URL}/person/{tmdb_id}"
            params = {
                'api_key': TMDB_API_KEY,
                'append_to_response': 'external_ids,images,movie_credits,tv_credits,combined_credits'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"TMDB person details returned {response.status_code} for person {tmdb_id}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching TMDB person details: {e}")
            return {}
    
    def _fetch_tmdb_person_credits(self, tmdb_id: int) -> Dict:
        """Fetch complete filmography from TMDB"""
        try:
            url = f"{TMDB_BASE_URL}/person/{tmdb_id}/combined_credits"
            params = {
                'api_key': TMDB_API_KEY
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"TMDB person credits returned {response.status_code} for person {tmdb_id}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching TMDB person credits: {e}")
            return {}
    
    def _fetch_tmdb_person_images(self, tmdb_id: int) -> Dict:
        """Fetch person images from TMDB"""
        try:
            url = f"{TMDB_BASE_URL}/person/{tmdb_id}/images"
            params = {
                'api_key': TMDB_API_KEY
            }
            
            response = self.session.get(url, params=params, timeout=8)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"TMDB person images returned {response.status_code} for person {tmdb_id}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching TMDB person images: {e}")
            return {}
    
    def _fetch_tmdb_external_ids(self, tmdb_id: int) -> Dict:
        """Fetch external IDs and social media links"""
        try:
            url = f"{TMDB_BASE_URL}/person/{tmdb_id}/external_ids"
            params = {
                'api_key': TMDB_API_KEY
            }
            
            response = self.session.get(url, params=params, timeout=8)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"TMDB external IDs returned {response.status_code} for person {tmdb_id}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching TMDB external IDs: {e}")
            return {}
    
    @ensure_app_context
    def _get_complete_database_filmography(self, person_id: int) -> Dict:
        """Get complete filmography from database with detailed information"""
        try:
            filmography = get_empty_filmography()
            
            # Get all content-person relationships
            filmography_entries = self.db.session.query(
                self.ContentPerson, self.Content
            ).join(
                self.Content
            ).filter(
                self.ContentPerson.person_id == person_id
            ).order_by(
                self.Content.release_date.desc().nullslast()
            ).all()
            
            years = []
            all_ratings = []
            all_popularity = []
            genre_counts = Counter()
            dept_counts = Counter()
            total_revenue = 0
            
            for cp, content in filmography_entries:
                # Ensure content has slug
                if not content.slug:
                    try:
                        from services.details import SlugManager
                        SlugManager.update_content_slug(self.db, content)
                    except Exception:
                        content.slug = f"content-{content.id}"
                
                # Build work entry
                work = {
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'original_title': content.original_title,
                    'year': content.release_date.year if content.release_date else None,
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'content_type': content.content_type,
                    'poster_path': f"{POSTER_BASE}{content.poster_path}" if content.poster_path else None,
                    'rating': content.rating,
                    'popularity': content.popularity,
                    'vote_count': content.vote_count,
                    'character': cp.character,
                    'job': cp.job,
                    'department': cp.department,
                    'order': cp.order,
                    'role_type': cp.role_type,
                    'genres': [],
                    'overview': content.overview
                }
                
                # Parse genres
                try:
                    if content.genres:
                        work['genres'] = json.loads(content.genres)
                        for genre in work['genres']:
                            genre_counts[genre] += 1
                except (json.JSONDecodeError, TypeError):
                    pass
                
                # Collect statistics
                if content.release_date:
                    years.append(content.release_date.year)
                if content.rating:
                    all_ratings.append((content.rating, work))
                if content.popularity:
                    all_popularity.append((content.popularity, work))
                
                # Department breakdown
                if cp.department:
                    dept_counts[cp.department] += 1
                
                # Categorize by role
                year_key = work['year'] or 'Unknown'
                decade_key = f"{(work['year'] // 10) * 10}s" if work['year'] else 'Unknown'
                
                filmography['by_year'][year_key].append(work)
                filmography['by_decade'][decade_key].append(work)
                
                # Add to genre breakdown
                for genre in work['genres']:
                    filmography['by_genre'][genre].append(work)
                
                # Categorize by role type
                if content.release_date and content.release_date > datetime.now().date():
                    filmography['upcoming'].append(work)
                elif cp.role_type == 'cast':
                    filmography['as_actor'].append(work)
                elif cp.department == 'Directing' or cp.job == 'Director':
                    filmography['as_director'].append(work)
                elif cp.department == 'Writing' or cp.job in ['Writer', 'Screenplay', 'Story', 'Novel']:
                    filmography['as_writer'].append(work)
                elif 'Producer' in (cp.job or ''):
                    filmography['as_producer'].append(work)
                else:
                    filmography['as_crew'].append(work)
            
            # Calculate statistics
            filmography['statistics']['total_projects'] = len(filmography_entries)
            
            if years:
                years.sort()
                filmography['statistics']['debut_year'] = years[0]
                filmography['statistics']['latest_year'] = years[-1]
                filmography['statistics']['years_active'] = years[-1] - years[0] + 1
            
            if all_ratings:
                filmography['statistics']['highest_rated'] = max(all_ratings, key=lambda x: x[0])[1]
                avg_rating = sum(rating for rating, _ in all_ratings) / len(all_ratings)
                filmography['statistics']['average_rating'] = round(avg_rating, 2)
            
            if all_popularity:
                filmography['statistics']['most_popular'] = max(all_popularity, key=lambda x: x[0])[1]
            
            filmography['statistics']['genre_breakdown'] = dict(genre_counts.most_common(10))
            filmography['statistics']['department_breakdown'] = dict(dept_counts.most_common(10))
            
            logger.info(f"Retrieved complete filmography: {filmography['statistics']['total_projects']} projects")
            return filmography
            
        except Exception as e:
            logger.error(f"Error getting complete database filmography: {e}")
            return get_empty_filmography()
    
    def _fetch_awards_and_honors(self, person: Any) -> List[Dict]:
        """Fetch awards and honors information"""
        try:
            # This would typically connect to awards databases
            # For now, we'll return basic structure
            awards = []
            
            # Add Oscar nominations/wins if available in TMDB data
            # Add Emmy nominations/wins
            # Add Golden Globe nominations/wins
            # Add other major awards
            
            return awards
            
        except Exception as e:
            logger.error(f"Error fetching awards: {e}")
            return []
    
    def _fetch_trivia_and_facts(self, person: Any) -> List[str]:
        """Fetch trivia and interesting facts"""
        try:
            trivia = []
            
            # This would typically connect to trivia databases or scrape information
            # For now, we'll return empty list
            
            return trivia
            
        except Exception as e:
            logger.error(f"Error fetching trivia: {e}")
            return []
    
    def _fetch_recent_news(self, person: Any) -> List[Dict]:
        """Fetch recent news and articles"""
        try:
            news = []
            
            # This would typically connect to news APIs
            # For now, we'll return empty list
            
            return news
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def _compile_person_details(self, person: Any, api_results: Dict) -> Dict:
        """Compile all person details into comprehensive response"""
        try:
            # Get results with safe defaults
            tmdb_data = api_results.get('tmdb_details') or {}
            tmdb_credits = api_results.get('tmdb_credits') or {}
            tmdb_images = api_results.get('tmdb_images') or {}
            external_ids = api_results.get('tmdb_external_ids') or {}
            db_filmography = api_results.get('db_filmography') or get_empty_filmography()
            awards = api_results.get('awards') or []
            trivia = api_results.get('trivia') or []
            news = api_results.get('news') or []
            
            # Update person data from TMDB if available
            self._update_person_from_tmdb(person, tmdb_data)
            
            # Build social media links
            social_media = self._build_social_media_links(external_ids)
            
            # Build external links
            external_links = self._build_external_links(external_ids, person)
            
            # Process images
            images = self._process_person_images(tmdb_images, person.profile_path)
            
            # Build career timeline - with null check
            career_timeline = self._build_career_timeline(db_filmography)
            
            # Build collaborations
            collaborations = self._build_collaborations(db_filmography)
            
            # Build personal life information
            personal_life = self._build_personal_life_info(person, tmdb_data)
            
            # Build career highlights - with null check
            career_highlights = self._build_detailed_career_highlights(db_filmography, tmdb_data)
            
            # Process also_known_as
            also_known_as = []
            try:
                if person.also_known_as:
                    also_known_as = json.loads(person.also_known_as)
            except (json.JSONDecodeError, TypeError):
                pass
            
            if not also_known_as and tmdb_data.get('also_known_as'):
                also_known_as = tmdb_data['also_known_as']
            
            # Build comprehensive response
            details = {
                'id': person.id,
                'slug': person.slug,
                'tmdb_id': person.tmdb_id,
                'name': person.name,
                'biography': person.biography or tmdb_data.get('biography', ''),
                'birthday': person.birthday.isoformat() if person.birthday else tmdb_data.get('birthday'),
                'deathday': person.deathday.isoformat() if person.deathday else tmdb_data.get('deathday'),
                'place_of_birth': person.place_of_birth or tmdb_data.get('place_of_birth'),
                'profile_path': f"{PROFILE_BASE}{person.profile_path}" if person.profile_path else None,
                'popularity': person.popularity or tmdb_data.get('popularity', 0),
                'known_for_department': person.known_for_department or tmdb_data.get('known_for_department'),
                'also_known_as': also_known_as,
                'gender': person.gender,
                
                # Comprehensive filmography
                'filmography': db_filmography,
                'detailed_filmography': self._enhance_filmography_with_tmdb(db_filmography, tmdb_credits),
                
                # Career information
                'career_timeline': career_timeline,
                'career_highlights': career_highlights,
                'collaborations': collaborations,
                
                # Personal information
                'personal_life': personal_life,
                'awards_and_honors': awards,
                'trivia_and_facts': trivia,
                'quotes': [],  # Would be populated from external sources
                
                # Media and links
                'images': images,
                'social_media': social_media,
                'external_links': external_links,
                
                # Statistics
                'total_works': db_filmography.get('statistics', {}).get('total_projects', 0),
                'statistics': db_filmography.get('statistics', {}),
                
                # Additional content
                'upcoming_projects': db_filmography.get('upcoming', []),
                'recent_news': news,
                
                # Enhanced data
                'genre_expertise': self._calculate_genre_expertise(db_filmography),
                'decade_activity': self._calculate_decade_activity(db_filmography),
                'collaboration_network': self._build_collaboration_network(person.id),
                'influence_score': self._calculate_influence_score(db_filmography),
                'versatility_rating': self._calculate_versatility_rating(db_filmography)
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Error compiling person details: {e}")
            return self._get_minimal_person_details(person)
    
    def _update_person_from_tmdb(self, person: Any, tmdb_data: Dict):
        """Update person data from TMDB"""
        try:
            if not tmdb_data:
                return
            
            updated = False
            
            # Update biography
            if not person.biography and tmdb_data.get('biography'):
                person.biography = tmdb_data['biography']
                updated = True
            
            # Update birthday
            if not person.birthday and tmdb_data.get('birthday'):
                try:
                    person.birthday = datetime.strptime(tmdb_data['birthday'], '%Y-%m-%d').date()
                    updated = True
                except:
                    pass
            
            # Update deathday
            if not person.deathday and tmdb_data.get('deathday'):
                try:
                    person.deathday = datetime.strptime(tmdb_data['deathday'], '%Y-%m-%d').date()
                    updated = True
                except:
                    pass
            
            # Update place of birth
            if not person.place_of_birth and tmdb_data.get('place_of_birth'):
                person.place_of_birth = tmdb_data['place_of_birth']
                updated = True
            
            # Update also known as
            if not person.also_known_as and tmdb_data.get('also_known_as'):
                person.also_known_as = json.dumps(tmdb_data['also_known_as'])
                updated = True
            
            # Update popularity if higher
            if tmdb_data.get('popularity') and tmdb_data['popularity'] > (person.popularity or 0):
                person.popularity = tmdb_data['popularity']
                updated = True
            
            if updated and has_app_context():
                self.db.session.commit()
                logger.info(f"Updated person {person.name} with TMDB data")
                
        except Exception as e:
            logger.error(f"Error updating person from TMDB: {e}")
            if has_app_context():
                self.db.session.rollback()
    
    def _build_social_media_links(self, external_ids: Dict) -> Dict:
        """Build social media links from external IDs"""
        try:
            social_media = {}
            
            if external_ids:
                if external_ids.get('twitter_id'):
                    social_media['twitter'] = f"https://twitter.com/{external_ids['twitter_id']}"
                
                if external_ids.get('instagram_id'):
                    social_media['instagram'] = f"https://instagram.com/{external_ids['instagram_id']}"
                
                if external_ids.get('facebook_id'):
                    social_media['facebook'] = f"https://facebook.com/{external_ids['facebook_id']}"
                
                if external_ids.get('tiktok_id'):
                    social_media['tiktok'] = f"https://tiktok.com/@{external_ids['tiktok_id']}"
                
                if external_ids.get('youtube_id'):
                    social_media['youtube'] = f"https://youtube.com/channel/{external_ids['youtube_id']}"
            
            return social_media
            
        except Exception as e:
            logger.error(f"Error building social media links: {e}")
            return {}
    
    def _build_external_links(self, external_ids: Dict, person: Any) -> Dict:
        """Build external links"""
        try:
            external_links = {}
            
            if external_ids:
                if external_ids.get('imdb_id'):
                    external_links['imdb'] = f"https://www.imdb.com/name/{external_ids['imdb_id']}"
                
                if external_ids.get('wikidata_id'):
                    external_links['wikidata'] = f"https://www.wikidata.org/wiki/{external_ids['wikidata_id']}"
            
            # Add TMDB link
            if person.tmdb_id:
                external_links['tmdb'] = f"https://www.themoviedb.org/person/{person.tmdb_id}"
            
            return external_links
            
        except Exception as e:
            logger.error(f"Error building external links: {e}")
            return {}
    
    def _process_person_images(self, tmdb_images: Dict, profile_path: str) -> List[str]:
        """Process person images"""
        try:
            images = []
            
            # Add main profile image
            if profile_path:
                images.append(f"{PROFILE_BASE}{profile_path}")
                images.append(f"{PROFILE_LARGE}{profile_path}")
            
            # Add additional images from TMDB
            if tmdb_images and tmdb_images.get('profiles'):
                for img in tmdb_images['profiles'][:20]:
                    image_url = f"{PROFILE_BASE}{img['file_path']}"
                    if image_url not in images:
                        images.append(image_url)
            
            return images
            
        except Exception as e:
            logger.error(f"Error processing person images: {e}")
            return []
    
    def _build_career_timeline(self, filmography: Dict) -> List[Dict]:
        """Build detailed career timeline - with null checks"""
        try:
            timeline = []
            
            if not filmography or not isinstance(filmography, dict):
                logger.warning("Invalid filmography data for timeline building")
                return timeline
            
            # Get all works sorted by year
            all_works = []
            for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer', 'as_crew']:
                works = filmography.get(role_type, [])
                if works and isinstance(works, list):
                    for work in works:
                        if work and isinstance(work, dict) and work.get('year'):
                            all_works.append(work)
            
            # Sort by year
            all_works.sort(key=lambda x: x.get('year', 0))
            
            # Group by year and create timeline entries
            current_year = None
            year_works = []
            
            for work in all_works:
                work_year = work.get('year')
                
                if current_year != work_year:
                    if year_works:
                        timeline.append({
                            'year': current_year,
                            'works': year_works,
                            'milestone': self._identify_career_milestone(current_year, year_works)
                        })
                    
                    current_year = work_year
                    year_works = [work]
                else:
                    year_works.append(work)
            
            # Add final year
            if year_works:
                timeline.append({
                    'year': current_year,
                    'works': year_works,
                    'milestone': self._identify_career_milestone(current_year, year_works)
                })
            
            return timeline
            
        except Exception as e:
            logger.error(f"Error building career timeline: {e}")
            return []
    
    def _identify_career_milestone(self, year: int, works: List[Dict]) -> Optional[str]:
        """Identify if a year represents a career milestone"""
        try:
            if not works or not isinstance(works, list):
                return None
            
            # Check for debut
            # Check for breakthrough roles
            # Check for awards
            # Check for career changes
            
            if len(works) >= 5:
                return f"Prolific year with {len(works)} projects"
            
            highest_rated = None
            try:
                rated_works = [w for w in works if w and isinstance(w, dict) and w.get('rating')]
                if rated_works:
                    highest_rated = max(rated_works, key=lambda x: x.get('rating', 0))
            except Exception:
                pass
            
            if highest_rated and highest_rated.get('rating', 0) >= 8.0:
                return f"Critical success with {highest_rated.get('title', 'Unknown')}"
            
            return None
            
        except Exception as e:
            logger.error(f"Error identifying career milestone: {e}")
            return None
    
    def _build_collaborations(self, filmography: Dict) -> Dict:
        """Build collaboration information - with null checks"""
        try:
            collaborations = {
                'frequent_directors': {},
                'frequent_actors': {},
                'frequent_writers': {},
                'frequent_producers': {},
                'studio_relationships': {},
                'genre_specialists': {}
            }
            
            if not filmography or not isinstance(filmography, dict):
                return collaborations
            
            # This would require additional database queries to find collaborators
            # For now, return empty structure
            
            return collaborations
            
        except Exception as e:
            logger.error(f"Error building collaborations: {e}")
            return {
                'frequent_directors': {},
                'frequent_actors': {},
                'frequent_writers': {},
                'frequent_producers': {},
                'studio_relationships': {},
                'genre_specialists': {}
            }
    
    def _build_personal_life_info(self, person: Any, tmdb_data: Dict) -> Dict:
        """Build personal life information"""
        try:
            personal_info = {
                'age': None,
                'zodiac_sign': None,
                'nationality': None,
                'education': [],
                'family': {},
                'relationships': [],
                'children': [],
                'residence': [],
                'interests': [],
                'philanthropy': []
            }
            
            # Calculate age
            if person.birthday:
                today = datetime.now().date()
                if person.deathday:
                    end_date = person.deathday
                else:
                    end_date = today
                
                personal_info['age'] = end_date.year - person.birthday.year
                if end_date.month < person.birthday.month or (end_date.month == person.birthday.month and end_date.day < person.birthday.day):
                    personal_info['age'] -= 1
                
                # Calculate zodiac sign
                personal_info['zodiac_sign'] = self._get_zodiac_sign(person.birthday)
            
            # Extract nationality from place of birth
            if person.place_of_birth:
                place_parts = person.place_of_birth.split(',')
                if len(place_parts) > 0:
                    personal_info['nationality'] = place_parts[-1].strip()
            
            return personal_info
            
        except Exception as e:
            logger.error(f"Error building personal life info: {e}")
            return {
                'age': None,
                'zodiac_sign': None,
                'nationality': None,
                'education': [],
                'family': {},
                'relationships': [],
                'children': [],
                'residence': [],
                'interests': [],
                'philanthropy': []
            }
    
    def _get_zodiac_sign(self, birthday) -> str:
        """Get zodiac sign from birthday"""
        try:
            month = birthday.month
            day = birthday.day
            
            if (month == 3 and day >= 21) or (month == 4 and day <= 19):
                return "Aries"
            elif (month == 4 and day >= 20) or (month == 5 and day <= 20):
                return "Taurus"
            elif (month == 5 and day >= 21) or (month == 6 and day <= 20):
                return "Gemini"
            elif (month == 6 and day >= 21) or (month == 7 and day <= 22):
                return "Cancer"
            elif (month == 7 and day >= 23) or (month == 8 and day <= 22):
                return "Leo"
            elif (month == 8 and day >= 23) or (month == 9 and day <= 22):
                return "Virgo"
            elif (month == 9 and day >= 23) or (month == 10 and day <= 22):
                return "Libra"
            elif (month == 10 and day >= 23) or (month == 11 and day <= 21):
                return "Scorpio"
            elif (month == 11 and day >= 22) or (month == 12 and day <= 21):
                return "Sagittarius"
            elif (month == 12 and day >= 22) or (month == 1 and day <= 19):
                return "Capricorn"
            elif (month == 1 and day >= 20) or (month == 2 and day <= 18):
                return "Aquarius"
            else:
                return "Pisces"
                
        except Exception:
            return "Unknown"
    
    def _build_detailed_career_highlights(self, filmography: Dict, tmdb_data: Dict) -> Dict:
        """Build detailed career highlights - with null checks"""
        try:
            highlights = {
                'debut_work': None,
                'breakthrough_role': None,
                'signature_works': [],
                'career_defining_moments': [],
                'box_office_successes': [],
                'critical_darlings': [],
                'awards_season_contenders': [],
                'genre_innovations': [],
                'directorial_debut': None,
                'producing_debut': None,
                'writing_debut': None,
                'international_recognition': [],
                'collaborations_of_note': [],
                'career_transformations': []
            }
            
            if not filmography or not isinstance(filmography, dict):
                logger.warning("Invalid filmography data for career highlights")
                return highlights
            
            stats = filmography.get('statistics', {})
            if not isinstance(stats, dict):
                return highlights
            
            # Find debut work
            debut_year = stats.get('debut_year')
            if debut_year:
                by_year = filmography.get('by_year', {})
                if isinstance(by_year, dict):
                    debut_year_works = by_year.get(str(debut_year), [])
                    if debut_year_works and isinstance(debut_year_works, list):
                        highlights['debut_work'] = debut_year_works[0]
            
            # Find highest rated work
            highest_rated = stats.get('highest_rated')
            if highest_rated and isinstance(highest_rated, dict):
                highlights['signature_works'].append(highest_rated)
            
            # Find most popular work
            most_popular = stats.get('most_popular')
            if most_popular and isinstance(most_popular, dict):
                highlights['box_office_successes'].append(most_popular)
            
            # Find works with high ratings
            for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer']:
                works = filmography.get(role_type, [])
                if works and isinstance(works, list):
                    critical_works = []
                    for work in works:
                        if work and isinstance(work, dict) and work.get('rating', 0) >= 8.0:
                            critical_works.append(work)
                    highlights['critical_darlings'].extend(critical_works[:3])
            
            return highlights
            
        except Exception as e:
            logger.error(f"Error building detailed career highlights: {e}")
            return {
                'debut_work': None,
                'breakthrough_role': None,
                'signature_works': [],
                'career_defining_moments': [],
                'box_office_successes': [],
                'critical_darlings': [],
                'awards_season_contenders': [],
                'genre_innovations': [],
                'directorial_debut': None,
                'producing_debut': None,
                'writing_debut': None,
                'international_recognition': [],
                'collaborations_of_note': [],
                'career_transformations': []
            }
    
    def _enhance_filmography_with_tmdb(self, db_filmography: Dict, tmdb_credits: Dict) -> Dict:
        """Enhance database filmography with additional TMDB data - with null checks"""
        try:
            if not db_filmography or not isinstance(db_filmography, dict):
                logger.warning("Invalid db_filmography for enhancement")
                return get_empty_filmography()
            
            enhanced = db_filmography.copy()
            
            if not tmdb_credits or not isinstance(tmdb_credits, dict):
                return enhanced
            
            # Process cast credits
            tmdb_cast = tmdb_credits.get('cast', [])
            tmdb_crew = tmdb_credits.get('crew', [])
            
            if not isinstance(tmdb_cast, list):
                tmdb_cast = []
            if not isinstance(tmdb_crew, list):
                tmdb_crew = []
            
            # Create lookup for TMDB data
            tmdb_lookup = {}
            
            for credit in tmdb_cast + tmdb_crew:
                if credit and isinstance(credit, dict):
                    tmdb_id = credit.get('id')
                    if tmdb_id:
                        tmdb_lookup[tmdb_id] = credit
            
            # Enhance existing works with TMDB data
            for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer', 'as_crew']:
                enhanced_works = []
                works = enhanced.get(role_type, [])
                if works and isinstance(works, list):
                    for work in works:
                        if work and isinstance(work, dict):
                            # Try to match with TMDB data
                            enhanced_work = work.copy()
                            
                            # Add additional metadata from TMDB if available
                            # This would require matching logic based on title and year
                            
                            enhanced_works.append(enhanced_work)
                
                enhanced[role_type] = enhanced_works
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing filmography with TMDB: {e}")
            return db_filmography if isinstance(db_filmography, dict) else get_empty_filmography()
    
    def _calculate_genre_expertise(self, filmography: Dict) -> Dict:
        """Calculate genre expertise and preferences - with null checks"""
        try:
            genre_stats = {}
            
            if not filmography or not isinstance(filmography, dict):
                return genre_stats
            
            stats = filmography.get('statistics', {})
            if not isinstance(stats, dict):
                return genre_stats
            
            genre_breakdown = stats.get('genre_breakdown', {})
            if not isinstance(genre_breakdown, dict):
                return genre_stats
            
            total_works = sum(genre_breakdown.values()) if genre_breakdown else 0
            
            for genre, count in genre_breakdown.items():
                percentage = (count / total_works * 100) if total_works > 0 else 0
                
                genre_stats[genre] = {
                    'count': count,
                    'percentage': round(percentage, 1),
                    'expertise_level': self._determine_expertise_level(count, percentage)
                }
            
            return genre_stats
            
        except Exception as e:
            logger.error(f"Error calculating genre expertise: {e}")
            return {}
    
    def _determine_expertise_level(self, count: int, percentage: float) -> str:
        """Determine expertise level based on count and percentage"""
        if percentage >= 30:
            return "Specialist"
        elif percentage >= 15:
            return "Frequent"
        elif count >= 5:
            return "Experienced"
        elif count >= 2:
            return "Familiar"
        else:
            return "Occasional"
    
    def _calculate_decade_activity(self, filmography: Dict) -> Dict:
        """Calculate activity by decade - with null checks"""
        try:
            decade_activity = {}
            
            if not filmography or not isinstance(filmography, dict):
                return decade_activity
            
            by_decade = filmography.get('by_decade', {})
            if not isinstance(by_decade, dict):
                return decade_activity
            
            for decade, works in by_decade.items():
                if decade != 'Unknown' and works and isinstance(works, list):
                    decade_activity[decade] = {
                        'count': len(works),
                        'notable_works': [w for w in works if w and isinstance(w, dict) and w.get('rating', 0) >= 7.5][:3],
                        'avg_rating': self._calculate_average_rating(works),
                        'most_popular': max(works, key=lambda x: x.get('popularity', 0) if x and isinstance(x, dict) else 0) if works else None
                    }
            
            return decade_activity
            
        except Exception as e:
            logger.error(f"Error calculating decade activity: {e}")
            return {}
    
    def _calculate_average_rating(self, works: List[Dict]) -> float:
        """Calculate average rating for a list of works - with null checks"""
        try:
            if not works or not isinstance(works, list):
                return 0
            
            ratings = []
            for work in works:
                if work and isinstance(work, dict) and work.get('rating'):
                    ratings.append(work['rating'])
            
            return round(sum(ratings) / len(ratings), 2) if ratings else 0
        except Exception:
            return 0
    
    def _build_collaboration_network(self, person_id: int) -> Dict:
        """Build collaboration network"""
        try:
            # This would require complex queries to find frequent collaborators
            # For now, return empty structure
            return {
                'frequent_collaborators': [],
                'collaboration_strength': {},
                'network_connections': []
            }
            
        except Exception as e:
            logger.error(f"Error building collaboration network: {e}")
            return {
                'frequent_collaborators': [],
                'collaboration_strength': {},
                'network_connections': []
            }
    
    def _calculate_influence_score(self, filmography: Dict) -> float:
        """Calculate influence score based on various factors - with null checks"""
        try:
            if not filmography or not isinstance(filmography, dict):
                return 0.0
            
            stats = filmography.get('statistics', {})
            if not isinstance(stats, dict):
                return 0.0
            
            factors = {
                'total_projects': min(stats.get('total_projects', 0) / 50, 1.0) * 0.3,
                'average_rating': min(stats.get('average_rating', 0) / 10, 1.0) * 0.4,
                'years_active': min(stats.get('years_active', 0) / 40, 1.0) * 0.2,
                'popularity': min((stats.get('most_popular', {}).get('popularity', 0) or 0) / 100, 1.0) * 0.1
            }
            
            influence_score = sum(factors.values()) * 100
            return round(influence_score, 1)
            
        except Exception as e:
            logger.error(f"Error calculating influence score: {e}")
            return 0.0
    
    def _calculate_versatility_rating(self, filmography: Dict) -> float:
        """Calculate versatility rating based on genre and role diversity - with null checks"""
        try:
            if not filmography or not isinstance(filmography, dict):
                return 0.0
            
            stats = filmography.get('statistics', {})
            if not isinstance(stats, dict):
                return 0.0
            
            # Genre diversity
            genre_breakdown = stats.get('genre_breakdown', {})
            genre_count = len(genre_breakdown) if isinstance(genre_breakdown, dict) else 0
            genre_score = min(genre_count / 10, 1.0) * 0.6
            
            # Role diversity
            role_types = ['as_actor', 'as_director', 'as_writer', 'as_producer']
            active_roles = 0
            for role in role_types:
                role_works = filmography.get(role, [])
                if role_works and isinstance(role_works, list) and len(role_works) > 0:
                    active_roles += 1
            
            role_score = (active_roles / len(role_types)) * 0.4
            
            versatility = (genre_score + role_score) * 100
            return round(versatility, 1)
            
        except Exception as e:
            logger.error(f"Error calculating versatility rating: {e}")
            return 0.0
    
    def _get_minimal_person_details(self, person: Any) -> Dict:
        """Get minimal person details as fallback"""
        try:
            return {
                'id': person.id,
                'slug': person.slug,
                'name': person.name,
                'biography': person.biography or '',
                'birthday': person.birthday.isoformat() if person.birthday else None,
                'deathday': person.deathday.isoformat() if person.deathday else None,
                'place_of_birth': person.place_of_birth,
                'profile_path': f"{PROFILE_BASE}{person.profile_path}" if person.profile_path else None,
                'popularity': person.popularity or 0,
                'known_for_department': person.known_for_department,
                'also_known_as': [],
                'gender': person.gender,
                'filmography': get_empty_filmography(),
                'detailed_filmography': get_empty_filmography(),
                'career_timeline': [],
                'career_highlights': {},
                'collaborations': {},
                'personal_life': {},
                'awards_and_honors': [],
                'trivia_and_facts': [],
                'quotes': [],
                'images': [],
                'social_media': {},
                'external_links': {},
                'total_works': 0,
                'statistics': {},
                'upcoming_projects': [],
                'recent_news': [],
                'genre_expertise': {},
                'decade_activity': {},
                'collaboration_network': {},
                'influence_score': 0.0,
                'versatility_rating': 0.0
            }
        except Exception as e:
            logger.error(f"Error creating minimal person details: {e}")
            return {'error': 'Failed to get person details'}

    def search_persons(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for persons"""
        try:
            if not query or len(query) < 2:
                return []
            
            if not has_app_context():
                logger.warning("No app context for person search")
                return []
            
            # Search in database
            persons = self.Person.query.filter(
                func.lower(self.Person.name).like(f"%{query.lower()}%")
            ).order_by(
                self.Person.popularity.desc()
            ).limit(limit).all()
            
            results = []
            for person in persons:
                if not person.slug:
                    try:
                        from services.details import SlugManager
                        SlugManager.update_content_slug(self.db, person)
                    except Exception:
                        person.slug = f"person-{person.id}"
                
                results.append({
                    'id': person.id,
                    'slug': person.slug,
                    'name': person.name,
                    'known_for_department': person.known_for_department,
                    'profile_path': f"{PROFILE_BASE}{person.profile_path}" if person.profile_path else None,
                    'popularity': person.popularity or 0
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching persons: {e}")
            return []

def init_persons_service(app, db, models, cache):
    """Initialize the persons service"""
    with app.app_context():
        service = PersonsService(db, models, cache)
        return service