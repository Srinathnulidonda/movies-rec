# backend/services/details.py
"""
Details Service Module - Production-Ready with Slug-Based Routing
Handles all details page logic including content, cast, reviews, and recommendations
"""

import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from functools import wraps
from slugify import slugify
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from sqlalchemy import and_, or_, func, desc, text
from sqlalchemy.orm import Session
from flask import current_app
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Configuration
TMDB_API_KEY = None
OMDB_API_KEY = None
YOUTUBE_API_KEY = None
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
OMDB_BASE_URL = 'http://www.omdbapi.com/'
YOUTUBE_BASE_URL = 'https://www.googleapis.com/youtube/v3'
JIKAN_BASE_URL = 'https://api.jikan.moe/v4'

# Image bases
POSTER_BASE = 'https://image.tmdb.org/t/p/w500'
BACKDROP_BASE = 'https://image.tmdb.org/t/p/w1280'
PROFILE_BASE = 'https://image.tmdb.org/t/p/w185'
STILL_BASE = 'https://image.tmdb.org/t/p/w780'

@dataclass
class ContentDetails:
    """Data class for content details"""
    slug: str
    id: int
    title: str
    original_title: Optional[str]
    overview: str
    content_type: str
    poster_url: Optional[str]
    backdrop_url: Optional[str]
    trailer: Optional[Dict]
    synopsis: Dict
    cast_crew: Dict
    ratings: Dict
    metadata: Dict
    more_like_this: List[Dict]
    reviews: List[Dict]
    gallery: Dict
    streaming_info: Optional[Dict]
    seasons_episodes: Optional[Dict]
    
    def to_dict(self):
        return asdict(self)

class SlugManager:
    """Manages slug generation and lookup"""
    
    @staticmethod
    def generate_slug(title: str, year: Optional[int] = None, content_type: str = 'movie') -> str:
        """
        Generate URL-safe slug from title
        Examples:
        - "The Dark Knight" (2008) → "the-dark-knight-2008"
        - "Attack on Titan" → "attack-on-titan"
        """
        if not title:
            return ""
        
        # Clean and slugify title
        slug = slugify(title, lowercase=True, separator='-')
        
        # Add year if available (common practice for movies)
        if year and content_type == 'movie':
            slug = f"{slug}-{year}"
        
        # Ensure slug is not too long
        if len(slug) > 100:
            slug = slug[:100].rsplit('-', 1)[0]
        
        return slug
    
    @staticmethod
    def generate_unique_slug(db, model, title: str, year: Optional[int] = None, 
                           content_type: str = 'movie') -> str:
        """Generate unique slug, adding suffix if necessary"""
        base_slug = SlugManager.generate_slug(title, year, content_type)
        slug = base_slug
        counter = 1
        
        while db.session.query(model).filter_by(slug=slug).first():
            slug = f"{base_slug}-{counter}"
            counter += 1
        
        return slug
    
    @staticmethod
    def extract_info_from_slug(slug: str) -> Dict:
        """Extract potential title and year from slug"""
        # Check if slug ends with a year (4 digits)
        match = re.match(r'^(.+)-(\d{4})$', slug)
        if match:
            return {
                'title': match.group(1).replace('-', ' ').title(),
                'year': int(match.group(2))
            }
        return {
            'title': slug.replace('-', ' ').title(),
            'year': None
        }

class DetailsService:
    """Main service for handling content details"""
    
    def __init__(self, db, models, cache=None):
        self.db = db
        self.Content = models.get('Content')
        self.User = models.get('User')
        self.UserInteraction = models.get('UserInteraction')
        self.Review = models.get('Review')
        self.Person = models.get('Person')
        self.ContentPerson = models.get('ContentPerson')
        self.cache = cache
        
        # Setup HTTP session with retry
        self.session = self._create_http_session()
        
        # Thread pool for concurrent API calls
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialize API keys from environment or config
        self._init_api_keys()
    
    def _init_api_keys(self):
        """Initialize API keys from app config"""
        global TMDB_API_KEY, OMDB_API_KEY, YOUTUBE_API_KEY
        
        if current_app:
            TMDB_API_KEY = current_app.config.get('TMDB_API_KEY')
            OMDB_API_KEY = current_app.config.get('OMDB_API_KEY')
            YOUTUBE_API_KEY = current_app.config.get('YOUTUBE_API_KEY')
    
    def _create_http_session(self) -> requests.Session:
        """Create HTTP session with retry strategy"""
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
    
    def get_details_by_slug(self, slug: str, user_id: Optional[int] = None) -> Dict:
        """
        Main method to get all content details by slug
        Returns comprehensive details for the details page
        """
        try:
            # Try cache first
            cache_key = f"details:{slug}"
            if self.cache:
                cached = self.cache.get(cache_key)
                if cached:
                    logger.info(f"Cache hit for slug: {slug}")
                    # Add user-specific data if authenticated
                    if user_id:
                        cached = self._add_user_data(cached, user_id)
                    return cached
            
            # Find content by slug
            content = self.Content.query.filter_by(slug=slug).first()
            
            if not content:
                # Try to find by alternative slug patterns or fuzzy matching
                content = self._find_content_fuzzy(slug)
                
                if not content:
                    logger.warning(f"Content not found for slug: {slug}")
                    return None
            
            # Build comprehensive details
            details = self._build_content_details(content, user_id)
            
            # Cache the result (without user-specific data)
            if self.cache and details:
                cache_data = details.copy()
                # Remove user-specific fields before caching
                cache_data.pop('user_data', None)
                self.cache.set(cache_key, cache_data, timeout=3600)  # 1 hour
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting details for slug {slug}: {e}")
            return None
    
    def _find_content_fuzzy(self, slug: str) -> Optional[Any]:
        """Find content using fuzzy matching if exact slug not found"""
        try:
            # Extract potential title from slug
            info = SlugManager.extract_info_from_slug(slug)
            title = info['title']
            year = info['year']
            
            # Try to find by title similarity
            query = self.Content.query.filter(
                func.lower(self.Content.title).like(f"%{title.lower()}%")
            )
            
            if year:
                query = query.filter(
                    func.extract('year', self.Content.release_date) == year
                )
            
            results = query.all()
            
            if results:
                # Return best match (highest similarity)
                best_match = max(results, key=lambda x: self._calculate_similarity(x.title, title))
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fuzzy content search: {e}")
            return None
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity score"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def _build_content_details(self, content: Any, user_id: Optional[int] = None) -> Dict:
        """Build comprehensive content details"""
        try:
            # Prepare futures for concurrent API calls
            futures = {}
            
            with self.executor as executor:
                # External API calls
                if content.tmdb_id:
                    futures['tmdb'] = executor.submit(self._fetch_tmdb_details, content.tmdb_id, content.content_type)
                if content.imdb_id:
                    futures['omdb'] = executor.submit(self._fetch_omdb_details, content.imdb_id)
                
                # Internal data fetching
                futures['cast_crew'] = executor.submit(self._get_cast_crew, content.id)
                futures['reviews'] = executor.submit(self._get_reviews, content.id)
                futures['similar'] = executor.submit(self._get_similar_content, content.id)
                futures['gallery'] = executor.submit(self._get_gallery, content.id)
                futures['trailer'] = executor.submit(self._get_trailer, content.title, content.content_type)
            
            # Collect results
            tmdb_data = futures.get('tmdb').result() if 'tmdb' in futures else {}
            omdb_data = futures.get('omdb').result() if 'omdb' in futures else {}
            cast_crew = futures['cast_crew'].result()
            reviews = futures['reviews'].result()
            similar = futures['similar'].result()
            gallery = futures['gallery'].result()
            trailer = futures['trailer'].result()
            
            # Build synopsis
            synopsis = self._build_synopsis(content, tmdb_data, omdb_data)
            
            # Build ratings
            ratings = self._build_ratings(content, tmdb_data, omdb_data)
            
            # Build metadata
            metadata = self._build_metadata(content, tmdb_data, omdb_data)
            
            # Get streaming information
            streaming_info = self._get_streaming_info(content, tmdb_data)
            
            # Get seasons/episodes for TV shows
            seasons_episodes = None
            if content.content_type in ['tv', 'anime']:
                seasons_episodes = self._get_seasons_episodes(content, tmdb_data)
            
            # Construct final details object
            details = {
                'slug': content.slug,
                'id': content.id,
                'title': content.title,
                'original_title': content.original_title,
                'overview': content.overview,
                'content_type': content.content_type,
                'poster_url': self._format_image_url(content.poster_path, 'poster'),
                'backdrop_url': self._format_image_url(content.backdrop_path or tmdb_data.get('backdrop_path'), 'backdrop'),
                'trailer': trailer,
                'synopsis': synopsis,
                'cast_crew': cast_crew,
                'ratings': ratings,
                'metadata': metadata,
                'more_like_this': similar,
                'reviews': reviews,
                'gallery': gallery,
                'streaming_info': streaming_info,
                'seasons_episodes': seasons_episodes
            }
            
            # Add user-specific data if authenticated
            if user_id:
                details['user_data'] = self._get_user_data(content.id, user_id)
            
            return details
            
        except Exception as e:
            logger.error(f"Error building content details: {e}")
            raise
    
    def _fetch_tmdb_details(self, tmdb_id: int, content_type: str) -> Dict:
        """Fetch comprehensive details from TMDB"""
        try:
            endpoint = 'movie' if content_type == 'movie' else 'tv'
            url = f"{TMDB_BASE_URL}/{endpoint}/{tmdb_id}"
            
            params = {
                'api_key': TMDB_API_KEY,
                'append_to_response': 'videos,images,credits,similar,recommendations,reviews,external_ids,watch/providers,content_ratings,release_dates'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching TMDB details: {e}")
            return {}
    
    def _fetch_omdb_details(self, imdb_id: str) -> Dict:
        """Fetch details from OMDB"""
        try:
            params = {
                'apikey': OMDB_API_KEY,
                'i': imdb_id,
                'plot': 'full'
            }
            
            response = self.session.get(OMDB_BASE_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching OMDB details: {e}")
            return {}
    
    def _get_trailer(self, title: str, content_type: str) -> Optional[Dict]:
        """Get trailer information from YouTube"""
        try:
            if not YOUTUBE_API_KEY:
                return None
            
            # Search for trailer
            search_query = f"{title} official trailer"
            if content_type == 'anime':
                search_query = f"{title} anime trailer PV"
            
            url = f"{YOUTUBE_BASE_URL}/search"
            params = {
                'key': YOUTUBE_API_KEY,
                'q': search_query,
                'part': 'snippet',
                'type': 'video',
                'maxResults': 5,
                'order': 'relevance'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('items'):
                    video = data['items'][0]
                    return {
                        'youtube_id': video['id']['videoId'],
                        'title': video['snippet']['title'],
                        'thumbnail': video['snippet']['thumbnails']['high']['url'],
                        'embed_url': f"https://www.youtube.com/embed/{video['id']['videoId']}",
                        'watch_url': f"https://www.youtube.com/watch?v={video['id']['videoId']}"
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting trailer: {e}")
            return None
    
    def _build_synopsis(self, content: Any, tmdb_data: Dict, omdb_data: Dict) -> Dict:
        """Build comprehensive synopsis information"""
        synopsis = {
            'overview': content.overview or tmdb_data.get('overview', ''),
            'plot': omdb_data.get('Plot', content.overview),
            'tagline': tmdb_data.get('tagline', ''),
            'content_warnings': [],
            'themes': [],
            'keywords': []
        }
        
        # Extract content warnings from ratings
        if tmdb_data.get('content_ratings'):
            ratings = tmdb_data['content_ratings'].get('results', [])
            us_rating = next((r for r in ratings if r['iso_3166_1'] == 'US'), None)
            if us_rating:
                synopsis['content_warnings'].append({
                    'rating': us_rating.get('rating'),
                    'descriptors': us_rating.get('descriptors', [])
                })
        
        # Extract keywords/themes
        if tmdb_data.get('keywords'):
            keywords = tmdb_data['keywords'].get('keywords', []) or tmdb_data['keywords'].get('results', [])
            synopsis['keywords'] = [kw['name'] for kw in keywords[:10]]
        
        return synopsis
    
    def _build_ratings(self, content: Any, tmdb_data: Dict, omdb_data: Dict) -> Dict:
        """Build comprehensive ratings information"""
        ratings = {
            'tmdb': {
                'score': content.rating or tmdb_data.get('vote_average', 0),
                'count': content.vote_count or tmdb_data.get('vote_count', 0)
            },
            'imdb': {
                'score': float(omdb_data.get('imdbRating', 0)) if omdb_data.get('imdbRating', 'N/A') != 'N/A' else 0,
                'votes': omdb_data.get('imdbVotes', 'N/A')
            },
            'composite_score': 0,
            'critics': [],
            'audience_score': None
        }
        
        # Parse critic ratings from OMDB
        if omdb_data.get('Ratings'):
            for rating in omdb_data['Ratings']:
                ratings['critics'].append({
                    'source': rating['Source'],
                    'value': rating['Value']
                })
        
        # Calculate composite score
        scores = []
        if ratings['tmdb']['score'] > 0:
            scores.append(ratings['tmdb']['score'])
        if ratings['imdb']['score'] > 0:
            scores.append(ratings['imdb']['score'])
        
        if scores:
            ratings['composite_score'] = round(sum(scores) / len(scores), 1)
        
        return ratings
    
    def _build_metadata(self, content: Any, tmdb_data: Dict, omdb_data: Dict) -> Dict:
        """Build comprehensive metadata"""
        metadata = {
            'genres': json.loads(content.genres) if content.genres else [],
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'runtime': content.runtime or tmdb_data.get('runtime'),
            'status': tmdb_data.get('status', 'Released'),
            'original_language': tmdb_data.get('original_language'),
            'spoken_languages': [],
            'production_companies': [],
            'production_countries': [],
            'budget': tmdb_data.get('budget', 0),
            'revenue': tmdb_data.get('revenue', 0),
            'advisories': {},
            'certifications': {},
            'awards': omdb_data.get('Awards', '')
        }
        
        # Languages
        if tmdb_data.get('spoken_languages'):
            metadata['spoken_languages'] = [
                {'code': lang['iso_639_1'], 'name': lang['name']} 
                for lang in tmdb_data['spoken_languages']
            ]
        
        # Production info
        if tmdb_data.get('production_companies'):
            metadata['production_companies'] = [
                {
                    'id': company['id'],
                    'name': company['name'],
                    'logo': self._format_image_url(company.get('logo_path'), 'logo') if company.get('logo_path') else None
                }
                for company in tmdb_data['production_companies'][:5]
            ]
        
        # Certifications
        if tmdb_data.get('release_dates'):
            for country in tmdb_data['release_dates'].get('results', []):
                if country['iso_3166_1'] in ['US', 'GB', 'IN']:
                    for release in country['release_dates']:
                        if release.get('certification'):
                            metadata['certifications'][country['iso_3166_1']] = release['certification']
                            break
        
        return metadata
    
    def _get_cast_crew(self, content_id: int) -> Dict:
        """Get cast and crew information"""
        try:
            cast_crew = {
                'cast': [],
                'crew': {
                    'directors': [],
                    'writers': [],
                    'producers': []
                }
            }
            
            # Query from database if we have person data
            if self.ContentPerson and self.Person:
                # Get cast
                cast_entries = self.db.session.query(
                    self.ContentPerson, self.Person
                ).join(
                    self.Person
                ).filter(
                    self.ContentPerson.content_id == content_id,
                    self.ContentPerson.role_type == 'cast'
                ).order_by(
                    self.ContentPerson.order
                ).limit(20).all()
                
                for cp, person in cast_entries:
                    cast_crew['cast'].append({
                        'id': person.id,
                        'name': person.name,
                        'character': cp.character,
                        'profile_path': self._format_image_url(person.profile_path, 'profile'),
                        'slug': person.slug,
                        'popularity': person.popularity
                    })
                
                # Get crew
                crew_entries = self.db.session.query(
                    self.ContentPerson, self.Person
                ).join(
                    self.Person
                ).filter(
                    self.ContentPerson.content_id == content_id,
                    self.ContentPerson.role_type == 'crew'
                ).all()
                
                for cp, person in crew_entries:
                    crew_data = {
                        'id': person.id,
                        'name': person.name,
                        'job': cp.job,
                        'profile_path': self._format_image_url(person.profile_path, 'profile'),
                        'slug': person.slug
                    }
                    
                    if cp.department == 'Directing':
                        cast_crew['crew']['directors'].append(crew_data)
                    elif cp.department == 'Writing':
                        cast_crew['crew']['writers'].append(crew_data)
                    elif cp.department == 'Production' and 'Producer' in cp.job:
                        cast_crew['crew']['producers'].append(crew_data)
            
            return cast_crew
            
        except Exception as e:
            logger.error(f"Error getting cast/crew: {e}")
            return {'cast': [], 'crew': {'directors': [], 'writers': [], 'producers': []}}
    
    def get_person_details(self, person_slug: str) -> Dict:
        """Get comprehensive person details"""
        try:
            # Find person by slug
            person = self.Person.query.filter_by(slug=person_slug).first()
            
            if not person:
                return None
            
            # Get filmography
            filmography = self.db.session.query(
                self.ContentPerson, self.Content
            ).join(
                self.Content
            ).filter(
                self.ContentPerson.person_id == person.id
            ).order_by(
                self.Content.release_date.desc()
            ).all()
            
            # Organize filmography
            works = {
                'as_actor': [],
                'as_director': [],
                'as_writer': [],
                'as_producer': [],
                'upcoming': []
            }
            
            for cp, content in filmography:
                work = {
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'year': content.release_date.year if content.release_date else None,
                    'content_type': content.content_type,
                    'poster_path': self._format_image_url(content.poster_path, 'poster'),
                    'rating': content.rating,
                    'character': cp.character,
                    'job': cp.job
                }
                
                # Check if upcoming
                if content.release_date and content.release_date > datetime.now().date():
                    works['upcoming'].append(work)
                elif cp.role_type == 'cast':
                    works['as_actor'].append(work)
                elif cp.department == 'Directing':
                    works['as_director'].append(work)
                elif cp.department == 'Writing':
                    works['as_writer'].append(work)
                elif 'Producer' in (cp.job or ''):
                    works['as_producer'].append(work)
            
            # Fetch additional data from TMDB if available
            tmdb_data = {}
            if person.tmdb_id:
                try:
                    url = f"{TMDB_BASE_URL}/person/{person.tmdb_id}"
                    params = {
                        'api_key': TMDB_API_KEY,
                        'append_to_response': 'images,external_ids,combined_credits'
                    }
                    response = self.session.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        tmdb_data = response.json()
                except Exception as e:
                    logger.error(f"Error fetching person from TMDB: {e}")
            
            return {
                'id': person.id,
                'slug': person.slug,
                'name': person.name,
                'biography': person.biography or tmdb_data.get('biography', ''),
                'birthday': person.birthday.isoformat() if person.birthday else tmdb_data.get('birthday'),
                'deathday': person.deathday.isoformat() if person.deathday else tmdb_data.get('deathday'),
                'place_of_birth': person.place_of_birth or tmdb_data.get('place_of_birth'),
                'profile_path': self._format_image_url(person.profile_path, 'profile'),
                'popularity': person.popularity or tmdb_data.get('popularity', 0),
                'known_for_department': person.known_for_department or tmdb_data.get('known_for_department'),
                'also_known_as': json.loads(person.also_known_as) if person.also_known_as else tmdb_data.get('also_known_as', []),
                'gender': person.gender,
                'filmography': works,
                'images': self._get_person_images(tmdb_data),
                'social_media': self._get_person_social_media(tmdb_data),
                'total_works': len(filmography),
                'awards': []  # Could be expanded with awards data
            }
            
        except Exception as e:
            logger.error(f"Error getting person details: {e}")
            return None
    
    def _get_person_images(self, tmdb_data: Dict) -> List[str]:
        """Get person images from TMDB data"""
        images = []
        if tmdb_data.get('images', {}).get('profiles'):
            for img in tmdb_data['images']['profiles'][:10]:
                images.append(self._format_image_url(img['file_path'], 'profile'))
        return images
    
    def _get_person_social_media(self, tmdb_data: Dict) -> Dict:
        """Get person's social media links"""
        social = {}
        if tmdb_data.get('external_ids'):
            ids = tmdb_data['external_ids']
            if ids.get('twitter_id'):
                social['twitter'] = f"https://twitter.com/{ids['twitter_id']}"
            if ids.get('instagram_id'):
                social['instagram'] = f"https://instagram.com/{ids['instagram_id']}"
            if ids.get('facebook_id'):
                social['facebook'] = f"https://facebook.com/{ids['facebook_id']}"
            if ids.get('imdb_id'):
                social['imdb'] = f"https://www.imdb.com/name/{ids['imdb_id']}"
        return social
    
    def _get_reviews(self, content_id: int, limit: int = 10) -> List[Dict]:
        """Get user reviews for content"""
        try:
            reviews = []
            
            if self.Review:
                review_entries = self.db.session.query(
                    self.Review, self.User
                ).join(
                    self.User
                ).filter(
                    self.Review.content_id == content_id,
                    self.Review.is_approved == True
                ).order_by(
                    self.Review.helpful_count.desc(),
                    self.Review.created_at.desc()
                ).limit(limit).all()
                
                for review, user in review_entries:
                    reviews.append({
                        'id': review.id,
                        'user': {
                            'id': user.id,
                            'username': user.username,
                            'avatar': user.avatar_url
                        },
                        'rating': review.rating,
                        'title': review.title,
                        'review_text': review.review_text,
                        'has_spoilers': review.has_spoilers,
                        'helpful_count': review.helpful_count,
                        'created_at': review.created_at.isoformat(),
                        'updated_at': review.updated_at.isoformat() if review.updated_at else None
                    })
            
            return reviews
            
        except Exception as e:
            logger.error(f"Error getting reviews: {e}")
            return []
    
    def _get_similar_content(self, content_id: int, limit: int = 12) -> List[Dict]:
        """Get similar/recommended content"""
        try:
            similar = []
            
            # Get the current content
            content = self.Content.query.get(content_id)
            if not content:
                return []
            
            # Parse genres
            genres = json.loads(content.genres) if content.genres else []
            
            # Find similar content based on genres and type
            query = self.Content.query.filter(
                self.Content.id != content_id,
                self.Content.content_type == content.content_type
            )
            
            # Filter by genres if available
            if genres:
                genre_filters = []
                for genre in genres[:3]:  # Use top 3 genres
                    genre_filters.append(self.Content.genres.contains(genre))
                query = query.filter(or_(*genre_filters))
            
            # Order by rating and popularity
            similar_content = query.order_by(
                self.Content.rating.desc(),
                self.Content.popularity.desc()
            ).limit(limit).all()
            
            for item in similar_content:
                similar.append({
                    'id': item.id,
                    'slug': item.slug,
                    'title': item.title,
                    'content_type': item.content_type,
                    'poster_path': self._format_image_url(item.poster_path, 'poster'),
                    'rating': item.rating,
                    'year': item.release_date.year if item.release_date else None,
                    'genres': json.loads(item.genres) if item.genres else []
                })
            
            return similar
            
        except Exception as e:
            logger.error(f"Error getting similar content: {e}")
            return []
    
    def _get_gallery(self, content_id: int) -> Dict:
        """Get content gallery (posters, backdrops, stills)"""
        try:
            gallery = {
                'posters': [],
                'backdrops': [],
                'stills': []
            }
            
            # Get from TMDB if available
            content = self.Content.query.get(content_id)
            if content and content.tmdb_id:
                endpoint = 'movie' if content.content_type == 'movie' else 'tv'
                url = f"{TMDB_BASE_URL}/{endpoint}/{content.tmdb_id}/images"
                
                params = {
                    'api_key': TMDB_API_KEY
                }
                
                response = self.session.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Add posters
                    for img in data.get('posters', [])[:10]:
                        gallery['posters'].append({
                            'url': self._format_image_url(img['file_path'], 'poster'),
                            'thumbnail': self._format_image_url(img['file_path'], 'thumbnail'),
                            'aspect_ratio': img.get('aspect_ratio'),
                            'width': img.get('width'),
                            'height': img.get('height')
                        })
                    
                    # Add backdrops
                    for img in data.get('backdrops', [])[:10]:
                        gallery['backdrops'].append({
                            'url': self._format_image_url(img['file_path'], 'backdrop'),
                            'thumbnail': self._format_image_url(img['file_path'], 'still'),
                            'aspect_ratio': img.get('aspect_ratio'),
                            'width': img.get('width'),
                            'height': img.get('height')
                        })
                    
                    # Add stills for TV shows
                    if content.content_type == 'tv' and data.get('stills'):
                        for img in data.get('stills', [])[:10]:
                            gallery['stills'].append({
                                'url': self._format_image_url(img['file_path'], 'still'),
                                'thumbnail': self._format_image_url(img['file_path'], 'thumbnail'),
                                'aspect_ratio': img.get('aspect_ratio'),
                                'width': img.get('width'),
                                'height': img.get('height')
                            })
            
            # Add default poster if no gallery images
            if not gallery['posters'] and content.poster_path:
                gallery['posters'].append({
                    'url': self._format_image_url(content.poster_path, 'poster'),
                    'thumbnail': self._format_image_url(content.poster_path, 'thumbnail')
                })
            
            return gallery
            
        except Exception as e:
            logger.error(f"Error getting gallery: {e}")
            return {'posters': [], 'backdrops': [], 'stills': []}
    
    def _get_streaming_info(self, content: Any, tmdb_data: Dict) -> Optional[Dict]:
        """Get streaming availability information"""
        try:
            streaming = {
                'rent': [],
                'buy': [],
                'stream': [],
                'free': []
            }
            
            if tmdb_data.get('watch/providers'):
                providers = tmdb_data['watch/providers'].get('results', {})
                
                # Get providers for user's region (defaulting to US)
                region_data = providers.get('US', {})
                
                if region_data.get('flatrate'):
                    for provider in region_data['flatrate']:
                        streaming['stream'].append({
                            'provider_id': provider['provider_id'],
                            'provider_name': provider['provider_name'],
                            'logo_path': self._format_image_url(provider.get('logo_path'), 'logo')
                        })
                
                if region_data.get('rent'):
                    for provider in region_data['rent']:
                        streaming['rent'].append({
                            'provider_id': provider['provider_id'],
                            'provider_name': provider['provider_name'],
                            'logo_path': self._format_image_url(provider.get('logo_path'), 'logo')
                        })
                
                if region_data.get('buy'):
                    for provider in region_data['buy']:
                        streaming['buy'].append({
                            'provider_id': provider['provider_id'],
                            'provider_name': provider['provider_name'],
                            'logo_path': self._format_image_url(provider.get('logo_path'), 'logo')
                        })
                
                if region_data.get('free'):
                    for provider in region_data['free']:
                        streaming['free'].append({
                            'provider_id': provider['provider_id'],
                            'provider_name': provider['provider_name'],
                            'logo_path': self._format_image_url(provider.get('logo_path'), 'logo')
                        })
            
            # Return None if no streaming options available
            if not any([streaming['rent'], streaming['buy'], streaming['stream'], streaming['free']]):
                return None
            
            return streaming
            
        except Exception as e:
            logger.error(f"Error getting streaming info: {e}")
            return None
    
    def _get_seasons_episodes(self, content: Any, tmdb_data: Dict) -> Optional[Dict]:
        """Get seasons and episodes for TV shows"""
        try:
            if content.content_type not in ['tv', 'anime']:
                return None
            
            seasons_data = {
                'season_count': tmdb_data.get('number_of_seasons', 0),
                'episode_count': tmdb_data.get('number_of_episodes', 0),
                'seasons': []
            }
            
            # Get season details
            if tmdb_data.get('seasons'):
                for season in tmdb_data['seasons']:
                    # Skip specials (season 0) unless it's the only season
                    if season['season_number'] == 0 and len(tmdb_data['seasons']) > 1:
                        continue
                    
                    seasons_data['seasons'].append({
                        'season_number': season['season_number'],
                        'name': season['name'],
                        'episode_count': season['episode_count'],
                        'air_date': season.get('air_date'),
                        'overview': season.get('overview'),
                        'poster_path': self._format_image_url(season.get('poster_path'), 'poster')
                    })
            
            # Get next episode info
            if tmdb_data.get('next_episode_to_air'):
                next_ep = tmdb_data['next_episode_to_air']
                seasons_data['next_episode'] = {
                    'name': next_ep['name'],
                    'season_number': next_ep['season_number'],
                    'episode_number': next_ep['episode_number'],
                    'air_date': next_ep['air_date'],
                    'overview': next_ep.get('overview')
                }
            
            # Get last episode info
            if tmdb_data.get('last_episode_to_air'):
                last_ep = tmdb_data['last_episode_to_air']
                seasons_data['last_episode'] = {
                    'name': last_ep['name'],
                    'season_number': last_ep['season_number'],
                    'episode_number': last_ep['episode_number'],
                    'air_date': last_ep['air_date']
                }
            
            return seasons_data
            
        except Exception as e:
            logger.error(f"Error getting seasons/episodes: {e}")
            return None
    
    def _get_user_data(self, content_id: int, user_id: int) -> Dict:
        """Get user-specific data for content"""
        try:
            user_data = {
                'in_watchlist': False,
                'in_favorites': False,
                'user_rating': None,
                'watch_status': None  # 'watching', 'completed', 'plan_to_watch', 'dropped'
            }
            
            # Check watchlist
            watchlist_item = self.UserInteraction.query.filter_by(
                user_id=user_id,
                content_id=content_id,
                interaction_type='watchlist'
            ).first()
            
            if watchlist_item:
                user_data['in_watchlist'] = True
            
            # Check favorites
            favorite_item = self.UserInteraction.query.filter_by(
                user_id=user_id,
                content_id=content_id,
                interaction_type='favorite'
            ).first()
            
            if favorite_item:
                user_data['in_favorites'] = True
            
            # Get user rating
            rating_item = self.UserInteraction.query.filter_by(
                user_id=user_id,
                content_id=content_id,
                interaction_type='rating'
            ).first()
            
            if rating_item:
                user_data['user_rating'] = rating_item.rating
            
            # Get watch status
            status_item = self.UserInteraction.query.filter_by(
                user_id=user_id,
                content_id=content_id,
                interaction_type='watch_status'
            ).first()
            
            if status_item:
                user_data['watch_status'] = status_item.metadata.get('status') if status_item.metadata else None
            
            return user_data
            
        except Exception as e:
            logger.error(f"Error getting user data: {e}")
            return {
                'in_watchlist': False,
                'in_favorites': False,
                'user_rating': None,
                'watch_status': None
            }
    
    def _add_user_data(self, details: Dict, user_id: int) -> Dict:
        """Add user-specific data to cached details"""
        if details and 'id' in details:
            details['user_data'] = self._get_user_data(details['id'], user_id)
        return details
    
    def _format_image_url(self, path: str, image_type: str = 'poster') -> Optional[str]:
        """Format image URL based on type"""
        if not path:
            return None
        
        if path.startswith('http'):
            return path
        
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
    
    def add_review(self, content_id: int, user_id: int, review_data: Dict) -> Dict:
        """Add a new review for content"""
        try:
            review = self.Review(
                content_id=content_id,
                user_id=user_id,
                rating=review_data.get('rating'),
                title=review_data.get('title'),
                review_text=review_data.get('review_text'),
                has_spoilers=review_data.get('has_spoilers', False),
                is_approved=False  # Reviews need moderation
            )
            
            self.db.session.add(review)
            self.db.session.commit()
            
            # Invalidate cache
            content = self.Content.query.get(content_id)
            if content and self.cache:
                cache_key = f"details:{content.slug}"
                self.cache.delete(cache_key)
            
            return {
                'success': True,
                'review_id': review.id,
                'message': 'Review submitted for moderation'
            }
            
        except Exception as e:
            logger.error(f"Error adding review: {e}")
            self.db.session.rollback()
            return {
                'success': False,
                'message': 'Failed to add review'
            }
    
    def vote_review_helpful(self, review_id: int, user_id: int, is_helpful: bool = True) -> bool:
        """Vote on review helpfulness"""
        try:
            review = self.Review.query.get(review_id)
            if not review:
                return False
            
            # Update helpful count
            if is_helpful:
                review.helpful_count = (review.helpful_count or 0) + 1
            else:
                review.helpful_count = max(0, (review.helpful_count or 0) - 1)
            
            self.db.session.commit()
            
            # Invalidate cache
            content = self.Content.query.get(review.content_id)
            if content and self.cache:
                cache_key = f"details:{content.slug}"
                self.cache.delete(cache_key)
            
            return True
            
        except Exception as e:
            logger.error(f"Error voting on review: {e}")
            self.db.session.rollback()
            return False

def init_details_service(app, db, models, cache):
    """Initialize the details service with app context"""
    service = DetailsService(db, models, cache)
    return service