# backend/services/detail.py
"""
Production-ready detail service for movie, TV show, and anime platform.
Handles SEO-friendly slugs, comprehensive content details, and related data.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from urllib.parse import quote, unquote
import unicodedata

from flask import current_app, jsonify
from sqlalchemy import and_, or_, func, text
from sqlalchemy.exc import SQLAlchemyError
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content type enumeration"""
    MOVIE = "movie"
    TV = "tv"
    ANIME = "anime"


class MediaQuality(Enum):
    """Media quality enumeration"""
    SD = "SD"
    HD = "HD"
    FHD = "FHD"
    UHD_4K = "4K"
    UHD_8K = "8K"
    HDR = "HDR"
    DOLBY_VISION = "Dolby Vision"


class ContentRating(Enum):
    """Content rating enumeration"""
    G = "G"
    PG = "PG"
    PG_13 = "PG-13"
    R = "R"
    NC_17 = "NC-17"
    TV_Y = "TV-Y"
    TV_Y7 = "TV-Y7"
    TV_G = "TV-G"
    TV_PG = "TV-PG"
    TV_14 = "TV-14"
    TV_MA = "TV-MA"


@dataclass
class PersonDetail:
    """Person (cast/crew) detail structure"""
    id: int
    name: str
    role: str
    character: Optional[str] = None
    profile_image: Optional[str] = None
    department: Optional[str] = None
    popularity: Optional[float] = None
    biography: Optional[str] = None
    birthday: Optional[str] = None
    place_of_birth: Optional[str] = None
    known_for: Optional[List[Dict]] = None
    upcoming_projects: Optional[List[Dict]] = None
    social_media: Optional[Dict] = None


@dataclass
class Review:
    """User review structure"""
    id: str
    author: str
    author_avatar: Optional[str]
    content: str
    rating: float
    created_at: datetime
    updated_at: Optional[datetime]
    helpful_count: int
    spoiler_tagged: bool
    verified_purchase: bool = False
    sentiment: Optional[str] = None


@dataclass
class MediaAsset:
    """Media asset (image/video) structure"""
    id: str
    type: str  # poster, backdrop, still, bts
    url: str
    thumbnail_url: Optional[str]
    width: int
    height: int
    aspect_ratio: float
    language: Optional[str]
    vote_average: Optional[float]
    vote_count: Optional[int]


@dataclass
class TrailerInfo:
    """Trailer information structure"""
    id: str
    key: str
    site: str  # YouTube, Vimeo
    type: str  # Trailer, Teaser, Clip, Behind the Scenes
    name: str
    official: bool
    published_at: datetime
    size: int  # 360, 480, 720, 1080
    language: str
    autoplay_enabled: bool
    adaptive_quality: bool
    custom_controls: bool


class SlugService:
    """Service for handling SEO-friendly slugs"""
    
    @staticmethod
    def generate_slug(title: str, year: Optional[int] = None, 
                     content_type: Optional[str] = None) -> str:
        """
        Generate SEO-friendly slug from title.
        Follows Netflix/Disney+ slug patterns.
        
        Args:
            title: Content title
            year: Release year
            content_type: Type of content (movie, tv, anime)
            
        Returns:
            SEO-friendly slug
        """
        if not title:
            return None
            
        # Normalize unicode characters
        title = unicodedata.normalize('NFKD', title)
        title = title.encode('ascii', 'ignore').decode('ascii')
        
        # Convert to lowercase and replace spaces/special chars
        slug = title.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[-\s]+', '-', slug)
        slug = slug.strip('-')
        
        # Add year if provided (Netflix pattern)
        if year:
            slug = f"{slug}-{year}"
            
        # Add content type suffix for disambiguation
        if content_type:
            type_suffix = {
                'movie': 'film',
                'tv': 'series',
                'anime': 'anime'
            }.get(content_type)
            if type_suffix:
                slug = f"{slug}-{type_suffix}"
        
        # Ensure slug is not too long (SEO best practice)
        if len(slug) > 60:
            slug = slug[:60].rsplit('-', 1)[0]
        
        return slug
    
    @staticmethod
    def generate_unique_slug(db, model, title: str, year: Optional[int] = None,
                           content_type: Optional[str] = None) -> str:
        """
        Generate unique slug, handling duplicates.
        
        Args:
            db: Database session
            model: Content model
            title: Content title
            year: Release year
            content_type: Type of content
            
        Returns:
            Unique SEO-friendly slug
        """
        base_slug = SlugService.generate_slug(title, year, content_type)
        if not base_slug:
            return None
            
        slug = base_slug
        counter = 1
        
        while db.session.query(model).filter_by(slug=slug).first():
            slug = f"{base_slug}-{counter}"
            counter += 1
            
        return slug
    
    @staticmethod
    def parse_slug(slug: str) -> Dict[str, Any]:
        """
        Parse slug to extract information.
        
        Args:
            slug: SEO-friendly slug
            
        Returns:
            Dictionary with parsed information
        """
        parts = slug.split('-')
        
        # Try to extract year (4-digit number)
        year = None
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) == 4:
                year = int(part)
                parts.pop(i)
                break
        
        # Check for content type suffix
        content_type = None
        if parts and parts[-1] in ['film', 'series', 'anime']:
            content_type = {
                'film': 'movie',
                'series': 'tv',
                'anime': 'anime'
            }[parts[-1]]
            parts.pop()
        
        # Reconstruct title
        title = ' '.join(parts).title()
        
        return {
            'title': title,
            'year': year,
            'content_type': content_type,
            'original_slug': slug
        }


class DetailService:
    """Main service for handling content details"""
    
    def __init__(self, db, cache, models, services):
        """
        Initialize DetailService.
        
        Args:
            db: Database instance
            cache: Cache instance
            models: Dictionary of database models
            services: Dictionary of external services
        """
        self.db = db
        self.cache = cache
        self.models = models
        self.services = services
        self.slug_service = SlugService()
        self.http_session = self._create_http_session()
        
        # API configurations - safely get from current_app config
        try:
            self.tmdb_api_key = current_app.config.get('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
            self.omdb_api_key = current_app.config.get('OMDB_API_KEY', '52260795')
            self.youtube_api_key = current_app.config.get('YOUTUBE_API_KEY', 'AIzaSyDU-JLASTdIdoLOmlpWuJYLTZDUspqw2T4')
        except:
            # Fallback to default values if current_app is not available
            self.tmdb_api_key = '1cf86635f20bb2aff8e70940e7c3ddd5'
            self.omdb_api_key = '52260795'
            self.youtube_api_key = 'AIzaSyDU-JLASTdIdoLOmlpWuJYLTZDUspqw2T4'
        
        # Cache timeouts
        self.CACHE_TIMEOUT = {
            'detail': 3600,  # 1 hour
            'person': 7200,  # 2 hours
            'reviews': 1800,  # 30 minutes
            'similar': 3600,  # 1 hour
            'gallery': 7200  # 2 hours
        }
    
    def _create_http_session(self) -> requests.Session:
        """Create HTTP session with retry logic"""
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
    
    def get_content_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """
        Get content details by SEO-friendly slug.
        
        Args:
            slug: SEO-friendly slug
            
        Returns:
            Complete content details or None
        """
        try:
            # Check cache first
            cache_key = f"detail:slug:{slug}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Cache hit for slug: {slug}")
                return cached_data
            
            # Find content by slug
            Content = self.models['Content']
            content = self.db.session.query(Content).filter_by(slug=slug).first()
            
            if not content:
                # Try to find by parsing slug
                parsed = self.slug_service.parse_slug(slug)
                content = self._find_content_by_parsed_slug(parsed)
            
            if not content:
                logger.warning(f"Content not found for slug: {slug}")
                return None
            
            # Build comprehensive response
            response = self._build_detail_response(content)
            
            # Cache the response
            self.cache.set(cache_key, response, timeout=self.CACHE_TIMEOUT['detail'])
            
            # Record view analytics
            self._record_view_analytics(content.id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting content by slug {slug}: {e}")
            return None
    
    def _find_content_by_parsed_slug(self, parsed: Dict[str, Any]) -> Optional[Any]:
        """Find content by parsed slug information"""
        Content = self.models['Content']
        query = self.db.session.query(Content)
        
        # Filter by title (fuzzy match)
        if parsed['title']:
            # Try exact match first
            exact_match = query.filter(
                func.lower(Content.title) == parsed['title'].lower()
            ).first()
            
            if exact_match:
                return exact_match
            
            # Try contains match
            query = query.filter(
                func.lower(Content.title).contains(parsed['title'].lower())
            )
        
        # Filter by year if available
        if parsed['year']:
            query = query.filter(
                func.extract('year', Content.release_date) == parsed['year']
            )
        
        # Filter by content type if identified
        if parsed['content_type']:
            query = query.filter(Content.content_type == parsed['content_type'])
        
        return query.first()
    
    def _build_detail_response(self, content: Any) -> Dict[str, Any]:
        """Build comprehensive detail response"""
        response = {
            'id': content.id,
            'slug': content.slug if hasattr(content, 'slug') else self.slug_service.generate_slug(
                content.title,
                content.release_date.year if content.release_date else None,
                content.content_type
            ),
            'title': content.title,
            'original_title': content.original_title,
            'overview': content.overview,
            'content_type': content.content_type,
            'poster': self._get_image_url(content.poster_path, 'poster'),
            'backdrop': self._get_image_url(content.backdrop_path, 'backdrop'),
            'trailer': self._get_trailer_info(content),
            'synopsis': self._get_synopsis(content),
            'cast_and_crew': self._get_cast_and_crew(content),
            'ratings': self._get_ratings(content),
            'metadata': self._get_metadata(content),
            'more_like_this': self._get_similar_content(content),
            'user_lists': self._get_user_lists(content),
            'reviews': self._get_reviews(content),
            'gallery': self._get_gallery(content),
            'streaming_info': self._get_streaming_info(content),
            'technical_specs': self._get_technical_specs(content),
            'awards': self._get_awards(content),
            'box_office': self._get_box_office(content),
            'seo_metadata': self._get_seo_metadata(content)
        }
        
        return response
    
    def _get_image_url(self, path: str, image_type: str) -> Optional[str]:
        """Get full image URL with appropriate size"""
        if not path:
            return None
        
        if path.startswith('http'):
            return path
        
        size_map = {
            'poster': 'w500',
            'backdrop': 'w1280',
            'profile': 'w185',
            'still': 'w300'
        }
        
        size = size_map.get(image_type, 'original')
        return f"https://image.tmdb.org/t/p/{size}{path}"
    
    def _get_trailer_info(self, content: Any) -> Optional[Dict[str, Any]]:
        """Get trailer information with player configuration"""
        try:
            trailer_data = None
            
            # Get from YouTube if available
            if hasattr(content, 'youtube_trailer_id') and content.youtube_trailer_id:
                trailer_data = {
                    'id': content.youtube_trailer_id,
                    'key': content.youtube_trailer_id,
                    'site': 'YouTube',
                    'type': 'Trailer',
                    'name': f"{content.title} - Official Trailer",
                    'official': True,
                    'url': f"https://www.youtube.com/embed/{content.youtube_trailer_id}",
                    'thumbnail': f"https://img.youtube.com/vi/{content.youtube_trailer_id}/maxresdefault.jpg",
                    'player_config': {
                        'autoplay': True,
                        'adaptive_quality': True,
                        'custom_controls': True,
                        'cc_load_policy': 1,  # Closed captions
                        'modestbranding': 1,
                        'rel': 0,  # No related videos
                        'quality_levels': ['auto', '1080p', '720p', '480p', '360p']
                    }
                }
            
            # Get from TMDB if not available
            elif content.tmdb_id:
                videos = self._fetch_tmdb_videos(content.tmdb_id, content.content_type)
                if videos and videos.get('results'):
                    video = videos['results'][0]
                    trailer_data = {
                        'id': video['id'],
                        'key': video['key'],
                        'site': video['site'],
                        'type': video['type'],
                        'name': video['name'],
                        'official': video.get('official', False),
                        'url': f"https://www.youtube.com/embed/{video['key']}",
                        'thumbnail': f"https://img.youtube.com/vi/{video['key']}/maxresdefault.jpg",
                        'player_config': {
                            'autoplay': True,
                            'adaptive_quality': True,
                            'custom_controls': True,
                            'quality_levels': ['auto', '1080p', '720p', '480p', '360p']
                        }
                    }
            
            return trailer_data
            
        except Exception as e:
            logger.error(f"Error getting trailer info: {e}")
            return None
    
    def _get_synopsis(self, content: Any) -> Dict[str, Any]:
        """Get detailed synopsis with content warnings"""
        synopsis = {
            'overview': content.overview,
            'plot_summary': None,
            'tagline': None,
            'content_warnings': [],
            'themes': [],
            'mood': None
        }
        
        try:
            # Get additional details from TMDB
            if content.tmdb_id:
                details = self._fetch_tmdb_details(content.tmdb_id, content.content_type)
                if details:
                    synopsis['tagline'] = details.get('tagline')
                    synopsis['plot_summary'] = details.get('overview', content.overview)
                    
                    # Extract content warnings from keywords
                    keywords = details.get('keywords', {}).get('keywords', [])
                    warning_keywords = ['violence', 'gore', 'nudity', 'profanity', 'drugs', 'alcohol']
                    for keyword in keywords:
                        if any(w in keyword.get('name', '').lower() for w in warning_keywords):
                            synopsis['content_warnings'].append(keyword['name'])
            
            # Analyze themes and mood
            if content.genres:
                genres = json.loads(content.genres) if isinstance(content.genres, str) else content.genres
                synopsis['themes'] = self._extract_themes(genres, content.overview)
                synopsis['mood'] = self._determine_mood(genres)
            
        except Exception as e:
            logger.error(f"Error getting synopsis: {e}")
        
        return synopsis
    
    def _get_cast_and_crew(self, content: Any) -> Dict[str, Any]:
        """Get cast and crew with detailed information"""
        cast_crew = {
            'cast': [],
            'crew': {
                'directors': [],
                'writers': [],
                'producers': [],
                'music': [],
                'cinematography': [],
                'editing': []
            }
        }
        
        try:
            # Cache key for cast and crew
            cache_key = f"cast_crew:{content.id}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data
            
            # Fetch from TMDB
            if content.tmdb_id:
                credits = self._fetch_tmdb_credits(content.tmdb_id, content.content_type)
                if credits:
                    # Process cast
                    for person in credits.get('cast', [])[:20]:  # Top 20 cast members
                        cast_member = PersonDetail(
                            id=person['id'],
                            name=person['name'],
                            role='Actor',
                            character=person.get('character'),
                            profile_image=self._get_image_url(person.get('profile_path'), 'profile'),
                            popularity=person.get('popularity'),
                            department='Acting'
                        )
                        cast_crew['cast'].append(asdict(cast_member))
                    
                    # Process crew
                    for person in credits.get('crew', []):
                        department = person.get('department', '').lower()
                        job = person.get('job', '').lower()
                        
                        crew_member = PersonDetail(
                            id=person['id'],
                            name=person['name'],
                            role=person['job'],
                            profile_image=self._get_image_url(person.get('profile_path'), 'profile'),
                            department=person['department']
                        )
                        
                        crew_dict = asdict(crew_member)
                        
                        if 'director' in job:
                            cast_crew['crew']['directors'].append(crew_dict)
                        elif 'writer' in job or 'screenplay' in job:
                            cast_crew['crew']['writers'].append(crew_dict)
                        elif 'producer' in job:
                            cast_crew['crew']['producers'].append(crew_dict)
                        elif department == 'sound' or 'composer' in job:
                            cast_crew['crew']['music'].append(crew_dict)
                        elif 'cinematograph' in job or 'photography' in job:
                            cast_crew['crew']['cinematography'].append(crew_dict)
                        elif 'editor' in job:
                            cast_crew['crew']['editing'].append(crew_dict)
            
            # Cache the result
            self.cache.set(cache_key, cast_crew, timeout=self.CACHE_TIMEOUT['person'])
            
        except Exception as e:
            logger.error(f"Error getting cast and crew: {e}")
        
        return cast_crew
    
    def get_person_details(self, person_id: int) -> Optional[Dict[str, Any]]:
        """
        Get full person details including biography and filmography.
        
        Args:
            person_id: TMDB person ID
            
        Returns:
            Complete person details
        """
        try:
            # Check cache
            cache_key = f"person:{person_id}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data
            
            # Fetch from TMDB
            url = f"https://api.themoviedb.org/3/person/{person_id}"
            params = {
                'api_key': self.tmdb_api_key,
                'append_to_response': 'movie_credits,tv_credits,external_ids,images'
            }
            
            response = self.http_session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            # Process filmography
            filmography = []
            
            # Movies
            for movie in data.get('movie_credits', {}).get('cast', [])[:10]:
                filmography.append({
                    'id': movie['id'],
                    'title': movie['title'],
                    'type': 'movie',
                    'role': movie.get('character'),
                    'year': movie.get('release_date', '')[:4] if movie.get('release_date') else None,
                    'poster': self._get_image_url(movie.get('poster_path'), 'poster')
                })
            
            # TV Shows
            for show in data.get('tv_credits', {}).get('cast', [])[:10]:
                filmography.append({
                    'id': show['id'],
                    'title': show['name'],
                    'type': 'tv',
                    'role': show.get('character'),
                    'year': show.get('first_air_date', '')[:4] if show.get('first_air_date') else None,
                    'poster': self._get_image_url(show.get('poster_path'), 'poster')
                })
            
            # Sort by year (newest first)
            filmography.sort(key=lambda x: x.get('year') or '0000', reverse=True)
            
            # Get upcoming projects
            upcoming = []
            for movie in data.get('movie_credits', {}).get('cast', []):
                if movie.get('release_date') and movie['release_date'] > datetime.now().isoformat():
                    upcoming.append({
                        'id': movie['id'],
                        'title': movie['title'],
                        'type': 'movie',
                        'role': movie.get('character'),
                        'release_date': movie['release_date'],
                        'poster': self._get_image_url(movie.get('poster_path'), 'poster')
                    })
            
            person_details = {
                'id': person_id,
                'name': data['name'],
                'biography': data.get('biography'),
                'birthday': data.get('birthday'),
                'deathday': data.get('deathday'),
                'place_of_birth': data.get('place_of_birth'),
                'profile_image': self._get_image_url(data.get('profile_path'), 'profile'),
                'popularity': data.get('popularity'),
                'known_for_department': data.get('known_for_department'),
                'also_known_as': data.get('also_known_as', []),
                'filmography': filmography[:20],  # Top 20 works
                'upcoming_projects': upcoming[:5],  # Next 5 upcoming
                'social_media': {
                    'imdb': f"https://www.imdb.com/name/{data.get('external_ids', {}).get('imdb_id')}" if data.get('external_ids', {}).get('imdb_id') else None,
                    'twitter': f"https://twitter.com/{data.get('external_ids', {}).get('twitter_id')}" if data.get('external_ids', {}).get('twitter_id') else None,
                    'instagram': f"https://instagram.com/{data.get('external_ids', {}).get('instagram_id')}" if data.get('external_ids', {}).get('instagram_id') else None,
                    'facebook': f"https://facebook.com/{data.get('external_ids', {}).get('facebook_id')}" if data.get('external_ids', {}).get('facebook_id') else None
                },
                'images': [
                    self._get_image_url(img['file_path'], 'profile')
                    for img in data.get('images', {}).get('profiles', [])[:10]
                ]
            }
            
            # Cache the result
            self.cache.set(cache_key, person_details, timeout=self.CACHE_TIMEOUT['person'])
            
            return person_details
            
        except Exception as e:
            logger.error(f"Error getting person details: {e}")
            return None
    
    def _get_ratings(self, content: Any) -> Dict[str, Any]:
        """Get comprehensive ratings from multiple sources"""
        ratings = {
            'imdb': {
                'rating': None,
                'votes': None,
                'url': None
            },
            'tmdb': {
                'rating': content.rating,
                'votes': content.vote_count
            },
            'rotten_tomatoes': {
                'critics_score': None,
                'audience_score': None
            },
            'metacritic': {
                'metascore': None,
                'user_score': None
            },
            'composite_score': None,
            'user_rating': {
                'average': None,
                'count': 0,
                'distribution': {
                    '5': 0,
                    '4': 0,
                    '3': 0,
                    '2': 0,
                    '1': 0
                }
            }
        }
        
        try:
            # Get IMDB rating
            if content.imdb_id:
                omdb_data = self._fetch_omdb_data(content.imdb_id)
                if omdb_data:
                    ratings['imdb']['rating'] = float(omdb_data.get('imdbRating', 0)) if omdb_data.get('imdbRating') != 'N/A' else None
                    ratings['imdb']['votes'] = omdb_data.get('imdbVotes', '').replace(',', '')
                    ratings['imdb']['url'] = f"https://www.imdb.com/title/{content.imdb_id}"
                    
                    # Rotten Tomatoes
                    for rating in omdb_data.get('Ratings', []):
                        if rating['Source'] == 'Rotten Tomatoes':
                            ratings['rotten_tomatoes']['critics_score'] = int(rating['Value'].replace('%', ''))
                        elif rating['Source'] == 'Metacritic':
                            ratings['metacritic']['metascore'] = int(rating['Value'].split('/')[0])
            
            # Get user ratings from database
            UserInteraction = self.models.get('UserInteraction')
            if UserInteraction:
                user_ratings = self.db.session.query(
                    func.avg(UserInteraction.rating).label('average'),
                    func.count(UserInteraction.rating).label('count')
                ).filter(
                    UserInteraction.content_id == content.id,
                    UserInteraction.rating.isnot(None)
                ).first()
                
                if user_ratings:
                    ratings['user_rating']['average'] = round(float(user_ratings.average or 0), 1)
                    ratings['user_rating']['count'] = user_ratings.count or 0
                    
                    # Get distribution
                    distribution = self.db.session.query(
                        func.floor(UserInteraction.rating).label('star'),
                        func.count(UserInteraction.rating).label('count')
                    ).filter(
                        UserInteraction.content_id == content.id,
                        UserInteraction.rating.isnot(None)
                    ).group_by(func.floor(UserInteraction.rating)).all()
                    
                    for star, count in distribution:
                        ratings['user_rating']['distribution'][str(int(star))] = count
            
            # Calculate composite score
            scores = []
            weights = []
            
            if ratings['imdb']['rating']:
                scores.append(ratings['imdb']['rating'])
                weights.append(3)  # IMDB weight
            
            if ratings['tmdb']['rating']:
                scores.append(ratings['tmdb']['rating'])
                weights.append(2)  # TMDB weight
            
            if ratings['rotten_tomatoes']['critics_score']:
                scores.append(ratings['rotten_tomatoes']['critics_score'] / 10)
                weights.append(2)  # RT weight
            
            if ratings['user_rating']['average'] and ratings['user_rating']['count'] >= 10:
                scores.append(ratings['user_rating']['average'] * 2)  # Convert 5-star to 10-scale
                weights.append(1)  # User rating weight
            
            if scores:
                weighted_sum = sum(s * w for s, w in zip(scores, weights))
                total_weight = sum(weights)
                ratings['composite_score'] = round(weighted_sum / total_weight, 1)
            
        except Exception as e:
            logger.error(f"Error getting ratings: {e}")
        
        return ratings
    
    def _get_metadata(self, content: Any) -> Dict[str, Any]:
        """Get comprehensive metadata"""
        metadata = {
            'genres': json.loads(content.genres) if content.genres else [],
            'languages': json.loads(content.languages) if content.languages else [],
            'runtime': content.runtime,
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'release_year': content.release_date.year if content.release_date else None,
            'status': None,
            'budget': None,
            'revenue': None,
            'production_companies': [],
            'production_countries': [],
            'spoken_languages': [],
            'content_rating': None,
            'advisory': {
                'violence': False,
                'profanity': False,
                'nudity': False,
                'substance': False,
                'frightening': False
            },
            'quality': {
                '4k': False,
                'hdr': False,
                'dolby_vision': False,
                'dolby_atmos': False,
                'imax': False
            },
            'availability': {
                'digital': True,
                'physical': False,
                'theatrical': False
            }
        }
        
        try:
            # Get additional metadata from TMDB
            if content.tmdb_id:
                details = self._fetch_tmdb_details(content.tmdb_id, content.content_type)
                if details:
                    metadata['status'] = details.get('status')
                    metadata['budget'] = details.get('budget')
                    metadata['revenue'] = details.get('revenue')
                    metadata['production_companies'] = [
                        {
                            'id': company['id'],
                            'name': company['name'],
                            'logo': self._get_image_url(company.get('logo_path'), 'poster'),
                            'country': company.get('origin_country')
                        }
                        for company in details.get('production_companies', [])
                    ]
                    metadata['production_countries'] = [
                        country['name'] for country in details.get('production_countries', [])
                    ]
                    metadata['spoken_languages'] = [
                        {
                            'code': lang['iso_639_1'],
                            'name': lang['name'],
                            'english_name': lang.get('english_name')
                        }
                        for lang in details.get('spoken_languages', [])
                    ]
                    
                    # Get content rating
                    if content.content_type == 'movie':
                        releases = self._fetch_tmdb_releases(content.tmdb_id)
                        if releases:
                            for country in releases.get('results', []):
                                if country['iso_3166_1'] == 'US':
                                    for release in country.get('release_dates', []):
                                        if release.get('certification'):
                                            metadata['content_rating'] = release['certification']
                                            break
                    else:  # TV shows
                        content_ratings = self._fetch_tmdb_content_ratings(content.tmdb_id)
                        if content_ratings:
                            for rating in content_ratings.get('results', []):
                                if rating['iso_3166_1'] == 'US':
                                    metadata['content_rating'] = rating['rating']
                                    break
            
            # Determine advisories based on genres and rating
            if metadata['content_rating']:
                metadata['advisory'] = self._determine_advisories(
                    metadata['content_rating'],
                    metadata['genres']
                )
            
            # Check for quality features (mock data - would come from streaming service)
            if content.release_date and content.release_date.year >= 2018:
                metadata['quality']['4k'] = True
                if content.release_date.year >= 2020:
                    metadata['quality']['hdr'] = True
            
        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
        
        return metadata
    
    def _get_similar_content(self, content: Any) -> List[Dict[str, Any]]:
        """Get similar content recommendations"""
        similar = []
        
        try:
            # Use the similarity engine from algorithms if available
            if self.services.get('recommendation_orchestrator'):
                Content = self.models['Content']
                content_pool = self.db.session.query(Content).filter(
                    Content.id != content.id,
                    Content.content_type == content.content_type
                ).limit(500).all()
                
                if hasattr(self.services['recommendation_orchestrator'], 'get_ultra_similar_content'):
                    similar_items = self.services['recommendation_orchestrator'].get_ultra_similar_content(
                        content.id,
                        content_pool,
                        limit=12,
                        strict_mode=True,
                        min_similarity=0.5
                    )
                    
                    for item in similar_items:
                        similar.append({
                            'id': item['id'],
                            'slug': self.slug_service.generate_slug(
                                item['title'],
                                item.get('release_year'),
                                item['content_type']
                            ),
                            'title': item['title'],
                            'poster': item.get('poster_path'),
                            'rating': item.get('rating'),
                            'similarity_score': item.get('similarity_score'),
                            'match_reason': item.get('match_type')
                        })
            
            # Fallback to TMDB recommendations if no results
            if not similar and content.tmdb_id:
                recommendations = self._fetch_tmdb_recommendations(content.tmdb_id, content.content_type)
                if recommendations:
                    for item in recommendations.get('results', [])[:12]:
                        similar.append({
                            'id': item['id'],
                            'title': item.get('title') or item.get('name'),
                            'poster': self._get_image_url(item.get('poster_path'), 'poster'),
                            'rating': item.get('vote_average'),
                            'overview': item.get('overview')
                        })
            
        except Exception as e:
            logger.error(f"Error getting similar content: {e}")
        
        return similar
    
    def _get_user_lists(self, content: Any) -> Dict[str, Any]:
        """Get user list status (watchlist, favorites)"""
        lists = {
            'in_watchlist': False,
            'in_favorites': False,
            'user_rating': None,
            'watched': False
        }
        
        # This would check against current user's lists
        # For now, returning default values
        return lists
    
    def _get_reviews(self, content: Any) -> Dict[str, Any]:
        """Get user reviews with moderation"""
        reviews_data = {
            'total_count': 0,
            'average_rating': 0,
            'reviews': [],
            'featured_review': None,
            'critic_reviews': []
        }
        
        try:
            # Get user reviews from database
            UserInteraction = self.models.get('UserInteraction')
            User = self.models.get('User')
            
            if UserInteraction and User:
                # Get reviews with ratings
                reviews_query = self.db.session.query(
                    UserInteraction,
                    User
                ).join(
                    User, UserInteraction.user_id == User.id
                ).filter(
                    UserInteraction.content_id == content.id,
                    UserInteraction.interaction_type == 'review'
                ).order_by(
                    UserInteraction.timestamp.desc()
                ).limit(20)
                
                for interaction, user in reviews_query:
                    review = Review(
                        id=str(interaction.id),
                        author=user.username,
                        author_avatar=None,  # Would get from user profile
                        content=interaction.review_text if hasattr(interaction, 'review_text') else '',
                        rating=interaction.rating or 0,
                        created_at=interaction.timestamp,
                        updated_at=None,
                        helpful_count=0,
                        spoiler_tagged=False,
                        verified_purchase=True,
                        sentiment=self._analyze_sentiment(interaction.review_text if hasattr(interaction, 'review_text') else '')
                    )
                    reviews_data['reviews'].append(asdict(review))
            
            # Get critic reviews from TMDB
            if content.tmdb_id:
                tmdb_reviews = self._fetch_tmdb_reviews(content.tmdb_id, content.content_type)
                if tmdb_reviews:
                    for review in tmdb_reviews.get('results', [])[:5]:
                        critic_review = {
                            'id': review['id'],
                            'author': review['author'],
                            'content': review['content'][:500] + '...' if len(review['content']) > 500 else review['content'],
                            'rating': review.get('author_details', {}).get('rating'),
                            'url': review['url'],
                            'created_at': review['created_at']
                        }
                        reviews_data['critic_reviews'].append(critic_review)
            
            # Calculate stats
            if reviews_data['reviews']:
                ratings = [r['rating'] for r in reviews_data['reviews'] if r['rating']]
                if ratings:
                    reviews_data['average_rating'] = round(sum(ratings) / len(ratings), 1)
                reviews_data['total_count'] = len(reviews_data['reviews'])
                
                # Set featured review (highest rated or most helpful)
                reviews_data['featured_review'] = max(
                    reviews_data['reviews'],
                    key=lambda x: (x['rating'] or 0) + (x['helpful_count'] or 0)
                )
            
        except Exception as e:
            logger.error(f"Error getting reviews: {e}")
        
        return reviews_data
    
    def _get_gallery(self, content: Any) -> Dict[str, Any]:
        """Get image gallery"""
        gallery = {
            'posters': [],
            'backdrops': [],
            'stills': [],
            'behind_the_scenes': [],
            'total_count': 0
        }
        
        try:
            if content.tmdb_id:
                images = self._fetch_tmdb_images(content.tmdb_id, content.content_type)
                if images:
                    # Process posters
                    for img in images.get('posters', [])[:10]:
                        gallery['posters'].append({
                            'url': self._get_image_url(img['file_path'], 'poster'),
                            'thumbnail': self._get_image_url(img['file_path'], 'still'),
                            'width': img['width'],
                            'height': img['height'],
                            'aspect_ratio': img['aspect_ratio'],
                            'language': img.get('iso_639_1'),
                            'vote_average': img.get('vote_average')
                        })
                    
                    # Process backdrops
                    for img in images.get('backdrops', [])[:10]:
                        gallery['backdrops'].append({
                            'url': self._get_image_url(img['file_path'], 'backdrop'),
                            'thumbnail': self._get_image_url(img['file_path'], 'still'),
                            'width': img['width'],
                            'height': img['height'],
                            'aspect_ratio': img['aspect_ratio'],
                            'vote_average': img.get('vote_average')
                        })
                    
                    # Process stills (for TV shows)
                    if content.content_type == 'tv':
                        for img in images.get('stills', [])[:10]:
                            gallery['stills'].append({
                                'url': self._get_image_url(img['file_path'], 'backdrop'),
                                'thumbnail': self._get_image_url(img['file_path'], 'still'),
                                'width': img['width'],
                                'height': img['height'],
                                'aspect_ratio': img['aspect_ratio']
                            })
                    
                    gallery['total_count'] = (
                        len(gallery['posters']) +
                        len(gallery['backdrops']) +
                        len(gallery['stills'])
                    )
            
        except Exception as e:
            logger.error(f"Error getting gallery: {e}")
        
        return gallery
    
    def _get_streaming_info(self, content: Any) -> Dict[str, Any]:
        """Get streaming availability information"""
        streaming = {
            'available_on': [],
            'rent_options': [],
            'buy_options': [],
            'free_with_ads': [],
            'coming_soon_to': []
        }
        
        # This would integrate with JustWatch or similar API
        # Mock data for demonstration
        if content.popularity and content.popularity > 20:
            streaming['available_on'] = [
                {
                    'provider': 'Netflix',
                    'logo': '/netflix-logo.png',
                    'url': 'https://netflix.com',
                    'quality': '4K HDR',
                    'added_date': datetime.now().isoformat()
                }
            ]
        
        return streaming
    
    def _get_technical_specs(self, content: Any) -> Dict[str, Any]:
        """Get technical specifications"""
        specs = {
            'runtime': f"{content.runtime} minutes" if content.runtime else None,
            'aspect_ratio': '2.39:1',  # Default cinematic
            'sound_mix': ['Dolby Digital', 'Dolby Atmos'],
            'color': 'Color',
            'camera': None,
            'laboratory': None,
            'film_length': None,
            'negative_format': None,
            'cinematographic_process': None,
            'printed_film_format': None
        }
        
        # These would come from technical databases
        return specs
    
    def _get_awards(self, content: Any) -> List[Dict[str, Any]]:
        """Get awards and nominations"""
        awards = []
        
        # This would fetch from awards databases
        # Mock data for high-rated content
        if content.rating and content.rating > 7.5:
            awards.append({
                'name': 'Academy Awards',
                'year': content.release_date.year if content.release_date else None,
                'category': 'Best Picture',
                'result': 'Nominated',
                'recipient': content.title
            })
        
        return awards
    
    def _get_box_office(self, content: Any) -> Dict[str, Any]:
        """Get box office information"""
        box_office = {
            'budget': None,
            'revenue': None,
            'opening_weekend': None,
            'domestic': None,
            'international': None,
            'worldwide': None,
            'profit': None,
            'roi': None  # Return on investment
        }
        
        try:
            if content.imdb_id:
                omdb_data = self._fetch_omdb_data(content.imdb_id)
                if omdb_data:
                    box_office['revenue'] = self._parse_currency(omdb_data.get('BoxOffice'))
        except Exception as e:
            logger.error(f"Error getting box office: {e}")
        
        return box_office
    
    def _get_seo_metadata(self, content: Any) -> Dict[str, Any]:
        """Generate SEO metadata for the detail page"""
        
        # Generate description
        description = content.overview[:155] if content.overview else f"Watch {content.title}"
        if len(content.overview or '') > 155:
            description += '...'
        
        # Generate keywords
        keywords = []
        if content.genres:
            genres = json.loads(content.genres) if isinstance(content.genres, str) else content.genres
            keywords.extend(genres)
        keywords.extend([content.title, content.content_type, 'watch', 'stream'])
        if content.release_date:
            keywords.append(str(content.release_date.year))
        
        # Generate structured data (JSON-LD)
        structured_data = {
            "@context": "https://schema.org",
            "@type": "Movie" if content.content_type == 'movie' else "TVSeries",
            "name": content.title,
            "description": content.overview,
            "datePublished": content.release_date.isoformat() if content.release_date else None,
            "genre": json.loads(content.genres) if content.genres else [],
            "aggregateRating": {
                "@type": "AggregateRating",
                "ratingValue": content.rating,
                "ratingCount": content.vote_count,
                "bestRating": 10,
                "worstRating": 0
            }
        }
        
        return {
            'title': f"{content.title} ({content.release_date.year if content.release_date else 'N/A'}) - Watch Online",
            'description': description,
            'keywords': ', '.join(keywords[:10]),  # Limit keywords
            'og_title': content.title,
            'og_description': description,
            'og_image': self._get_image_url(content.poster_path, 'poster'),
            'og_type': 'video.movie' if content.content_type == 'movie' else 'video.tv_show',
            'twitter_card': 'summary_large_image',
            'canonical_url': f"/details/{content.slug if hasattr(content, 'slug') else self.slug_service.generate_slug(content.title, content.release_date.year if content.release_date else None, content.content_type)}",
            'structured_data': structured_data
        }
    
    # Helper methods for external API calls
    
    def _fetch_tmdb_details(self, tmdb_id: int, content_type: str) -> Optional[Dict]:
        """Fetch detailed information from TMDB"""
        try:
            endpoint = 'movie' if content_type == 'movie' else 'tv'
            url = f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}"
            params = {
                'api_key': self.tmdb_api_key,
                'append_to_response': 'keywords,alternative_titles'
            }
            
            response = self.http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching TMDB details: {e}")
        return None
    
    def _fetch_tmdb_credits(self, tmdb_id: int, content_type: str) -> Optional[Dict]:
        """Fetch cast and crew from TMDB"""
        try:
            endpoint = 'movie' if content_type == 'movie' else 'tv'
            url = f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}/credits"
            params = {'api_key': self.tmdb_api_key}
            
            response = self.http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching TMDB credits: {e}")
        return None
    
    def _fetch_tmdb_videos(self, tmdb_id: int, content_type: str) -> Optional[Dict]:
        """Fetch videos/trailers from TMDB"""
        try:
            endpoint = 'movie' if content_type == 'movie' else 'tv'
            url = f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}/videos"
            params = {'api_key': self.tmdb_api_key}
            
            response = self.http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching TMDB videos: {e}")
        return None
    
    def _fetch_tmdb_reviews(self, tmdb_id: int, content_type: str) -> Optional[Dict]:
        """Fetch reviews from TMDB"""
        try:
            endpoint = 'movie' if content_type == 'movie' else 'tv'
            url = f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}/reviews"
            params = {'api_key': self.tmdb_api_key}
            
            response = self.http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching TMDB reviews: {e}")
        return None
    
    def _fetch_tmdb_recommendations(self, tmdb_id: int, content_type: str) -> Optional[Dict]:
        """Fetch recommendations from TMDB"""
        try:
            endpoint = 'movie' if content_type == 'movie' else 'tv'
            url = f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}/recommendations"
            params = {'api_key': self.tmdb_api_key}
            
            response = self.http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching TMDB recommendations: {e}")
        return None
    
    def _fetch_tmdb_images(self, tmdb_id: int, content_type: str) -> Optional[Dict]:
        """Fetch images from TMDB"""
        try:
            endpoint = 'movie' if content_type == 'movie' else 'tv'
            url = f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}/images"
            params = {'api_key': self.tmdb_api_key}
            
            response = self.http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching TMDB images: {e}")
        return None
    
    def _fetch_tmdb_releases(self, tmdb_id: int) -> Optional[Dict]:
        """Fetch release dates and certifications from TMDB"""
        try:
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/release_dates"
            params = {'api_key': self.tmdb_api_key}
            
            response = self.http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching TMDB releases: {e}")
        return None
    
    def _fetch_tmdb_content_ratings(self, tmdb_id: int) -> Optional[Dict]:
        """Fetch TV content ratings from TMDB"""
        try:
            url = f"https://api.themoviedb.org/3/tv/{tmdb_id}/content_ratings"
            params = {'api_key': self.tmdb_api_key}
            
            response = self.http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching TMDB content ratings: {e}")
        return None
    
    def _fetch_omdb_data(self, imdb_id: str) -> Optional[Dict]:
        """Fetch data from OMDB API"""
        try:
            params = {
                'apikey': self.omdb_api_key,
                'i': imdb_id,
                'plot': 'full'
            }
            
            response = self.http_session.get('http://www.omdbapi.com/', params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('Response') == 'True':
                    return data
        except Exception as e:
            logger.error(f"Error fetching OMDB data: {e}")
        return None
    
    # Utility methods
    
    def _extract_themes(self, genres: List[str], overview: str) -> List[str]:
        """Extract themes from genres and overview"""
        themes = []
        
        theme_keywords = {
            'revenge': ['revenge', 'vengeance', 'payback'],
            'redemption': ['redemption', 'forgiveness', 'second chance'],
            'survival': ['survival', 'survive', 'wilderness'],
            'love': ['love', 'romance', 'relationship'],
            'family': ['family', 'father', 'mother', 'siblings'],
            'friendship': ['friendship', 'friends', 'companion'],
            'betrayal': ['betrayal', 'betray', 'deceive'],
            'justice': ['justice', 'law', 'court'],
            'war': ['war', 'battle', 'conflict'],
            'coming-of-age': ['growing up', 'teenage', 'youth']
        }
        
        overview_lower = overview.lower() if overview else ''
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in overview_lower for keyword in keywords):
                themes.append(theme)
        
        # Add genre-based themes
        genre_themes = {
            'Action': ['adventure', 'heroism'],
            'Drama': ['emotions', 'relationships'],
            'Thriller': ['suspense', 'mystery'],
            'Horror': ['fear', 'supernatural'],
            'Comedy': ['humor', 'satire'],
            'Romance': ['love', 'relationships']
        }
        
        for genre in genres:
            if genre in genre_themes:
                themes.extend(genre_themes[genre])
        
        return list(set(themes))[:5]  # Return top 5 unique themes
    
    def _determine_mood(self, genres: List[str]) -> str:
        """Determine content mood based on genres"""
        mood_map = {
            'Action': 'exciting',
            'Comedy': 'lighthearted',
            'Drama': 'emotional',
            'Horror': 'dark',
            'Thriller': 'suspenseful',
            'Romance': 'romantic',
            'Documentary': 'informative',
            'Animation': 'whimsical',
            'Crime': 'gritty',
            'Mystery': 'mysterious'
        }
        
        for genre in genres:
            if genre in mood_map:
                return mood_map[genre]
        
        return 'neutral'
    
    def _determine_advisories(self, rating: str, genres: List[str]) -> Dict[str, bool]:
        """Determine content advisories based on rating and genres"""
        advisories = {
            'violence': False,
            'profanity': False,
            'nudity': False,
            'substance': False,
            'frightening': False
        }
        
        # Rating-based advisories
        if rating in ['R', 'TV-MA', 'NC-17']:
            advisories['violence'] = True
            advisories['profanity'] = True
            advisories['nudity'] = True
        elif rating in ['PG-13', 'TV-14']:
            advisories['violence'] = True
            advisories['profanity'] = True
        
        # Genre-based advisories
        if 'Horror' in genres:
            advisories['frightening'] = True
            advisories['violence'] = True
        if 'Crime' in genres or 'Thriller' in genres:
            advisories['violence'] = True
        if 'War' in genres:
            advisories['violence'] = True
        
        return advisories
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of review text"""
        # Simple sentiment analysis (would use NLP library in production)
        positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'boring', 'disappointing']
        
        text_lower = text.lower() if text else ''
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _parse_currency(self, value: str) -> Optional[int]:
        """Parse currency string to integer"""
        if not value or value == 'N/A':
            return None
        
        # Remove currency symbols and commas
        cleaned = re.sub(r'[^\d]', '', value)
        
        try:
            return int(cleaned)
        except ValueError:
            return None
    
    def _record_view_analytics(self, content_id: int):
        """Record content view for analytics"""
        try:
            # This would record view analytics
            # Could integrate with analytics services
            logger.info(f"Recording view for content {content_id}")
        except Exception as e:
            logger.error(f"Error recording analytics: {e}")


# Integration function for Flask app - This MUST be at module level
def init_detail_service(app, db, cache, models, services):
    """
    Initialize detail service and register routes.
    
    Args:
        app: Flask application
        db: Database instance
        cache: Cache instance
        models: Dictionary of database models
        services: Dictionary of external services
    """
    # Create detail service instance
    detail_service = DetailService(db, cache, models, services)
    
    # Register routes
    @app.route('/api/details/<slug>', methods=['GET'])
    def get_content_details(slug):
        """Get content details by SEO-friendly slug"""
        try:
            details = detail_service.get_content_by_slug(slug)
            
            if not details:
                return jsonify({'error': 'Content not found'}), 404
            
            return jsonify(details), 200
            
        except Exception as e:
            logger.error(f"Error in details endpoint: {e}")
            return jsonify({'error': 'Failed to get content details'}), 500
    
    @app.route('/api/person/<int:person_id>', methods=['GET'])
    def get_person_details(person_id):
        """Get person (cast/crew) details"""
        try:
            person = detail_service.get_person_details(person_id)
            
            if not person:
                return jsonify({'error': 'Person not found'}), 404
            
            return jsonify(person), 200
            
        except Exception as e:
            logger.error(f"Error in person endpoint: {e}")
            return jsonify({'error': 'Failed to get person details'}), 500
    
    # Add slug generation endpoint for existing content
    @app.route('/api/admin/generate-slugs', methods=['POST'])
    def generate_slugs():
        """Generate slugs for existing content (admin only)"""
        try:
            # This would require admin authentication
            Content = models['Content']
            contents = db.session.query(Content).filter(
                or_(Content.slug == None, Content.slug == '')
            ).all()
            
            updated = 0
            for content in contents:
                content.slug = SlugService.generate_unique_slug(
                    db, Content,
                    content.title,
                    content.release_date.year if content.release_date else None,
                    content.content_type
                )
                updated += 1
            
            db.session.commit()
            
            return jsonify({
                'message': f'Generated slugs for {updated} content items'
            }), 200
            
        except Exception as e:
            logger.error(f"Error generating slugs: {e}")
            db.session.rollback()
            return jsonify({'error': 'Failed to generate slugs'}), 500
    
    return detail_service


# Export all classes and functions - This MUST be at module level
__all__ = [
    'DetailService',
    'SlugService',
    'init_detail_service',
    'ContentType',
    'MediaQuality',
    'ContentRating',
    'PersonDetail',
    'Review',
    'MediaAsset',
    'TrailerInfo'
]