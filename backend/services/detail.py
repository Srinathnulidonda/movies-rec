# backend/services/detail.py
import re
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from functools import wraps
from flask import jsonify, request, current_app
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_, or_, func, desc
from werkzeug.exceptions import NotFound, BadRequest
import unicodedata
from concurrent.futures import ThreadPoolExecutor
import requests

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Content type enumeration"""
    MOVIE = "movie"
    TV_SHOW = "tv_show"
    ANIME = "anime"

class MediaQuality(Enum):
    """Media quality enumeration"""
    SD = "SD"
    HD = "HD"
    FHD = "FHD"
    UHD_4K = "4K"
    UHD_8K = "8K"
    HDR = "HDR"
    DOLBY_VISION = "DOLBY_VISION"

@dataclass
class PersonDetails:
    """Person details data class"""
    id: int
    slug: str
    name: str
    role: str
    character: Optional[str] = None
    profile_image: Optional[str] = None
    biography: Optional[str] = None
    known_for: Optional[str] = None
    birthday: Optional[str] = None
    place_of_birth: Optional[str] = None
    popularity_score: float = 0.0
    social_media: Optional[Dict] = None
    filmography_count: int = 0

@dataclass
class TrailerInfo:
    """Trailer information data class"""
    id: str
    title: str
    url: str
    thumbnail: str
    quality: str
    duration: int
    autoplay: bool = False
    adaptive_quality: List[str] = None
    custom_controls: bool = True
    source: str = "youtube"

@dataclass
class ReviewData:
    """Review data class"""
    id: int
    user_id: int
    username: str
    avatar_url: Optional[str]
    rating: float
    title: str
    content: str
    spoiler_tagged: bool
    helpful_votes: int
    total_votes: int
    created_at: str
    updated_at: Optional[str]
    verified_purchase: bool = False

@dataclass
class RatingData:
    """Rating aggregation data class"""
    imdb_rating: Optional[float]
    imdb_votes: Optional[int]
    rotten_tomatoes_critics: Optional[int]
    rotten_tomatoes_audience: Optional[int]
    metacritic_score: Optional[int]
    tmdb_rating: Optional[float]
    mal_score: Optional[float]  # For anime
    composite_score: float
    user_rating: Optional[float]
    total_reviews: int

class SlugGenerator:
    """
    Advanced slug generation with multilingual support and duplicate handling
    """
    
    @staticmethod
    def create_slug(title: str, year: Optional[int] = None, content_type: Optional[str] = None) -> str:
        """
        Generate SEO-friendly slug from title
        Examples:
            "Inception" (2010) -> "inception-2010"
            "The Dark Knight" -> "the-dark-knight"
            "進撃の巨人" -> "shingeki-no-kyojin"
        """
        # Normalize unicode characters
        slug = unicodedata.normalize('NFKD', title)
        
        # Transliterate common non-ASCII characters
        transliterations = {
            'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'ā': 'a', 'ă': 'a', 'ą': 'a',
            'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e', 'ē': 'e', 'ė': 'e', 'ę': 'e',
            'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i', 'ī': 'i', 'į': 'i',
            'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'ō': 'o', 'ő': 'o',
            'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u', 'ū': 'u', 'ű': 'u',
            'ñ': 'n', 'ń': 'n', 'ň': 'n',
            'ç': 'c', 'ć': 'c', 'č': 'c',
            'ß': 'ss', 'ś': 's', 'š': 's',
            'ž': 'z', 'ź': 'z', 'ż': 'z',
            'æ': 'ae', 'œ': 'oe'
        }
        
        for char, replacement in transliterations.items():
            slug = slug.replace(char, replacement)
            slug = slug.replace(char.upper(), replacement.upper())
        
        # Convert to lowercase and replace spaces/special chars with hyphens
        slug = slug.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[-\s]+', '-', slug)
        slug = slug.strip('-')
        
        # Add year if provided
        if year:
            slug = f"{slug}-{year}"
        
        # Add content type suffix for disambiguation if needed
        if content_type and content_type != 'movie':
            type_suffix = {
                'tv_show': 'series',
                'anime': 'anime'
            }.get(content_type, content_type)
            slug = f"{slug}-{type_suffix}"
        
        # Ensure slug is not empty
        if not slug:
            slug = hashlib.md5(title.encode()).hexdigest()[:8]
        
        return slug[:100]  # Limit length for URLs
    
    @staticmethod
    def create_person_slug(name: str, birth_year: Optional[int] = None) -> str:
        """Generate slug for person/cast member"""
        slug = SlugGenerator.create_slug(name)
        if birth_year:
            slug = f"{slug}-{birth_year}"
        return slug
    
    @staticmethod
    def resolve_duplicate(slug: str, existing_slugs: List[str]) -> str:
        """Handle duplicate slugs by appending counter"""
        if slug not in existing_slugs:
            return slug
        
        counter = 1
        base_slug = slug
        while slug in existing_slugs:
            slug = f"{base_slug}-{counter}"
            counter += 1
        
        return slug

class DetailService:
    """
    Main detail service handling all detail page operations
    """
    
    def __init__(self, app=None, db=None, cache=None):
        self.app = app
        self.db = db
        self.cache = cache
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.tmdb_api_key = None
        self.omdb_api_key = None
        self.youtube_api_key = None
        
        if app:
            self.init_app(app, db)
    
    def init_app(self, app, db, cache=None):
        """Initialize service with Flask app"""
        self.app = app
        self.db = db
        self.cache = cache or app.config.get('cache')
        self.tmdb_api_key = app.config.get('TMDB_API_KEY')
        self.omdb_api_key = app.config.get('OMDB_API_KEY')
        self.youtube_api_key = app.config.get('YOUTUBE_API_KEY')
        
        # Register routes
        self._register_routes(app)
    
    def _register_routes(self, app):
        """Register all detail service routes"""
        
        @app.route('/api/details/<slug>', methods=['GET'])
        def get_content_details(slug):
            return self.get_full_details(slug)
        
        @app.route('/api/details/<slug>/reviews', methods=['GET'])
        def get_content_reviews(slug):
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 20, type=int)
            sort_by = request.args.get('sort_by', 'helpful')
            return self.get_reviews(slug, page, per_page, sort_by)
        
        @app.route('/api/details/<slug>/lists', methods=['POST'])
        def manage_lists(slug):
            return self.manage_user_lists(slug)
        
        @app.route('/api/details/<slug>/ratings', methods=['GET'])
        def get_ratings(slug):
            return self.get_composite_ratings(slug)
        
        @app.route('/api/details/<slug>/gallery', methods=['GET'])
        def get_gallery(slug):
            return self.get_media_gallery(slug)
        
        @app.route('/api/details/person/<person_slug>', methods=['GET'])
        def get_person_details(person_slug):
            return self.get_person_full_details(person_slug)
        
        @app.route('/api/details/person/<person_slug>/works', methods=['GET'])
        def get_person_works(person_slug):
            page = request.args.get('page', 1, type=int)
            return self.get_person_filmography(person_slug, page)
        
        @app.route('/api/details/person/<person_slug>/upcoming', methods=['GET'])
        def get_person_upcoming(person_slug):
            return self.get_person_upcoming_projects(person_slug)
    
    def _cache_key(self, prefix: str, *args) -> str:
        """Generate cache key"""
        key_parts = [prefix] + [str(arg) for arg in args]
        return ':'.join(key_parts)
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache"""
        if not self.cache:
            return None
        try:
            return self.cache.get(key)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def _set_cache(self, key: str, value: Any, timeout: int = 3600):
        """Set data in cache"""
        if not self.cache:
            return
        try:
            self.cache.set(key, value, timeout=timeout)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def _get_content_by_slug(self, slug: str) -> Optional[Any]:
        """Retrieve content from database by slug"""
        from app import Content  # Import here to avoid circular imports
        
        # Try exact match first
        content = Content.query.filter_by(slug=slug).first()
        if content:
            return content
        
        # Try variations (remove year, content type suffix)
        slug_parts = slug.rsplit('-', 1)
        if len(slug_parts) == 2 and slug_parts[1].isdigit():
            # Try without year
            base_slug = slug_parts[0]
            content = Content.query.filter_by(slug=base_slug).first()
            if content:
                return content
        
        # Try fuzzy match for similar slugs
        similar = Content.query.filter(
            Content.slug.like(f"{slug.split('-')[0]}%")
        ).first()
        
        return similar
    
    def _get_person_by_slug(self, slug: str) -> Optional[Any]:
        """Retrieve person from database by slug"""
        from app import Person  # Import here to avoid circular imports
        
        person = Person.query.filter_by(slug=slug).first()
        if not person:
            # Try without birth year
            base_slug = slug.rsplit('-', 1)[0]
            person = Person.query.filter_by(slug=base_slug).first()
        
        return person
    
    def _fetch_external_data(self, content_id: int, source: str) -> Optional[Dict]:
        """Fetch data from external APIs"""
        try:
            if source == 'tmdb' and self.tmdb_api_key:
                url = f"https://api.themoviedb.org/3/movie/{content_id}"
                params = {
                    'api_key': self.tmdb_api_key,
                    'append_to_response': 'credits,videos,images,reviews,similar'
                }
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    return response.json()
            
            elif source == 'omdb' and self.omdb_api_key:
                url = "http://www.omdbapi.com/"
                params = {
                    'apikey': self.omdb_api_key,
                    'i': content_id,
                    'plot': 'full'
                }
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    return response.json()
            
        except Exception as e:
            logger.error(f"External API fetch error ({source}): {e}")
        
        return None
    
    def _calculate_composite_score(self, ratings: Dict) -> float:
        """Calculate weighted composite score from multiple rating sources"""
        weights = {
            'imdb': 0.3,
            'rotten_tomatoes': 0.25,
            'metacritic': 0.2,
            'tmdb': 0.15,
            'user': 0.1
        }
        
        total_weight = 0
        weighted_sum = 0
        
        # IMDB (scale 0-10)
        if ratings.get('imdb_rating'):
            weighted_sum += ratings['imdb_rating'] * weights['imdb']
            total_weight += weights['imdb']
        
        # Rotten Tomatoes (scale 0-100, convert to 0-10)
        if ratings.get('rotten_tomatoes_critics'):
            rt_score = ratings['rotten_tomatoes_critics'] / 10
            weighted_sum += rt_score * weights['rotten_tomatoes']
            total_weight += weights['rotten_tomatoes']
        
        # Metacritic (scale 0-100, convert to 0-10)
        if ratings.get('metacritic_score'):
            mc_score = ratings['metacritic_score'] / 10
            weighted_sum += mc_score * weights['metacritic']
            total_weight += weights['metacritic']
        
        # TMDB (scale 0-10)
        if ratings.get('tmdb_rating'):
            weighted_sum += ratings['tmdb_rating'] * weights['tmdb']
            total_weight += weights['tmdb']
        
        # User rating (scale 0-10)
        if ratings.get('user_rating'):
            weighted_sum += ratings['user_rating'] * weights['user']
            total_weight += weights['user']
        
        if total_weight > 0:
            return round(weighted_sum / total_weight, 1)
        
        return 0.0
    
    def get_full_details(self, slug: str) -> Tuple[Dict, int]:
        """
        Get complete content details by slug
        Returns comprehensive detail page data
        """
        try:
            # Check cache first
            cache_key = self._cache_key('details', slug)
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                logger.info(f"Cache hit for details: {slug}")
                return jsonify(cached_data), 200
            
            # Get content from database
            content = self._get_content_by_slug(slug)
            if not content:
                return jsonify({'error': 'Content not found'}), 404
            
            # Build comprehensive response
            response_data = {
                'hero_section': self._build_hero_section(content),
                'trailer_player': self._build_trailer_section(content),
                'synopsis': self._build_synopsis_section(content),
                'cast_and_crew': self._build_cast_crew_section(content),
                'ratings': self._build_ratings_section(content),
                'metadata': self._build_metadata_section(content),
                'more_like_this': self._build_similar_content(content),
                'lists': self._build_lists_section(content),
                'reviews_preview': self._build_reviews_preview(content),
                'gallery_preview': self._build_gallery_preview(content),
                'seo': self._build_seo_data(content)
            }
            
            # Cache the response
            self._set_cache(cache_key, response_data, timeout=3600)
            
            # Track view analytics
            self._track_content_view(content.id, request)
            
            return jsonify(response_data), 200
            
        except Exception as e:
            logger.error(f"Error getting full details for {slug}: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    def _build_hero_section(self, content) -> Dict:
        """Build hero section data"""
        return {
            'title': content.title,
            'original_title': content.original_title,
            'tagline': content.tagline,
            'overview': content.overview,
            'poster': {
                'small': f"https://image.tmdb.org/t/p/w300{content.poster_path}",
                'medium': f"https://image.tmdb.org/t/p/w500{content.poster_path}",
                'large': f"https://image.tmdb.org/t/p/original{content.poster_path}"
            } if content.poster_path else None,
            'backdrop': {
                'small': f"https://image.tmdb.org/t/p/w780{content.backdrop_path}",
                'medium': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}",
                'large': f"https://image.tmdb.org/t/p/original{content.backdrop_path}"
            } if content.backdrop_path else None,
            'logo': content.logo_path,
            'content_type': content.content_type,
            'release_year': content.release_date.year if content.release_date else None
        }
    
    def _build_trailer_section(self, content) -> Dict:
        """Build trailer player section"""
        trailers = []
        
        # Get trailers from database or external API
        if content.youtube_trailer_id:
            trailers.append(TrailerInfo(
                id=content.youtube_trailer_id,
                title=f"{content.title} - Official Trailer",
                url=f"https://www.youtube.com/embed/{content.youtube_trailer_id}",
                thumbnail=f"https://img.youtube.com/vi/{content.youtube_trailer_id}/maxresdefault.jpg",
                quality="HD",
                duration=0,  # Would need YouTube API to get actual duration
                autoplay=True,
                adaptive_quality=["360p", "720p", "1080p"],
                custom_controls=True
            ))
        
        # Fetch additional trailers from TMDB if available
        if content.tmdb_id:
            external_data = self._fetch_external_data(content.tmdb_id, 'tmdb')
            if external_data and 'videos' in external_data:
                for video in external_data['videos'].get('results', [])[:5]:
                    if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                        trailers.append(TrailerInfo(
                            id=video['key'],
                            title=video['name'],
                            url=f"https://www.youtube.com/embed/{video['key']}",
                            thumbnail=f"https://img.youtube.com/vi/{video['key']}/maxresdefault.jpg",
                            quality=video.get('quality', 'HD'),
                            duration=0,
                            autoplay=False,
                            adaptive_quality=["360p", "720p", "1080p"],
                            custom_controls=True
                        ))
        
        return {
            'trailers': [asdict(trailer) for trailer in trailers],
            'autoplay_enabled': True,
            'default_quality': 'auto',
            'player_settings': {
                'controls': True,
                'show_info': False,
                'modestbranding': True,
                'rel': False
            }
        }
    
    def _build_synopsis_section(self, content) -> Dict:
        """Build synopsis section with content warnings"""
        return {
            'overview': content.overview,
            'plot_summary': content.plot_summary or content.overview,
            'content_warnings': json.loads(content.content_warnings) if content.content_warnings else [],
            'themes': json.loads(content.themes) if content.themes else [],
            'mood': content.mood,
            'pacing': content.pacing
        }
    
    def _build_cast_crew_section(self, content) -> Dict:
        """Build cast and crew section"""
        cast = []
        crew = []
        
        # Get from database relationships
        if hasattr(content, 'cast_members'):
            for member in content.cast_members[:20]:  # Limit to top 20
                person_slug = SlugGenerator.create_person_slug(member.name)
                cast.append({
                    'id': member.id,
                    'slug': person_slug,
                    'name': member.name,
                    'character': member.character,
                    'profile_image': member.profile_image,
                    'order': member.order
                })
        
        if hasattr(content, 'crew_members'):
            for member in content.crew_members[:10]:  # Limit to top 10
                person_slug = SlugGenerator.create_person_slug(member.name)
                crew.append({
                    'id': member.id,
                    'slug': person_slug,
                    'name': member.name,
                    'job': member.job,
                    'department': member.department,
                    'profile_image': member.profile_image
                })
        
        # Fetch from external API if needed
        if not cast and content.tmdb_id:
            external_data = self._fetch_external_data(content.tmdb_id, 'tmdb')
            if external_data and 'credits' in external_data:
                for person in external_data['credits'].get('cast', [])[:20]:
                    person_slug = SlugGenerator.create_person_slug(person['name'])
                    cast.append({
                        'id': person['id'],
                        'slug': person_slug,
                        'name': person['name'],
                        'character': person.get('character'),
                        'profile_image': f"https://image.tmdb.org/t/p/w185{person['profile_path']}" if person.get('profile_path') else None,
                        'order': person.get('order', 999)
                    })
                
                for person in external_data['credits'].get('crew', [])[:10]:
                    if person['job'] in ['Director', 'Producer', 'Writer', 'Composer']:
                        person_slug = SlugGenerator.create_person_slug(person['name'])
                        crew.append({
                            'id': person['id'],
                            'slug': person_slug,
                            'name': person['name'],
                            'job': person['job'],
                            'department': person['department'],
                            'profile_image': f"https://image.tmdb.org/t/p/w185{person['profile_path']}" if person.get('profile_path') else None
                        })
        
        return {
            'cast': sorted(cast, key=lambda x: x.get('order', 999)),
            'crew': crew,
            'director': next((c for c in crew if c.get('job') == 'Director'), None),
            'producers': [c for c in crew if c.get('job') == 'Producer'],
            'writers': [c for c in crew if c.get('job') in ['Writer', 'Screenplay']],
            'total_cast': len(cast),
            'total_crew': len(crew)
        }
    
    def _build_ratings_section(self, content) -> Dict:
        """Build comprehensive ratings section"""
        ratings_data = {
            'imdb_rating': content.imdb_rating,
            'imdb_votes': content.imdb_votes,
            'rotten_tomatoes_critics': content.rt_critics_score,
            'rotten_tomatoes_audience': content.rt_audience_score,
            'metacritic_score': content.metacritic_score,
            'tmdb_rating': content.rating,
            'mal_score': content.mal_score if content.content_type == 'anime' else None,
            'user_rating': content.user_rating,
            'total_reviews': content.review_count or 0
        }
        
        # Fetch additional ratings from external APIs if missing
        if not ratings_data['imdb_rating'] and content.imdb_id:
            omdb_data = self._fetch_external_data(content.imdb_id, 'omdb')
            if omdb_data:
                ratings_data['imdb_rating'] = float(omdb_data.get('imdbRating', 0)) if omdb_data.get('imdbRating') != 'N/A' else None
                ratings_data['imdb_votes'] = int(omdb_data.get('imdbVotes', '0').replace(',', '')) if omdb_data.get('imdbVotes') != 'N/A' else None
                
                # Parse Rotten Tomatoes
                for rating in omdb_data.get('Ratings', []):
                    if rating['Source'] == 'Rotten Tomatoes':
                        ratings_data['rotten_tomatoes_critics'] = int(rating['Value'].rstrip('%'))
                    elif rating['Source'] == 'Metacritic':
                        ratings_data['metacritic_score'] = int(rating['Value'].split('/')[0])
        
        # Calculate composite score
        composite_score = self._calculate_composite_score(ratings_data)
        
        return RatingData(
            **ratings_data,
            composite_score=composite_score
        ).__dict__
    
    def _build_metadata_section(self, content) -> Dict:
        """Build metadata section"""
        return {
            'genres': json.loads(content.genres) if content.genres else [],
            'runtime': content.runtime,
            'runtime_formatted': f"{content.runtime // 60}h {content.runtime % 60}m" if content.runtime else None,
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'release_date_formatted': content.release_date.strftime('%B %d, %Y') if content.release_date else None,
            'status': content.status,
            'original_language': content.original_language,
            'spoken_languages': json.loads(content.spoken_languages) if content.spoken_languages else [],
            'production_companies': json.loads(content.production_companies) if content.production_companies else [],
            'production_countries': json.loads(content.production_countries) if content.production_countries else [],
            'budget': content.budget,
            'revenue': content.revenue,
            'keywords': json.loads(content.keywords) if content.keywords else [],
            'content_rating': content.content_rating,
            'content_rating_reason': content.content_rating_reason,
            'quality': {
                '4k': content.is_4k or False,
                'hdr': content.is_hdr or False,
                'dolby_vision': content.is_dolby_vision or False,
                'dolby_atmos': content.is_dolby_atmos or False,
                'imax': content.is_imax or False
            },
            'availability': {
                'streaming': json.loads(content.streaming_platforms) if content.streaming_platforms else [],
                'rental': json.loads(content.rental_platforms) if content.rental_platforms else [],
                'purchase': json.loads(content.purchase_platforms) if content.purchase_platforms else []
            }
        }
    
    def _build_similar_content(self, content) -> List[Dict]:
        """Build similar/recommended content section"""
        similar = []
        
        # Get from database relationships
        if hasattr(content, 'similar_content'):
            for item in content.similar_content[:12]:
                similar.append({
                    'id': item.id,
                    'slug': item.slug,
                    'title': item.title,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{item.poster_path}" if item.poster_path else None,
                    'rating': item.rating,
                    'release_year': item.release_date.year if item.release_date else None,
                    'content_type': item.content_type
                })
        
        # Use ML recommendations if available
        if hasattr(self.app, 'ml_service'):
            ml_similar = self.app.ml_service.get_similar_content(content.id, limit=12)
            for item in ml_similar:
                if not any(s['id'] == item['id'] for s in similar):
                    similar.append(item)
        
        return similar[:12]  # Limit to 12 items
    
    def _build_lists_section(self, content) -> Dict:
        """Build user lists section"""
        user_id = request.headers.get('User-Id')  # Or get from JWT token
        
        return {
            'in_watchlist': False,  # Would check user's watchlist
            'in_favorites': False,  # Would check user's favorites
            'user_rating': None,  # Would get user's rating
            'lists_count': {
                'watchlist': content.watchlist_count or 0,
                'favorites': content.favorites_count or 0
            }
        }
    
    def _build_reviews_preview(self, content) -> Dict:
        """Build reviews preview section"""
        # Get top 3 reviews
        from app import Review  # Import here to avoid circular imports
        
        top_reviews = Review.query.filter_by(
            content_id=content.id
        ).order_by(
            desc(Review.helpful_votes)
        ).limit(3).all()
        
        reviews = []
        for review in top_reviews:
            reviews.append({
                'id': review.id,
                'username': review.user.username,
                'avatar_url': review.user.avatar_url,
                'rating': review.rating,
                'title': review.title,
                'content': review.content[:200] + '...' if len(review.content) > 200 else review.content,
                'spoiler_tagged': review.has_spoilers,
                'helpful_votes': review.helpful_votes,
                'created_at': review.created_at.isoformat()
            })
        
        return {
            'reviews': reviews,
            'total_reviews': content.review_count or 0,
            'average_rating': content.user_rating or 0
        }
    
    def _build_gallery_preview(self, content) -> Dict:
        """Build gallery preview section"""
        gallery = {
            'posters': [],
            'backdrops': [],
            'stills': []
        }
        
        # Get from TMDB API
        if content.tmdb_id:
            external_data = self._fetch_external_data(content.tmdb_id, 'tmdb')
            if external_data and 'images' in external_data:
                # Posters
                for image in external_data['images'].get('posters', [])[:4]:
                    gallery['posters'].append({
                        'thumbnail': f"https://image.tmdb.org/t/p/w185{image['file_path']}",
                        'full': f"https://image.tmdb.org/t/p/original{image['file_path']}",
                        'aspect_ratio': image['aspect_ratio'],
                        'width': image['width'],
                        'height': image['height']
                    })
                
                # Backdrops
                for image in external_data['images'].get('backdrops', [])[:4]:
                    gallery['backdrops'].append({
                        'thumbnail': f"https://image.tmdb.org/t/p/w300{image['file_path']}",
                        'full': f"https://image.tmdb.org/t/p/original{image['file_path']}",
                        'aspect_ratio': image['aspect_ratio'],
                        'width': image['width'],
                        'height': image['height']
                    })
        
        return gallery
    
    def _build_seo_data(self, content) -> Dict:
        """Build SEO metadata"""
        return {
            'title': f"{content.title} ({content.release_date.year if content.release_date else 'N/A'}) - Watch Online",
            'description': content.overview[:160] if content.overview else f"Watch {content.title} online. Stream or download in HD quality.",
            'keywords': ', '.join(json.loads(content.keywords)[:10]) if content.keywords else f"{content.title}, watch online, streaming, {content.content_type}",
            'canonical_url': f"/details/{content.slug}",
            'og_image': f"https://image.tmdb.org/t/p/original{content.backdrop_path or content.poster_path}" if (content.backdrop_path or content.poster_path) else None,
            'structured_data': {
                '@context': 'https://schema.org',
                '@type': 'Movie' if content.content_type == 'movie' else 'TVSeries',
                'name': content.title,
                'description': content.overview,
                'datePublished': content.release_date.isoformat() if content.release_date else None,
                'genre': json.loads(content.genres) if content.genres else [],
                'aggregateRating': {
                    '@type': 'AggregateRating',
                    'ratingValue': content.rating,
                    'ratingCount': content.vote_count
                } if content.rating else None
            }
        }
    
    def _track_content_view(self, content_id: int, request):
        """Track content view for analytics"""
        try:
            from app import AnonymousInteraction, db
            
            session_id = request.headers.get('Session-Id') or hashlib.md5(
                f"{request.remote_addr}{datetime.now()}".encode()
            ).hexdigest()
            
            interaction = AnonymousInteraction(
                session_id=session_id,
                content_id=content_id,
                interaction_type='view',
                ip_address=request.remote_addr,
                timestamp=datetime.utcnow()
            )
            db.session.add(interaction)
            db.session.commit()
        except Exception as e:
            logger.error(f"Error tracking view: {e}")
    
    def get_reviews(self, slug: str, page: int = 1, per_page: int = 20, sort_by: str = 'helpful') -> Tuple[Dict, int]:
        """Get paginated reviews for content"""
        try:
            content = self._get_content_by_slug(slug)
            if not content:
                return jsonify({'error': 'Content not found'}), 404
            
            from app import Review  # Import here
            
            # Build query
            query = Review.query.filter_by(content_id=content.id)
            
            # Apply sorting
            if sort_by == 'helpful':
                query = query.order_by(desc(Review.helpful_votes))
            elif sort_by == 'recent':
                query = query.order_by(desc(Review.created_at))
            elif sort_by == 'rating_high':
                query = query.order_by(desc(Review.rating))
            elif sort_by == 'rating_low':
                query = query.order_by(Review.rating)
            
            # Paginate
            paginated = query.paginate(page=page, per_page=per_page, error_out=False)
            
            reviews = []
            for review in paginated.items:
                reviews.append(ReviewData(
                    id=review.id,
                    user_id=review.user_id,
                    username=review.user.username,
                    avatar_url=review.user.avatar_url,
                    rating=review.rating,
                    title=review.title,
                    content=review.content,
                    spoiler_tagged=review.has_spoilers,
                    helpful_votes=review.helpful_votes,
                    total_votes=review.total_votes,
                    created_at=review.created_at.isoformat(),
                    updated_at=review.updated_at.isoformat() if review.updated_at else None,
                    verified_purchase=review.verified_purchase
                ).__dict__)
            
            return jsonify({
                'reviews': reviews,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total_pages': paginated.pages,
                    'total_items': paginated.total,
                    'has_next': paginated.has_next,
                    'has_prev': paginated.has_prev
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Error getting reviews: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    def manage_user_lists(self, slug: str) -> Tuple[Dict, int]:
        """Add/remove content from user lists"""
        try:
            # Check authentication
            user_id = request.headers.get('User-Id')
            if not user_id:
                return jsonify({'error': 'Authentication required'}), 401
            
            content = self._get_content_by_slug(slug)
            if not content:
                return jsonify({'error': 'Content not found'}), 404
            
            data = request.get_json()
            action = data.get('action')  # 'add' or 'remove'
            list_type = data.get('list_type')  # 'watchlist' or 'favorites'
            
            if action not in ['add', 'remove'] or list_type not in ['watchlist', 'favorites']:
                return jsonify({'error': 'Invalid action or list type'}), 400
            
            from app import UserList, db  # Import here
            
            if action == 'add':
                # Check if already exists
                existing = UserList.query.filter_by(
                    user_id=user_id,
                    content_id=content.id,
                    list_type=list_type
                ).first()
                
                if existing:
                    return jsonify({'message': 'Already in list'}), 200
                
                # Add to list
                user_list = UserList(
                    user_id=user_id,
                    content_id=content.id,
                    list_type=list_type,
                    added_at=datetime.utcnow()
                )
                db.session.add(user_list)
                
                # Update counters
                if list_type == 'watchlist':
                    content.watchlist_count = (content.watchlist_count or 0) + 1
                else:
                    content.favorites_count = (content.favorites_count or 0) + 1
                
            else:  # remove
                user_list = UserList.query.filter_by(
                    user_id=user_id,
                    content_id=content.id,
                    list_type=list_type
                ).first()
                
                if not user_list:
                    return jsonify({'message': 'Not in list'}), 200
                
                db.session.delete(user_list)
                
                # Update counters
                if list_type == 'watchlist':
                    content.watchlist_count = max(0, (content.watchlist_count or 1) - 1)
                else:
                    content.favorites_count = max(0, (content.favorites_count or 1) - 1)
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'action': action,
                'list_type': list_type,
                'content_id': content.id
            }), 200
            
        except Exception as e:
            logger.error(f"Error managing user lists: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    def get_composite_ratings(self, slug: str) -> Tuple[Dict, int]:
        """Get all ratings for content"""
        try:
            content = self._get_content_by_slug(slug)
            if not content:
                return jsonify({'error': 'Content not found'}), 404
            
            ratings = self._build_ratings_section(content)
            
            # Add rating distribution
            from app import Review, db
            
            distribution = db.session.query(
                Review.rating,
                func.count(Review.id)
            ).filter_by(
                content_id=content.id
            ).group_by(Review.rating).all()
            
            ratings['distribution'] = {
                '5_stars': 0,
                '4_stars': 0,
                '3_stars': 0,
                '2_stars': 0,
                '1_star': 0
            }
            
            for rating, count in distribution:
                if rating >= 4.5:
                    ratings['distribution']['5_stars'] += count
                elif rating >= 3.5:
                    ratings['distribution']['4_stars'] += count
                elif rating >= 2.5:
                    ratings['distribution']['3_stars'] += count
                elif rating >= 1.5:
                    ratings['distribution']['2_stars'] += count
                else:
                    ratings['distribution']['1_star'] += count
            
            return jsonify(ratings), 200
            
        except Exception as e:
            logger.error(f"Error getting ratings: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    def get_media_gallery(self, slug: str) -> Tuple[Dict, int]:
        """Get full media gallery for content"""
        try:
            content = self._get_content_by_slug(slug)
            if not content:
                return jsonify({'error': 'Content not found'}), 404
            
            gallery = {
                'posters': [],
                'backdrops': [],
                'stills': [],
                'behind_the_scenes': [],
                'logos': []
            }
            
            # Fetch from TMDB
            if content.tmdb_id:
                external_data = self._fetch_external_data(content.tmdb_id, 'tmdb')
                if external_data and 'images' in external_data:
                    # Process posters
                    for image in external_data['images'].get('posters', []):
                        gallery['posters'].append({
                            'id': hashlib.md5(image['file_path'].encode()).hexdigest(),
                            'thumbnail': f"https://image.tmdb.org/t/p/w342{image['file_path']}",
                            'medium': f"https://image.tmdb.org/t/p/w500{image['file_path']}",
                            'full': f"https://image.tmdb.org/t/p/original{image['file_path']}",
                            'aspect_ratio': image['aspect_ratio'],
                            'width': image['width'],
                            'height': image['height'],
                            'language': image.get('iso_639_1'),
                            'vote_average': image.get('vote_average', 0)
                        })
                    
                    # Process backdrops
                    for image in external_data['images'].get('backdrops', []):
                        gallery['backdrops'].append({
                            'id': hashlib.md5(image['file_path'].encode()).hexdigest(),
                            'thumbnail': f"https://image.tmdb.org/t/p/w780{image['file_path']}",
                            'medium': f"https://image.tmdb.org/t/p/w1280{image['file_path']}",
                            'full': f"https://image.tmdb.org/t/p/original{image['file_path']}",
                            'aspect_ratio': image['aspect_ratio'],
                            'width': image['width'],
                            'height': image['height'],
                            'language': image.get('iso_639_1'),
                            'vote_average': image.get('vote_average', 0)
                        })
                    
                    # Process logos
                    for image in external_data['images'].get('logos', []):
                        gallery['logos'].append({
                            'id': hashlib.md5(image['file_path'].encode()).hexdigest(),
                            'thumbnail': f"https://image.tmdb.org/t/p/w185{image['file_path']}",
                            'medium': f"https://image.tmdb.org/t/p/w300{image['file_path']}",
                            'full': f"https://image.tmdb.org/t/p/original{image['file_path']}",
                            'aspect_ratio': image['aspect_ratio'],
                            'width': image['width'],
                            'height': image['height'],
                            'language': image.get('iso_639_1')
                        })
            
            # Sort by vote average
            for key in ['posters', 'backdrops']:
                gallery[key] = sorted(
                    gallery[key],
                    key=lambda x: x.get('vote_average', 0),
                    reverse=True
                )
            
            return jsonify({
                'gallery': gallery,
                'total_images': sum(len(gallery[key]) for key in gallery),
                'content_slug': slug
            }), 200
            
        except Exception as e:
            logger.error(f"Error getting gallery: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    def get_person_full_details(self, person_slug: str) -> Tuple[Dict, int]:
        """Get complete person/cast member details"""
        try:
            person = self._get_person_by_slug(person_slug)
            if not person:
                return jsonify({'error': 'Person not found'}), 404
            
            # Build person details
            details = PersonDetails(
                id=person.id,
                slug=person.slug,
                name=person.name,
                role=person.primary_role,
                profile_image=person.profile_image,
                biography=person.biography,
                known_for=person.known_for_department,
                birthday=person.birthday.isoformat() if person.birthday else None,
                place_of_birth=person.place_of_birth,
                popularity_score=person.popularity or 0,
                social_media={
                    'twitter': person.twitter_id,
                    'instagram': person.instagram_id,
                    'facebook': person.facebook_id,
                    'imdb': person.imdb_id
                } if any([person.twitter_id, person.instagram_id, person.facebook_id, person.imdb_id]) else None,
                filmography_count=person.filmography_count or 0
            )
            
            # Get career highlights
            from app import Content, CastMember, CrewMember
            
            # Get notable works
            notable_works = []
            
            # As cast
            cast_works = db.session.query(Content).join(
                CastMember
            ).filter(
                CastMember.person_id == person.id
            ).order_by(
                desc(Content.popularity)
            ).limit(10).all()
            
            for work in cast_works:
                notable_works.append({
                    'id': work.id,
                    'slug': work.slug,
                    'title': work.title,
                    'year': work.release_date.year if work.release_date else None,
                    'role': 'Actor',
                    'character': None,  # Would need to join to get character name
                    'poster_path': f"https://image.tmdb.org/t/p/w185{work.poster_path}" if work.poster_path else None,
                    'rating': work.rating
                })
            
            # As crew
            crew_works = db.session.query(Content).join(
                CrewMember
            ).filter(
                CrewMember.person_id == person.id
            ).order_by(
                desc(Content.popularity)
            ).limit(10).all()
            
            for work in crew_works:
                notable_works.append({
                    'id': work.id,
                    'slug': work.slug,
                    'title': work.title,
                    'year': work.release_date.year if work.release_date else None,
                    'role': None,  # Would need to join to get job
                    'poster_path': f"https://image.tmdb.org/t/p/w185{work.poster_path}" if work.poster_path else None,
                    'rating': work.rating
                })
            
            # Remove duplicates and sort by rating
            seen = set()
            unique_works = []
            for work in notable_works:
                if work['id'] not in seen:
                    seen.add(work['id'])
                    unique_works.append(work)
            
            notable_works = sorted(unique_works, key=lambda x: x.get('rating', 0), reverse=True)[:10]
            
            return jsonify({
                'person': asdict(details),
                'notable_works': notable_works,
                'statistics': {
                    'total_movies': person.movie_count or 0,
                    'total_tv_shows': person.tv_count or 0,
                    'years_active': f"{person.career_start_year}-present" if person.career_start_year else None,
                    'awards_count': person.awards_count or 0
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Error getting person details: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    def get_person_filmography(self, person_slug: str, page: int = 1) -> Tuple[Dict, int]:
        """Get person's complete filmography"""
        try:
            person = self._get_person_by_slug(person_slug)
            if not person:
                return jsonify({'error': 'Person not found'}), 404
            
            from app import Content, CastMember, CrewMember
            
            per_page = 20
            
            # Get all works
            all_works = []
            
            # Cast works
            cast_query = db.session.query(
                Content,
                CastMember.character,
                CastMember.order
            ).join(
                CastMember
            ).filter(
                CastMember.person_id == person.id
            )
            
            for content, character, order in cast_query.all():
                all_works.append({
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'year': content.release_date.year if content.release_date else None,
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'role_type': 'cast',
                    'role': 'Actor',
                    'character': character,
                    'billing_order': order,
                    'poster_path': f"https://image.tmdb.org/t/p/w185{content.poster_path}" if content.poster_path else None,
                    'rating': content.rating,
                    'popularity': content.popularity
                })
            
            # Crew works
            crew_query = db.session.query(
                Content,
                CrewMember.job,
                CrewMember.department
            ).join(
                CrewMember
            ).filter(
                CrewMember.person_id == person.id
            )
            
            for content, job, department in crew_query.all():
                # Check if already added as cast
                if not any(w['id'] == content.id for w in all_works):
                    all_works.append({
                        'id': content.id,
                        'slug': content.slug,
                        'title': content.title,
                        'content_type': content.content_type,
                        'year': content.release_date.year if content.release_date else None,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'role_type': 'crew',
                        'role': job,
                        'department': department,
                        'poster_path': f"https://image.tmdb.org/t/p/w185{content.poster_path}" if content.poster_path else None,
                        'rating': content.rating,
                        'popularity': content.popularity
                    })
            
            # Sort by release date (newest first)
            all_works.sort(key=lambda x: x.get('release_date', ''), reverse=True)
            
            # Paginate
            total_items = len(all_works)
            total_pages = (total_items + per_page - 1) // per_page
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_works = all_works[start_idx:end_idx]
            
            # Group by year
            works_by_year = {}
            for work in paginated_works:
                year = work.get('year', 'Unknown')
                if year not in works_by_year:
                    works_by_year[year] = []
                works_by_year[year].append(work)
            
            return jsonify({
                'filmography': paginated_works,
                'grouped_by_year': works_by_year,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total_pages': total_pages,
                    'total_items': total_items,
                    'has_next': page < total_pages,
                    'has_prev': page > 1
                },
                'statistics': {
                    'total_works': total_items,
                    'as_cast': sum(1 for w in all_works if w['role_type'] == 'cast'),
                    'as_crew': sum(1 for w in all_works if w['role_type'] == 'crew'),
                    'movies': sum(1 for w in all_works if w['content_type'] == 'movie'),
                    'tv_shows': sum(1 for w in all_works if w['content_type'] == 'tv_show'),
                    'anime': sum(1 for w in all_works if w['content_type'] == 'anime')
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Error getting filmography: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    def get_person_upcoming_projects(self, person_slug: str) -> Tuple[Dict, int]:
        """Get person's upcoming projects"""
        try:
            person = self._get_person_by_slug(person_slug)
            if not person:
                return jsonify({'error': 'Person not found'}), 404
            
            from app import Content, CastMember, CrewMember
            
            # Get upcoming projects (not yet released)
            today = datetime.now().date()
            
            upcoming = []
            
            # Cast in upcoming
            cast_upcoming = db.session.query(
                Content,
                CastMember.character
            ).join(
                CastMember
            ).filter(
                CastMember.person_id == person.id,
                Content.release_date > today
            ).order_by(
                Content.release_date
            ).limit(10).all()
            
            for content, character in cast_upcoming:
                upcoming.append({
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'release_date': content.release_date.isoformat(),
                    'days_until_release': (content.release_date - today).days,
                    'role': 'Actor',
                    'character': character,
                    'poster_path': f"https://image.tmdb.org/t/p/w185{content.poster_path}" if content.poster_path else None,
                    'status': content.status,
                    'in_production': content.status == 'In Production'
                })
            
            # Crew in upcoming
            crew_upcoming = db.session.query(
                Content,
                CrewMember.job
            ).join(
                CrewMember
            ).filter(
                CrewMember.person_id == person.id,
                Content.release_date > today
            ).order_by(
                Content.release_date
            ).limit(10).all()
            
            for content, job in crew_upcoming:
                # Avoid duplicates
                if not any(u['id'] == content.id for u in upcoming):
                    upcoming.append({
                        'id': content.id,
                        'slug': content.slug,
                        'title': content.title,
                        'content_type': content.content_type,
                        'release_date': content.release_date.isoformat(),
                        'days_until_release': (content.release_date - today).days,
                        'role': job,
                        'poster_path': f"https://image.tmdb.org/t/p/w185{content.poster_path}" if content.poster_path else None,
                        'status': content.status,
                        'in_production': content.status == 'In Production'
                    })
            
            # Sort by release date
            upcoming.sort(key=lambda x: x['release_date'])
            
            # Group by status
            in_production = [p for p in upcoming if p.get('in_production')]
            post_production = [p for p in upcoming if p.get('status') == 'Post Production']
            announced = [p for p in upcoming if p.get('status') == 'Announced']
            
            return jsonify({
                'upcoming_projects': upcoming[:20],
                'grouped_by_status': {
                    'in_production': in_production,
                    'post_production': post_production,
                    'announced': announced
                },
                'total_upcoming': len(upcoming),
                'next_release': upcoming[0] if upcoming else None
            }), 200
            
        except Exception as e:
            logger.error(f"Error getting upcoming projects: {e}")
            return jsonify({'error': 'Internal server error'}), 500

# Initialize service as a singleton
detail_service = DetailService()