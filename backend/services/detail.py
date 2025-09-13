# backend/services/detail.py
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
import re
import unicodedata
from functools import wraps
from slugify import slugify
import hashlib
from sqlalchemy import and_, or_, func
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# Create details blueprint
details_bp = Blueprint('details', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Will be initialized by main app
db = None
cache = None
models = {}
services = {}

def init_details(flask_app, database, cache_instance, app_models, app_services):
    """Initialize details module with app context and models"""
    global db, cache, models, services
    
    db = database
    cache = cache_instance
    models = app_models
    services = app_services

class SlugManager:
    """Handles slug generation and resolution"""
    
    @staticmethod
    def generate_slug(title: str, year: Optional[int] = None, content_type: str = 'movie') -> str:
        """Generate SEO-friendly slug from title"""
        try:
            # Normalize unicode characters
            title_normalized = unicodedata.normalize('NFKD', title)
            title_ascii = title_normalized.encode('ascii', 'ignore').decode('ascii')
            
            # Create base slug
            base_slug = slugify(title_ascii, lowercase=True, separator='-')
            
            # Remove common articles for better SEO
            articles = ['the', 'a', 'an', 'le', 'la', 'les', 'un', 'une', 'der', 'die', 'das']
            slug_parts = base_slug.split('-')
            if slug_parts and slug_parts[0] in articles:
                base_slug = '-'.join(slug_parts[1:])
            
            # Add year if provided
            if year:
                slug = f"{base_slug}-{year}"
            else:
                slug = base_slug
            
            # Add content type prefix for TV shows and anime
            if content_type == 'tv':
                slug = f"tv-{slug}"
            elif content_type == 'anime':
                slug = f"anime-{slug}"
            
            # Ensure slug is not too long (max 100 chars)
            if len(slug) > 100:
                slug = slug[:97] + '...'
            
            return slug
            
        except Exception as e:
            logger.error(f"Slug generation error: {e}")
            # Fallback to hash-based slug
            return hashlib.md5(title.encode()).hexdigest()[:10]
    
    @staticmethod
    def resolve_slug(slug: str) -> Optional[Dict]:
        """Resolve slug to content"""
        try:
            Content = models.get('Content')
            
            # Try exact slug match first
            content = Content.query.filter_by(slug=slug).first()
            
            if not content:
                # Try variations (remove prefixes)
                base_slug = slug
                for prefix in ['tv-', 'anime-', 'movie-']:
                    if slug.startswith(prefix):
                        base_slug = slug[len(prefix):]
                        break
                
                # Search with LIKE for partial matches
                content = Content.query.filter(
                    Content.slug.like(f'%{base_slug}%')
                ).first()
            
            if content:
                return {
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'slug': content.slug
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Slug resolution error: {e}")
            return None
    
    @staticmethod
    def ensure_unique_slug(slug: str, content_id: Optional[int] = None) -> str:
        """Ensure slug is unique, append counter if needed"""
        Content = models.get('Content')
        original_slug = slug
        counter = 1
        
        while True:
            query = Content.query.filter_by(slug=slug)
            if content_id:
                query = query.filter(Content.id != content_id)
            
            if not query.first():
                return slug
            
            slug = f"{original_slug}-{counter}"
            counter += 1
            
            if counter > 100:  # Prevent infinite loop
                return f"{original_slug}-{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]}"

class DetailService:
    """Comprehensive content detail service"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.tmdb_base_url = 'https://api.themoviedb.org/3'
        self.omdb_base_url = 'http://www.omdbapi.com/'
        self.jikan_base_url = 'https://api.jikan.moe/v4'
        self.youtube_base_url = 'https://www.googleapis.com/youtube/v3'
    
    def get_content_details(self, slug: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive content details by slug"""
        try:
            # Resolve slug to content
            slug_data = SlugManager.resolve_slug(slug)
            if not slug_data:
                return None
            
            content_id = slug_data['id']
            
            # Get base content from database
            Content = models.get('Content')
            content = Content.query.get(content_id)
            
            if not content:
                return None
            
            # Build comprehensive response using concurrent requests
            futures = []
            
            with self.executor as executor:
                # Fetch all data concurrently
                futures.append(executor.submit(self._get_base_info, content))
                futures.append(executor.submit(self._get_trailer_info, content))
                futures.append(executor.submit(self._get_cast_crew, content))
                futures.append(executor.submit(self._get_ratings_reviews, content))
                futures.append(executor.submit(self._get_similar_content, content))
                futures.append(executor.submit(self._get_gallery, content))
                futures.append(executor.submit(self._get_metadata, content))
                futures.append(executor.submit(self._get_streaming_info, content))
                
                if user_id:
                    futures.append(executor.submit(self._get_user_status, content_id, user_id))
            
            # Collect results
            results = {}
            result_keys = [
                'base_info', 'trailer', 'cast_crew', 'ratings_reviews',
                'similar', 'gallery', 'metadata', 'streaming', 'user_status'
            ]
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    if i < len(result_keys):
                        results[result_keys[i]] = future.result()
                except Exception as e:
                    logger.error(f"Error fetching {result_keys[i] if i < len(result_keys) else 'unknown'}: {e}")
            
            # Combine all results
            response = self._build_response(content, results, user_id)
            
            # Record view interaction
            if user_id:
                self._record_interaction(user_id, content_id, 'view')
            
            return response
            
        except Exception as e:
            logger.error(f"Get content details error: {e}")
            return None
    
    def _get_base_info(self, content) -> Dict:
        """Get base content information"""
        try:
            base_info = {
                'id': content.id,
                'slug': content.slug,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'overview': content.overview,
                'tagline': getattr(content, 'tagline', ''),
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'runtime': content.runtime,
                'status': getattr(content, 'status', 'Released'),
                'language': json.loads(content.languages or '[]'),
                'genres': json.loads(content.genres or '[]'),
                'keywords': json.loads(getattr(content, 'keywords', '[]')),
                'poster_url': self._get_image_url(content.poster_path, 'poster'),
                'backdrop_url': self._get_image_url(content.backdrop_path, 'backdrop'),
                'logo_url': self._get_image_url(getattr(content, 'logo_path', None), 'logo')
            }
            
            # Add series-specific info
            if content.content_type == 'tv':
                base_info.update({
                    'seasons': getattr(content, 'seasons', 1),
                    'episodes': getattr(content, 'episodes', 0),
                    'networks': json.loads(getattr(content, 'networks', '[]')),
                    'next_episode': getattr(content, 'next_episode', None),
                    'last_episode': getattr(content, 'last_episode', None)
                })
            
            # Add anime-specific info
            if content.content_type == 'anime':
                base_info.update({
                    'anime_genres': json.loads(content.anime_genres or '[]'),
                    'mal_id': content.mal_id,
                    'episodes': getattr(content, 'episodes', 0),
                    'airing_status': getattr(content, 'airing_status', 'Finished'),
                    'studios': json.loads(getattr(content, 'studios', '[]'))
                })
            
            return base_info
            
        except Exception as e:
            logger.error(f"Base info error: {e}")
            return {}
    
    def _get_trailer_info(self, content) -> Dict:
        """Get trailer and video information"""
        try:
            trailer_info = {
                'youtube_id': content.youtube_trailer_id,
                'youtube_url': f"https://www.youtube.com/watch?v={content.youtube_trailer_id}" if content.youtube_trailer_id else None,
                'embed_url': f"https://www.youtube.com/embed/{content.youtube_trailer_id}?autoplay=0&rel=0&showinfo=0" if content.youtube_trailer_id else None,
                'videos': []
            }
            
            # Get additional videos from TMDB
            if content.tmdb_id:
                videos = self._fetch_tmdb_videos(content.tmdb_id, content.content_type)
                trailer_info['videos'] = videos
            
            # Get adaptive quality options
            if content.youtube_trailer_id:
                trailer_info['quality_options'] = ['auto', '1080p', '720p', '480p', '360p']
                trailer_info['playback_settings'] = {
                    'autoplay': False,
                    'controls': True,
                    'loop': False,
                    'muted': False,
                    'pip': True  # Picture-in-picture
                }
            
            return trailer_info
            
        except Exception as e:
            logger.error(f"Trailer info error: {e}")
            return {}
    
    def _get_cast_crew(self, content) -> Dict:
        """Get cast and crew information"""
        try:
            cast_crew = {
                'cast': [],
                'crew': [],
                'directors': [],
                'writers': [],
                'producers': []
            }
            
            # Fetch from TMDB
            if content.tmdb_id:
                tmdb_credits = self._fetch_tmdb_credits(content.tmdb_id, content.content_type)
                
                # Process cast
                for person in tmdb_credits.get('cast', [])[:20]:
                    cast_crew['cast'].append({
                        'id': person.get('id'),
                        'name': person.get('name'),
                        'character': person.get('character'),
                        'profile_image': self._get_image_url(person.get('profile_path'), 'profile'),
                        'order': person.get('order'),
                        'known_for': person.get('known_for_department'),
                        'popularity': person.get('popularity', 0)
                    })
                
                # Process crew
                for person in tmdb_credits.get('crew', []):
                    crew_member = {
                        'id': person.get('id'),
                        'name': person.get('name'),
                        'job': person.get('job'),
                        'department': person.get('department'),
                        'profile_image': self._get_image_url(person.get('profile_path'), 'profile')
                    }
                    
                    # Categorize by role
                    if person.get('job') == 'Director':
                        cast_crew['directors'].append(crew_member)
                    elif person.get('department') == 'Writing':
                        cast_crew['writers'].append(crew_member)
                    elif person.get('job') in ['Producer', 'Executive Producer']:
                        cast_crew['producers'].append(crew_member)
                    
                    cast_crew['crew'].append(crew_member)
            
            # For anime, get voice actors
            if content.content_type == 'anime' and content.mal_id:
                anime_cast = self._fetch_anime_cast(content.mal_id)
                cast_crew['voice_actors'] = anime_cast
            
            return cast_crew
            
        except Exception as e:
            logger.error(f"Cast crew error: {e}")
            return {}
    
    def _get_ratings_reviews(self, content) -> Dict:
        """Get ratings and reviews"""
        try:
            ratings = {
                'tmdb_rating': content.rating,
                'tmdb_votes': content.vote_count,
                'imdb_rating': None,
                'imdb_votes': None,
                'rotten_tomatoes': None,
                'metacritic': None,
                'mal_score': None,
                'composite_score': None,
                'user_reviews': [],
                'critic_reviews': [],
                'content_rating': getattr(content, 'content_rating', 'NR'),
                'advisories': []
            }
            
            # Get IMDB rating
            if content.imdb_id:
                omdb_data = self._fetch_omdb_data(content.imdb_id)
                if omdb_data:
                    ratings['imdb_rating'] = omdb_data.get('imdbRating')
                    ratings['imdb_votes'] = omdb_data.get('imdbVotes')
                    ratings['rotten_tomatoes'] = omdb_data.get('Ratings', [{}])[1].get('Value') if len(omdb_data.get('Ratings', [])) > 1 else None
                    ratings['metacritic'] = omdb_data.get('Metascore')
                    ratings['content_rating'] = omdb_data.get('Rated', 'NR')
            
            # Get MAL score for anime
            if content.content_type == 'anime' and content.mal_id:
                mal_data = self._fetch_mal_data(content.mal_id)
                if mal_data:
                    ratings['mal_score'] = mal_data.get('score')
                    ratings['mal_votes'] = mal_data.get('scored_by')
            
            # Calculate composite score
            scores = []
            if ratings['tmdb_rating']:
                scores.append(ratings['tmdb_rating'])
            if ratings['imdb_rating']:
                try:
                    scores.append(float(ratings['imdb_rating']))
                except:
                    pass
            if ratings['mal_score']:
                scores.append(ratings['mal_score'])
            
            if scores:
                ratings['composite_score'] = round(sum(scores) / len(scores), 1)
            
            # Get user reviews from database
            UserInteraction = models.get('UserInteraction')
            reviews = UserInteraction.query.filter_by(
                content_id=content.id,
                interaction_type='review'
            ).limit(10).all()
            
            for review in reviews:
                User = models.get('User')
                user = User.query.get(review.user_id)
                if user:
                    ratings['user_reviews'].append({
                        'user': user.username,
                        'rating': review.rating,
                        'review': getattr(review, 'review_text', ''),
                        'date': review.timestamp.isoformat(),
                        'helpful_count': getattr(review, 'helpful_count', 0),
                        'spoiler': getattr(review, 'has_spoiler', False)
                    })
            
            # Content advisories
            ratings['advisories'] = self._get_content_advisories(content)
            
            return ratings
            
        except Exception as e:
            logger.error(f"Ratings reviews error: {e}")
            return {}
    
    def _get_similar_content(self, content) -> Dict:
        """Get similar and recommended content"""
        try:
            similar = {
                'more_like_this': [],
                'same_genre': [],
                'same_director': [],
                'same_franchise': [],
                'recommended': []
            }
            
            Content = models.get('Content')
            
            # Get similar by genre
            if content.genres:
                genres = json.loads(content.genres)
                if genres:
                    similar_query = Content.query.filter(
                        Content.id != content.id,
                        Content.content_type == content.content_type,
                        Content.genres.contains(genres[0])
                    ).limit(20)
                    
                    for item in similar_query:
                        similar['same_genre'].append(self._format_content_card(item))
            
            # Get TMDB recommendations
            if content.tmdb_id:
                tmdb_similar = self._fetch_tmdb_similar(content.tmdb_id, content.content_type)
                for item in tmdb_similar[:10]:
                    similar['more_like_this'].append({
                        'tmdb_id': item.get('id'),
                        'title': item.get('title') or item.get('name'),
                        'poster_url': self._get_image_url(item.get('poster_path'), 'poster'),
                        'rating': item.get('vote_average'),
                        'overview': item.get('overview', '')[:150] + '...',
                        'release_date': item.get('release_date') or item.get('first_air_date')
                    })
            
            # Use advanced similarity algorithms
            orchestrator = services.get('recommendation_orchestrator')
            if orchestrator:
                content_pool = Content.query.filter(
                    Content.id != content.id,
                    Content.content_type == content.content_type
                ).limit(500).all()
                
                ultra_similar = orchestrator.get_ultra_similar_content(
                    content.id,
                    content_pool,
                    limit=15,
                    strict_mode=True,
                    min_similarity=0.5
                )
                
                for item in ultra_similar:
                    similar['recommended'].append(item)
            
            return similar
            
        except Exception as e:
            logger.error(f"Similar content error: {e}")
            return {}
    
    def _get_gallery(self, content) -> Dict:
        """Get images and gallery"""
        try:
            gallery = {
                'posters': [],
                'backdrops': [],
                'stills': [],
                'logos': []
            }
            
            # Get images from TMDB
            if content.tmdb_id:
                images = self._fetch_tmdb_images(content.tmdb_id, content.content_type)
                
                # Process posters
                for img in images.get('posters', [])[:10]:
                    gallery['posters'].append({
                        'url': self._get_image_url(img.get('file_path'), 'poster'),
                        'thumbnail': self._get_image_url(img.get('file_path'), 'thumbnail'),
                        'width': img.get('width'),
                        'height': img.get('height'),
                        'aspect_ratio': img.get('aspect_ratio')
                    })
                
                # Process backdrops
                for img in images.get('backdrops', [])[:10]:
                    gallery['backdrops'].append({
                        'url': self._get_image_url(img.get('file_path'), 'backdrop'),
                        'thumbnail': self._get_image_url(img.get('file_path'), 'thumbnail'),
                        'width': img.get('width'),
                        'height': img.get('height')
                    })
                
                # Process logos
                for img in images.get('logos', [])[:5]:
                    gallery['logos'].append({
                        'url': self._get_image_url(img.get('file_path'), 'logo'),
                        'language': img.get('iso_639_1')
                    })
            
            return gallery
            
        except Exception as e:
            logger.error(f"Gallery error: {e}")
            return {}
    
    def _get_metadata(self, content) -> Dict:
        """Get technical metadata"""
        try:
            metadata = {
                'quality': {
                    '4k': getattr(content, 'has_4k', False),
                    'hdr': getattr(content, 'has_hdr', False),
                    'dolby_vision': getattr(content, 'has_dolby_vision', False),
                    'dolby_atmos': getattr(content, 'has_dolby_atmos', False),
                    'imax': getattr(content, 'has_imax', False)
                },
                'audio_languages': json.loads(getattr(content, 'audio_languages', '[]')),
                'subtitle_languages': json.loads(getattr(content, 'subtitle_languages', '[]')),
                'production': {
                    'companies': json.loads(getattr(content, 'production_companies', '[]')),
                    'countries': json.loads(getattr(content, 'production_countries', '[]')),
                    'budget': getattr(content, 'budget', None),
                    'revenue': getattr(content, 'revenue', None)
                },
                'external_ids': {
                    'tmdb_id': content.tmdb_id,
                    'imdb_id': content.imdb_id,
                    'mal_id': content.mal_id
                },
                'keywords': json.loads(getattr(content, 'keywords', '[]')),
                'certifications': json.loads(getattr(content, 'certifications', '[]'))
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata error: {e}")
            return {}
    
    def _get_streaming_info(self, content) -> Dict:
        """Get streaming availability"""
        try:
            streaming = {
                'available_on': [],
                'rent_options': [],
                'buy_options': [],
                'free_with_ads': []
            }
            
            # This would typically call a streaming availability API
            # For now, return mock data
            streaming['available_on'] = [
                {'provider': 'Netflix', 'link': '#', 'quality': '4K'},
                {'provider': 'Amazon Prime', 'link': '#', 'quality': 'HD'}
            ]
            
            return streaming
            
        except Exception as e:
            logger.error(f"Streaming info error: {e}")
            return {}
    
    def _get_user_status(self, content_id: int, user_id: int) -> Dict:
        """Get user-specific status for content"""
        try:
            UserInteraction = models.get('UserInteraction')
            
            # Check watchlist
            watchlist = UserInteraction.query.filter_by(
                user_id=user_id,
                content_id=content_id,
                interaction_type='watchlist'
            ).first()
            
            # Check favorites
            favorite = UserInteraction.query.filter_by(
                user_id=user_id,
                content_id=content_id,
                interaction_type='favorite'
            ).first()
            
            # Get user rating
            rating = UserInteraction.query.filter_by(
                user_id=user_id,
                content_id=content_id,
                interaction_type='rating'
            ).first()
            
            return {
                'in_watchlist': watchlist is not None,
                'is_favorite': favorite is not None,
                'user_rating': rating.rating if rating else None,
                'last_watched': None  # Would need tracking implementation
            }
            
        except Exception as e:
            logger.error(f"User status error: {e}")
            return {}
    
    def _build_response(self, content, results: Dict, user_id: Optional[int]) -> Dict:
        """Build final response combining all data"""
        try:
            response = {
                'slug': content.slug,
                'base_info': results.get('base_info', {}),
                'trailer': results.get('trailer', {}),
                'synopsis': {
                    'overview': content.overview,
                    'tagline': getattr(content, 'tagline', ''),
                    'plot_summary': getattr(content, 'plot_summary', content.overview),
                    'content_warnings': results.get('ratings_reviews', {}).get('advisories', [])
                },
                'cast_crew': results.get('cast_crew', {}),
                'ratings': results.get('ratings_reviews', {}),
                'metadata': results.get('metadata', {}),
                'similar_content': results.get('similar', {}),
                'gallery': results.get('gallery', {}),
                'streaming': results.get('streaming', {}),
                'user_status': results.get('user_status', {}) if user_id else None,
                'seo': {
                    'title': f"{content.title} ({content.release_date.year if content.release_date else 'N/A'})",
                    'description': (content.overview or '')[:160],
                    'keywords': json.loads(getattr(content, 'keywords', '[]')),
                    'canonical_url': f"/details/{content.slug}",
                    'og_image': results.get('base_info', {}).get('backdrop_url'),
                    'schema_type': 'Movie' if content.content_type == 'movie' else 'TVSeries'
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Build response error: {e}")
            return {}
    
    def get_person_details(self, person_id: int) -> Dict:
        """Get detailed person information"""
        try:
            # Fetch from TMDB
            person_data = self._fetch_tmdb_person(person_id)
            
            if not person_data:
                return None
            
            # Process filmography
            filmography = {
                'movies': [],
                'tv_shows': [],
                'upcoming': []
            }
            
            for credit in person_data.get('movie_credits', {}).get('cast', []):
                filmography['movies'].append({
                    'id': credit.get('id'),
                    'title': credit.get('title'),
                    'character': credit.get('character'),
                    'year': credit.get('release_date', '')[:4] if credit.get('release_date') else None,
                    'poster': self._get_image_url(credit.get('poster_path'), 'poster')
                })
            
            for credit in person_data.get('tv_credits', {}).get('cast', []):
                filmography['tv_shows'].append({
                    'id': credit.get('id'),
                    'title': credit.get('name'),
                    'character': credit.get('character'),
                    'year': credit.get('first_air_date', '')[:4] if credit.get('first_air_date') else None,
                    'poster': self._get_image_url(credit.get('poster_path'), 'poster')
                })
            
            return {
                'id': person_data.get('id'),
                'name': person_data.get('name'),
                'biography': person_data.get('biography'),
                'birthday': person_data.get('birthday'),
                'deathday': person_data.get('deathday'),
                'place_of_birth': person_data.get('place_of_birth'),
                'profile_image': self._get_image_url(person_data.get('profile_path'), 'profile'),
                'known_for': person_data.get('known_for_department'),
                'popularity': person_data.get('popularity'),
                'filmography': filmography,
                'images': person_data.get('images', {}).get('profiles', [])[:10],
                'external_ids': person_data.get('external_ids', {})
            }
            
        except Exception as e:
            logger.error(f"Person details error: {e}")
            return None
    
    # Helper methods for API calls
    def _fetch_tmdb_videos(self, tmdb_id: int, content_type: str) -> List:
        """Fetch videos from TMDB"""
        try:
            TMDBService = services.get('TMDBService')
            if TMDBService:
                url = f"{self.tmdb_base_url}/{content_type}/{tmdb_id}/videos"
                params = {'api_key': services.get('TMDB_API_KEY')}
                response = services.get('http_session').get(url, params=params, timeout=5)
                if response.status_code == 200:
                    return response.json().get('results', [])
        except Exception as e:
            logger.error(f"TMDB videos fetch error: {e}")
        return []
    
    def _fetch_tmdb_credits(self, tmdb_id: int, content_type: str) -> Dict:
        """Fetch credits from TMDB"""
        try:
            url = f"{self.tmdb_base_url}/{content_type}/{tmdb_id}/credits"
            params = {'api_key': services.get('TMDB_API_KEY')}
            response = services.get('http_session').get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB credits fetch error: {e}")
        return {}
    
    def _fetch_tmdb_similar(self, tmdb_id: int, content_type: str) -> List:
        """Fetch similar content from TMDB"""
        try:
            url = f"{self.tmdb_base_url}/{content_type}/{tmdb_id}/recommendations"
            params = {'api_key': services.get('TMDB_API_KEY')}
            response = services.get('http_session').get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json().get('results', [])
        except Exception as e:
            logger.error(f"TMDB similar fetch error: {e}")
        return []
    
    def _fetch_tmdb_images(self, tmdb_id: int, content_type: str) -> Dict:
        """Fetch images from TMDB"""
        try:
            url = f"{self.tmdb_base_url}/{content_type}/{tmdb_id}/images"
            params = {'api_key': services.get('TMDB_API_KEY')}
            response = services.get('http_session').get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB images fetch error: {e}")
        return {}
    
    def _fetch_tmdb_person(self, person_id: int) -> Dict:
        """Fetch person details from TMDB"""
        try:
            url = f"{self.tmdb_base_url}/person/{person_id}"
            params = {
                'api_key': services.get('TMDB_API_KEY'),
                'append_to_response': 'movie_credits,tv_credits,images,external_ids'
            }
            response = services.get('http_session').get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB person fetch error: {e}")
        return None
    
    def _fetch_omdb_data(self, imdb_id: str) -> Dict:
        """Fetch data from OMDB"""
        try:
            params = {
                'apikey': services.get('OMDB_API_KEY'),
                'i': imdb_id,
                'plot': 'full'
            }
            response = services.get('http_session').get(self.omdb_base_url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"OMDB fetch error: {e}")
        return None
    
    def _fetch_mal_data(self, mal_id: int) -> Dict:
        """Fetch anime data from MAL via Jikan"""
        try:
            url = f"{self.jikan_base_url}/anime/{mal_id}"
            response = services.get('http_session').get(url, timeout=5)
            if response.status_code == 200:
                return response.json().get('data', {})
        except Exception as e:
            logger.error(f"MAL fetch error: {e}")
        return None
    
    def _fetch_anime_cast(self, mal_id: int) -> List:
        """Fetch anime cast from MAL"""
        try:
            url = f"{self.jikan_base_url}/anime/{mal_id}/characters"
            response = services.get('http_session').get(url, timeout=5)
            if response.status_code == 200:
                characters = response.json().get('data', [])
                cast = []
                for char in characters[:10]:
                    cast.append({
                        'character': char.get('character', {}).get('name'),
                        'role': char.get('role'),
                        'voice_actors': [
                            {
                                'name': va.get('person', {}).get('name'),
                                'language': va.get('language')
                            }
                            for va in char.get('voice_actors', [])
                        ]
                    })
                return cast
        except Exception as e:
            logger.error(f"Anime cast fetch error: {e}")
        return []
    
    def _get_image_url(self, path: str, image_type: str) -> Optional[str]:
        """Get full image URL"""
        if not path:
            return None
        
        if path.startswith('http'):
            return path
        
        sizes = {
            'poster': 'w500',
            'backdrop': 'w1280',
            'profile': 'w185',
            'logo': 'w300',
            'thumbnail': 'w200'
        }
        
        size = sizes.get(image_type, 'original')
        return f"https://image.tmdb.org/t/p/{size}{path}"
    
    def _format_content_card(self, content) -> Dict:
        """Format content for card display"""
        return {
            'id': content.id,
            'slug': content.slug,
            'title': content.title,
            'content_type': content.content_type,
            'poster_url': self._get_image_url(content.poster_path, 'poster'),
            'rating': content.rating,
            'year': content.release_date.year if content.release_date else None
        }
    
    def _get_content_advisories(self, content) -> List[str]:
        """Get content advisories based on rating and content"""
        advisories = []
        
        rating = getattr(content, 'content_rating', 'NR')
        
        # Basic advisories based on rating
        rating_advisories = {
            'R': ['Strong Language', 'Violence', 'Adult Content'],
            'PG-13': ['Some Violence', 'Mild Language', 'Thematic Elements'],
            'TV-MA': ['Mature Content', 'Strong Language', 'Violence'],
            'TV-14': ['Suggestive Content', 'Violence', 'Language']
        }
        
        if rating in rating_advisories:
            advisories.extend(rating_advisories[rating])
        
        # Genre-based advisories
        if content.genres:
            genres = json.loads(content.genres)
            if 'Horror' in genres:
                advisories.append('Frightening Scenes')
            if 'War' in genres:
                advisories.append('War Violence')
            if 'Crime' in genres:
                advisories.append('Criminal Content')
        
        return list(set(advisories))  # Remove duplicates
    
    def _record_interaction(self, user_id: int, content_id: int, interaction_type: str):
        """Record user interaction"""
        try:
            UserInteraction = models.get('UserInteraction')
            interaction = UserInteraction(
                user_id=user_id,
                content_id=content_id,
                interaction_type=interaction_type
            )
            db.session.add(interaction)
            db.session.commit()
        except Exception as e:
            logger.error(f"Record interaction error: {e}")
            db.session.rollback()

# Initialize service
detail_service = DetailService()

# API Endpoints
@details_bp.route('/api/details/<slug>', methods=['GET'])
def get_content_details(slug):
    """Get comprehensive content details by slug"""
    try:
        # Get user ID from token if authenticated
        user_id = None
        auth_header = request.headers.get('Authorization')
        if auth_header:
            try:
                token = auth_header.replace('Bearer ', '')
                data = jwt.decode(token, current_app.secret_key, algorithms=['HS256'])
                user_id = data.get('user_id')
            except:
                pass
        
        # Check cache
        cache_key = f"details:{slug}:{user_id or 'anon'}"
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return jsonify(cached_data), 200
        
        # Get details
        details = detail_service.get_content_details(slug, user_id)
        
        if not details:
            return jsonify({'error': 'Content not found'}), 404
        
        # Cache for 30 minutes
        cache.set(cache_key, details, timeout=1800)
        
        return jsonify(details), 200
        
    except Exception as e:
        logger.error(f"Get content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

@details_bp.route('/api/person/<int:person_id>', methods=['GET'])
def get_person_details(person_id):
    """Get person details"""
    try:
        # Check cache
        cache_key = f"person:{person_id}"
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return jsonify(cached_data), 200
        
        # Get details
        details = detail_service.get_person_details(person_id)
        
        if not details:
            return jsonify({'error': 'Person not found'}), 404
        
        # Cache for 1 hour
        cache.set(cache_key, details, timeout=3600)
        
        return jsonify(details), 200
        
    except Exception as e:
        logger.error(f"Get person details error: {e}")
        return jsonify({'error': 'Failed to get person details'}), 500

@details_bp.route('/api/details/<slug>/reviews', methods=['GET'])
def get_content_reviews(slug):
    """Get paginated reviews for content"""
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        sort_by = request.args.get('sort_by', 'recent')  # recent, helpful, rating
        
        # Resolve slug
        slug_data = SlugManager.resolve_slug(slug)
        if not slug_data:
            return jsonify({'error': 'Content not found'}), 404
        
        UserInteraction = models.get('UserInteraction')
        User = models.get('User')
        
        # Build query
        query = UserInteraction.query.filter_by(
            content_id=slug_data['id'],
            interaction_type='review'
        )
        
        # Apply sorting
        if sort_by == 'helpful':
            query = query.order_by(UserInteraction.helpful_count.desc())
        elif sort_by == 'rating':
            query = query.order_by(UserInteraction.rating.desc())
        else:
            query = query.order_by(UserInteraction.timestamp.desc())
        
        # Paginate
        reviews_paginated = query.paginate(page=page, per_page=limit, error_out=False)
        
        reviews = []
        for review in reviews_paginated.items:
            user = User.query.get(review.user_id)
            if user:
                reviews.append({
                    'id': review.id,
                    'user': {
                        'id': user.id,
                        'username': user.username,
                        'avatar': getattr(user, 'avatar_url', None)
                    },
                    'rating': review.rating,
                    'review': getattr(review, 'review_text', ''),
                    'date': review.timestamp.isoformat(),
                    'helpful_count': getattr(review, 'helpful_count', 0),
                    'spoiler': getattr(review, 'has_spoiler', False)
                })
        
        return jsonify({
            'reviews': reviews,
            'pagination': {
                'page': page,
                'total_pages': reviews_paginated.pages,
                'total_reviews': reviews_paginated.total,
                'has_next': reviews_paginated.has_next,
                'has_prev': reviews_paginated.has_prev
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Get reviews error: {e}")
        return jsonify({'error': 'Failed to get reviews'}), 500

@details_bp.route('/api/details/<slug>/review', methods=['POST'])
def add_review(slug):
    """Add user review for content"""
    try:
        # Require authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authentication required'}), 401
        
        try:
            token = auth_header.replace('Bearer ', '')
            data = jwt.decode(token, current_app.secret_key, algorithms=['HS256'])
            user_id = data.get('user_id')
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        # Resolve slug
        slug_data = SlugManager.resolve_slug(slug)
        if not slug_data:
            return jsonify({'error': 'Content not found'}), 404
        
        # Get review data
        review_data = request.get_json()
        
        if not review_data.get('rating'):
            return jsonify({'error': 'Rating is required'}), 400
        
        UserInteraction = models.get('UserInteraction')
        
        # Check if user already reviewed
        existing = UserInteraction.query.filter_by(
            user_id=user_id,
            content_id=slug_data['id'],
            interaction_type='review'
        ).first()
        
        if existing:
            # Update existing review
            existing.rating = review_data['rating']
            existing.review_text = review_data.get('review', '')
            existing.has_spoiler = review_data.get('spoiler', False)
            existing.timestamp = datetime.utcnow()
        else:
            # Create new review
            review = UserInteraction(
                user_id=user_id,
                content_id=slug_data['id'],
                interaction_type='review',
                rating=review_data['rating']
            )
            review.review_text = review_data.get('review', '')
            review.has_spoiler = review_data.get('spoiler', False)
            db.session.add(review)
        
        db.session.commit()
        
        # Clear cache
        cache.delete(f"details:{slug}:*")
        
        return jsonify({'message': 'Review added successfully'}), 201
        
    except Exception as e:
        logger.error(f"Add review error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to add review'}), 500