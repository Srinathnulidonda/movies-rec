"""
CineBrain Details Core Service
Main orchestrator for content details management
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import has_app_context
from sqlalchemy import and_, or_, func, desc
import json

from .tmdb_service import TMDBService
from .omdb_service import OMDBService
from .youtube_service import YouTubeService
from .jikan_service import JikanService
from .cast_crew import CastCrewManager
from .similar_content import SimilarContentManager
from .metadata_utils import MetadataUtils
from .slug import SlugManager
from .cache_manager import CacheManager
from .analytics import AnalyticsManager
from .errors import DetailsError, APIError, ContentNotFoundError, PersonNotFoundError
from .validator import DetailsValidator

logger = logging.getLogger(__name__)

class DetailsService:
    """Main service class for handling content details"""
    
    def __init__(self, db, models, cache_manager: CacheManager):
        self.db = db
        self.models = models
        self.cache_manager = cache_manager
        
        # Initialize sub-services
        self.tmdb_service = TMDBService()
        self.omdb_service = OMDBService()
        self.youtube_service = YouTubeService()
        self.jikan_service = JikanService()
        
        # Initialize managers
        self.cast_crew_manager = CastCrewManager(db, models)
        self.similar_content_manager = SimilarContentManager(db, models)
        self.metadata_utils = MetadataUtils()
        self.slug_manager = SlugManager()
        self.analytics_manager = AnalyticsManager(db, models)
        
        # Initialize models
        self.Content = models.get('Content')
        self.Person = models.get('Person')
        self.User = models.get('User')
        self.Review = models.get('Review')
        self.UserInteraction = models.get('UserInteraction')
        self.AnonymousInteraction = models.get('AnonymousInteraction')
        
        logger.info("CineBrain DetailsService initialized successfully")
    
    def get_comprehensive_details(self, slug: str, user_id: Optional[int] = None, 
                                force_refresh: bool = False, include_trailers: bool = True,
                                include_reviews: bool = True, include_similar: bool = True) -> Optional[Dict]:
        """Get comprehensive content details with intelligent caching"""
        try:
            if not has_app_context():
                logger.warning("No app context available for comprehensive details")
                return None
            
            # Validate input
            if not DetailsValidator.validate_slug(slug):
                raise DetailsError(f"Invalid slug format: {slug}")
            
            # Check cache first
            cache_key = self.cache_manager.generate_details_cache_key(slug, user_id)
            if not force_refresh:
                cached_details = self.cache_manager.get_cached_details(cache_key)
                if cached_details:
                    logger.info(f"Cache hit for details: {slug}")
                    # Add user-specific data if needed
                    if user_id and 'user_data' not in cached_details:
                        cached_details['user_data'] = self._get_user_data(cached_details.get('id'), user_id)
                    return cached_details
            
            # Get content from database
            content = self._get_content_by_slug(slug)
            if not content:
                logger.warning(f"Content not found for slug: {slug}")
                raise ContentNotFoundError(slug)
            
            # Build details with parallel processing
            details = self._build_comprehensive_details(
                content, user_id, include_trailers, include_reviews, include_similar
            )
            
            if details:
                # Cache the results (without user-specific data)
                cache_data = details.copy()
                cache_data.pop('user_data', None)
                self.cache_manager.cache_details(cache_key, cache_data)
                
                logger.info(f"Successfully built comprehensive details for: {slug}")
            
            return details
            
        except (ContentNotFoundError, DetailsError) as e:
            raise e
        except Exception as e:
            logger.error(f"Error getting comprehensive details for {slug}: {e}")
            raise DetailsError(f"Failed to get details for {slug}")
    
    def get_person_details(self, slug: str, force_refresh: bool = False) -> Optional[Dict]:
        """Get comprehensive person details"""
        try:
            if not has_app_context():
                logger.warning("No app context available for person details")
                return None
            
            # Validate input
            if not DetailsValidator.validate_slug(slug):
                raise DetailsError(f"Invalid person slug format: {slug}")
            
            # Check cache
            cache_key = self.cache_manager.generate_person_cache_key(slug)
            if not force_refresh:
                cached = self.cache_manager.get_cached_data(cache_key)
                if cached:
                    logger.info(f"Cache hit for person: {slug}")
                    return cached
            
            # Get person from database
            person = self._get_person_by_slug(slug)
            if not person:
                logger.warning(f"Person not found for slug: {slug}")
                raise PersonNotFoundError(slug)
            
            # Build person details
            details = self._build_person_details(person)
            
            if details:
                self.cache_manager.cache_data(cache_key, details, timeout=3600)
                logger.info(f"Successfully built person details for: {slug}")
            
            return details
            
        except (PersonNotFoundError, DetailsError) as e:
            raise e
        except Exception as e:
            logger.error(f"Error getting person details for {slug}: {e}")
            raise DetailsError(f"Failed to get person details for {slug}")
    
    def get_trailer_for_content(self, slug: str) -> Optional[Dict]:
        """Get trailer for specific content - on-demand fetching"""
        try:
            # Validate input
            if not DetailsValidator.validate_slug(slug):
                raise DetailsError(f"Invalid slug format: {slug}")
            
            content = self._get_content_by_slug(slug)
            if not content:
                raise ContentNotFoundError(slug)
            
            # Check cache first
            trailer_cache_key = self.cache_manager.generate_trailer_cache_key(content.id)
            cached_trailer = self.cache_manager.get_cached_data(trailer_cache_key)
            if cached_trailer:
                logger.info(f"Cache hit for trailer: {slug}")
                return cached_trailer
            
            # Check if we already have a stored trailer
            if content.youtube_trailer_id:
                trailer_data = {
                    'youtube_id': content.youtube_trailer_id,
                    'embed_url': f"https://www.youtube.com/embed/{content.youtube_trailer_id}",
                    'watch_url': f"https://www.youtube.com/watch?v={content.youtube_trailer_id}",
                    'source': 'stored',
                    'title': f"{content.title} - Official Trailer",
                    'thumbnail': f"https://img.youtube.com/vi/{content.youtube_trailer_id}/hqdefault.jpg"
                }
                
                # Cache the stored trailer
                self.cache_manager.cache_data(trailer_cache_key, trailer_data, timeout=86400)
                return trailer_data
            
            # Fetch fresh trailer from YouTube
            logger.info(f"Fetching fresh trailer for: {content.title}")
            trailer = self.youtube_service.search_and_validate_trailer(
                content.title, content.content_type
            )
            
            if trailer:
                # Save trailer ID to content
                try:
                    content.youtube_trailer_id = trailer['youtube_id']
                    self.db.session.commit()
                    logger.info(f"Saved trailer for {content.title}: {trailer['youtube_id']}")
                    
                    # Cache the new trailer
                    self.cache_manager.cache_data(trailer_cache_key, trailer, timeout=86400)
                    
                except Exception as e:
                    logger.warning(f"Failed to save trailer ID: {e}")
                    self.db.session.rollback()
            else:
                logger.info(f"No trailer found for: {content.title}")
            
            return trailer
            
        except (ContentNotFoundError, DetailsError) as e:
            raise e
        except Exception as e:
            logger.error(f"Error getting trailer for {slug}: {e}")
            return None
    
    def get_content_reviews(self, slug: str, page: int = 1, limit: int = 10, 
                          sort_by: str = 'newest', user_id: Optional[int] = None) -> Dict:
        """Get reviews for content"""
        try:
            # Validate inputs
            if not DetailsValidator.validate_slug(slug):
                raise DetailsError(f"Invalid slug format: {slug}")
            
            if not DetailsValidator.validate_pagination(page, limit):
                raise DetailsError("Invalid pagination parameters")
            
            if not DetailsValidator.validate_sort_option(sort_by):
                raise DetailsError(f"Invalid sort option: {sort_by}")
            
            content = self._get_content_by_slug(slug)
            if not content:
                raise ContentNotFoundError(slug)
            
            # Check cache
            reviews_cache_key = self.cache_manager.generate_reviews_cache_key(slug, page, limit, sort_by)
            cached_reviews = self.cache_manager.get_cached_data(reviews_cache_key)
            if cached_reviews:
                logger.info(f"Cache hit for reviews: {slug}")
                return cached_reviews
            
            # Build query
            if not self.Review or not self.User:
                return {'reviews': [], 'stats': {}, 'pagination': {}}
            
            query = self.db.session.query(self.Review, self.User).join(self.User).filter(
                self.Review.content_id == content.id,
                self.Review.is_approved == True
            )
            
            # Apply sorting
            sort_mappings = {
                'newest': self.Review.created_at.desc(),
                'oldest': self.Review.created_at.asc(),
                'helpful': self.Review.helpful_count.desc(),
                'rating_high': self.Review.rating.desc(),
                'rating_low': self.Review.rating.asc()
            }
            
            if sort_by in sort_mappings:
                query = query.order_by(sort_mappings[sort_by])
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination
            offset = (page - 1) * limit
            reviews_data = query.offset(offset).limit(limit).all()
            
            # Build review objects
            reviews = []
            for review, user in reviews_data:
                review_data = {
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
                    'updated_at': review.updated_at.isoformat() if review.updated_at else None,
                    'is_user_review': review.user_id == user_id if user_id else False
                }
                reviews.append(review_data)
            
            # Calculate stats
            stats = self._calculate_review_stats(content.id)
            
            # Build pagination info
            pagination = {
                'current_page': page,
                'total_pages': (total_count + limit - 1) // limit,
                'total_reviews': total_count,
                'has_next': page * limit < total_count,
                'has_prev': page > 1,
                'per_page': limit
            }
            
            result = {
                'reviews': reviews,
                'stats': stats,
                'pagination': pagination,
                'content_info': {
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type
                }
            }
            
            # Cache the result
            self.cache_manager.cache_data(reviews_cache_key, result, timeout=600)
            
            return result
            
        except (ContentNotFoundError, DetailsError) as e:
            raise e
        except Exception as e:
            logger.error(f"Error getting reviews for {slug}: {e}")
            return {'reviews': [], 'stats': {}, 'pagination': {}}
    
    def get_similar_content(self, slug: str, limit: int = 12, algorithm: str = 'genre_based') -> List[Dict]:
        """Get similar content using specified algorithm"""
        try:
            # Validate inputs
            if not DetailsValidator.validate_slug(slug):
                raise DetailsError(f"Invalid slug format: {slug}")
            
            if not DetailsValidator.validate_limit(limit, 24):
                raise DetailsError("Invalid limit parameter")
            
            if not DetailsValidator.validate_algorithm(algorithm):
                raise DetailsError(f"Invalid algorithm: {algorithm}")
            
            content = self._get_content_by_slug(slug)
            if not content:
                raise ContentNotFoundError(slug)
            
            # Check cache
            similar_cache_key = self.cache_manager.generate_similar_cache_key(content.id, algorithm, limit)
            cached_similar = self.cache_manager.get_cached_data(similar_cache_key)
            if cached_similar:
                logger.info(f"Cache hit for similar content: {slug}")
                return cached_similar
            
            # Get similar content
            similar_content = self.similar_content_manager.get_similar_content(
                content, limit, algorithm
            )
            
            # Cache the result
            if similar_content:
                self.cache_manager.cache_data(similar_cache_key, similar_content, timeout=1800)
            
            return similar_content
            
        except (ContentNotFoundError, DetailsError) as e:
            raise e
        except Exception as e:
            logger.error(f"Error getting similar content for {slug}: {e}")
            return []
    
    def add_review(self, slug: str, user_id: int, review_data: Dict) -> Dict:
        """Add a review for content"""
        try:
            # Validate inputs
            if not DetailsValidator.validate_slug(slug):
                raise DetailsError(f"Invalid slug format: {slug}")
            
            content = self._get_content_by_slug(slug)
            if not content:
                raise ContentNotFoundError(slug)
            
            if not self.Review:
                raise DetailsError("Review system not available")
            
            # Check if user already reviewed this content
            existing_review = self.Review.query.filter_by(
                content_id=content.id,
                user_id=user_id
            ).first()
            
            if existing_review:
                raise DetailsError("You have already reviewed this content")
            
            # Validate review data
            rating = review_data.get('rating')
            if rating is not None and not DetailsValidator.validate_rating(rating):
                raise DetailsError("Invalid rating value")
            
            # Get user for auto-approval check
            user = self.User.query.get(user_id) if self.User else None
            should_auto_approve = self._should_auto_approve_review(user, review_data)
            
            # Create review
            review = self.Review(
                content_id=content.id,
                user_id=user_id,
                rating=rating,
                title=review_data.get('title', '').strip(),
                review_text=review_data.get('review_text', '').strip(),
                has_spoilers=review_data.get('has_spoilers', False),
                is_approved=should_auto_approve
            )
            
            self.db.session.add(review)
            self.db.session.commit()
            
            # Clear relevant caches
            self._clear_content_caches(content.slug)
            
            message = 'Review published successfully' if should_auto_approve else 'Review submitted for moderation'
            
            return {
                'success': True,
                'review_id': review.id,
                'message': message,
                'auto_approved': should_auto_approve
            }
            
        except (ContentNotFoundError, DetailsError) as e:
            raise e
        except Exception as e:
            logger.error(f"Error adding review: {e}")
            self.db.session.rollback()
            raise DetailsError("Failed to add review")
    
    def vote_review_helpful(self, review_id: int, user_id: int, is_helpful: bool = True) -> bool:
        """Vote on review helpfulness"""
        try:
            if not self.Review:
                raise DetailsError("Review system not available")
            
            review = self.Review.query.get(review_id)
            if not review:
                raise DetailsError("Review not found")
            
            # Update helpful count
            if is_helpful:
                review.helpful_count = (review.helpful_count or 0) + 1
            else:
                review.helpful_count = max(0, (review.helpful_count or 0) - 1)
            
            self.db.session.commit()
            
            # Clear review caches
            content = self.Content.query.get(review.content_id)
            if content:
                self._clear_content_caches(content.slug)
            
            return True
            
        except DetailsError as e:
            raise e
        except Exception as e:
            logger.error(f"Error voting on review: {e}")
            self.db.session.rollback()
            raise DetailsError("Failed to vote on review")
    
    def clear_all_cache(self) -> int:
        """Clear all details cache"""
        try:
            return self.cache_manager.clear_all_cache()
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
    
    def get_health_status(self) -> Dict:
        """Get health status of details service"""
        try:
            health = {
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'healthy',
                'services': {
                    'tmdb': self.tmdb_service.check_health(),
                    'omdb': self.omdb_service.check_health(),
                    'youtube': self.youtube_service.check_health(),
                    'jikan': self.jikan_service.check_health()
                },
                'cache': {
                    'status': 'available' if self.cache_manager.is_available() else 'unavailable',
                    'stats': self.cache_manager.get_cache_stats()
                },
                'database': self._get_database_health(),
                'performance': self._get_performance_metrics()
            }
            
            # Determine overall health
            service_issues = sum(1 for service in health['services'].values() 
                               if service.get('status') != 'healthy')
            
            if service_issues >= 3:
                health['status'] = 'unhealthy'
            elif service_issues >= 1:
                health['status'] = 'degraded'
            
            return health
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                'status': 'error', 
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_analytics(self, content_id: Optional[int] = None) -> Dict:
        """Get analytics data"""
        try:
            if content_id:
                return self.analytics_manager.get_content_analytics(content_id)
            else:
                return {
                    'popularity_trends': self.analytics_manager.get_popularity_trends(),
                    'genre_analytics': self.analytics_manager.get_genre_analytics()
                }
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {}
    
    def _get_content_by_slug(self, slug: str) -> Optional[Any]:
        """Get content by slug with fuzzy search fallback"""
        try:
            # Direct slug match
            content = self.Content.query.filter_by(slug=slug).first()
            
            if not content:
                # Try fuzzy search
                content = self._find_content_fuzzy(slug)
                
                if not content:
                    # Try external search as last resort
                    info = self.slug_manager.extract_info_from_slug(slug)
                    if self._should_fetch_from_external_strict(slug, info['title'], info['year']):
                        content = self._try_fetch_from_external(info['title'], info['year'], info['content_type'])
            
            return content
            
        except Exception as e:
            logger.error(f"Error getting content by slug {slug}: {e}")
            return None
    
    def _get_person_by_slug(self, slug: str) -> Optional[Any]:
        """Get person by slug with fuzzy search fallback"""
        try:
            person = self.Person.query.filter_by(slug=slug).first()
            
            if not person:
                # Try fuzzy search
                person = self._find_person_fuzzy(slug)
            
            return person
            
        except Exception as e:
            logger.error(f"Error getting person by slug {slug}: {e}")
            return None
    
    def _build_comprehensive_details(self, content: Any, user_id: Optional[int],
                                   include_trailers: bool, include_reviews: bool,
                                   include_similar: bool) -> Dict:
        """Build comprehensive content details with parallel processing"""
        try:
            # Base details
            details = {
                'id': content.id,
                'slug': content.slug,
                'title': content.title,
                'original_title': content.original_title,
                'overview': content.overview,
                'content_type': content.content_type,
                'poster_url': self._format_image_url(content.poster_path, 'poster'),
                'backdrop_url': self._format_image_url(content.backdrop_path, 'backdrop'),
                'tmdb_id': content.tmdb_id,
                'imdb_id': content.imdb_id,
                'mal_id': content.mal_id if hasattr(content, 'mal_id') else None,
                'created_at': content.created_at.isoformat() if hasattr(content, 'created_at') else None,
                'updated_at': content.updated_at.isoformat() if hasattr(content, 'updated_at') else None
            }
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {}
                
                # Core data futures
                if content.tmdb_id:
                    futures['tmdb'] = executor.submit(
                        self.tmdb_service.get_content_details,
                        content.tmdb_id, content.content_type
                    )
                
                if content.imdb_id:
                    futures['omdb'] = executor.submit(
                        self.omdb_service.get_content_details,
                        content.imdb_id
                    )
                
                # Always get cast/crew
                futures['cast_crew'] = executor.submit(
                    self.cast_crew_manager.get_cast_crew,
                    content.id
                )
                
                # Gallery/images
                futures['gallery'] = executor.submit(
                    self._get_gallery,
                    content.id
                )
                
                # Optional futures based on request
                if include_trailers:
                    futures['trailer'] = executor.submit(
                        self.get_trailer_for_content,
                        content.slug
                    )
                
                if include_reviews:
                    futures['reviews'] = executor.submit(
                        self.get_content_reviews,
                        content.slug, page=1, limit=5, user_id=user_id
                    )
                
                if include_similar:
                    futures['similar'] = executor.submit(
                        self.get_similar_content,
                        content.slug, limit=12
                    )
                
                # Streaming info
                futures['streaming'] = executor.submit(
                    self._get_streaming_info,
                    content
                )
                
                # Collect results with timeouts
                results = {}
                for key, future in futures.items():
                    try:
                        timeout = 20 if key in ['tmdb', 'omdb'] else 15
                        results[key] = future.result(timeout=timeout)
                    except Exception as e:
                        logger.warning(f"Error getting {key} data: {e}")
                        results[key] = None
            
            # Merge API results
            tmdb_data = results.get('tmdb', {})
            omdb_data = results.get('omdb', {})
            
            # Build comprehensive sections
            details['synopsis'] = self.metadata_utils.build_synopsis(content, tmdb_data, omdb_data)
            details['ratings'] = self.metadata_utils.build_ratings(content, tmdb_data, omdb_data)
            details['metadata'] = self.metadata_utils.build_metadata(content, tmdb_data, omdb_data)
            
            # Add component data
            details['cast_crew'] = results.get('cast_crew', self._empty_cast_crew())
            details['gallery'] = results.get('gallery', self._empty_gallery())
            details['streaming_info'] = results.get('streaming')
            
            # Optional components
            if include_trailers:
                details['trailer'] = results.get('trailer')
            
            if include_reviews:
                review_data = results.get('reviews', {})
                details['reviews'] = review_data.get('reviews', [])
                details['review_stats'] = review_data.get('stats', {})
            
            if include_similar:
                details['more_like_this'] = results.get('similar', [])
            
            # Seasons/episodes for TV content
            if content.content_type in ['tv', 'anime']:
                details['seasons_episodes'] = self._get_seasons_episodes(content, tmdb_data)
            
            # Add user-specific data if user_id provided
            if user_id:
                details['user_data'] = self._get_user_data(content.id, user_id)
            
            # Add analytics if available
            try:
                details['analytics'] = self.analytics_manager.get_content_analytics(content.id)
            except Exception as e:
                logger.warning(f"Failed to get analytics: {e}")
                details['analytics'] = {}
            
            return details
            
        except Exception as e:
            logger.error(f"Error building comprehensive details: {e}")
            raise DetailsError(f"Failed to build details for content {content.id}")
    
    def _build_person_details(self, person: Any) -> Dict:
        """Build comprehensive person details"""
        try:
            # Get external data if available
            tmdb_data = {}
            if person.tmdb_id:
                try:
                    tmdb_data = self.tmdb_service.get_person_details(person.tmdb_id) or {}
                except Exception as e:
                    logger.warning(f"Failed to get TMDB person data: {e}")
            
            # Update person with TMDB data if needed
            self._update_person_from_tmdb(person, tmdb_data)
            
            # Get filmography
            filmography = self.cast_crew_manager.get_person_filmography(person.id)
            
            # Build comprehensive details
            details = {
                'id': person.id,
                'slug': person.slug,
                'tmdb_id': person.tmdb_id,
                'name': person.name,
                'biography': person.biography or tmdb_data.get('biography', ''),
                'birthday': person.birthday.isoformat() if person.birthday else tmdb_data.get('birthday'),
                'deathday': person.deathday.isoformat() if person.deathday else tmdb_data.get('deathday'),
                'place_of_birth': person.place_of_birth or tmdb_data.get('place_of_birth'),
                'profile_url': self._format_image_url(person.profile_path, 'profile'),
                'popularity': person.popularity or tmdb_data.get('popularity', 0),
                'known_for_department': person.known_for_department or tmdb_data.get('known_for_department'),
                'gender': person.gender,
                'filmography': filmography,
                'images': self._get_person_images(tmdb_data),
                'social_media': self._get_person_social_media(tmdb_data),
                'personal_info': self._build_enhanced_personal_info(person, tmdb_data, filmography),
                'career_highlights': self._build_enhanced_career_highlights(filmography),
                'external_ids': tmdb_data.get('external_ids', {}),
                'total_works': self._calculate_total_works(filmography),
                'created_at': person.created_at.isoformat() if hasattr(person, 'created_at') else None,
                'updated_at': person.updated_at.isoformat() if hasattr(person, 'updated_at') else None
            }
            
            # Parse also_known_as
            try:
                if person.also_known_as:
                    if isinstance(person.also_known_as, str):
                        details['also_known_as'] = json.loads(person.also_known_as)
                    else:
                        details['also_known_as'] = person.also_known_as
                else:
                    details['also_known_as'] = tmdb_data.get('also_known_as', [])
            except (json.JSONDecodeError, TypeError):
                details['also_known_as'] = []
            
            return details
            
        except Exception as e:
            logger.error(f"Error building person details: {e}")
            return self._get_minimal_person_details(person)
    
    # Helper methods continue here...
    def _find_content_fuzzy(self, slug: str) -> Optional[Any]:
        """Find content using fuzzy matching"""
        try:
            info = self.slug_manager.extract_info_from_slug(slug)
            title = info['title']
            year = info['year']
            content_type = info['content_type']
            
            # Conservative fuzzy search
            title_variations = self._generate_conservative_title_variations(title)
            
            results = []
            for variation in title_variations[:5]:  # Limit variations
                # Exact title matches first
                matches = self.Content.query.filter(
                    func.lower(self.Content.title) == variation.lower(),
                    self.Content.content_type == content_type
                ).all()
                
                if year:
                    # Filter by year if provided
                    year_matches = [m for m in matches if m.release_date and abs(m.release_date.year - year) <= 2]
                    if year_matches:
                        results.extend(year_matches)
                    else:
                        results.extend(matches)
                else:
                    results.extend(matches)
                
                if results:
                    break
            
            if results:
                # Return best match based on popularity/rating
                best_match = max(results, key=lambda x: (x.popularity or 0, x.rating or 0))
                logger.info(f"Found fuzzy match for '{slug}': {best_match.title}")
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fuzzy content search: {e}")
            return None
    
    def _find_person_fuzzy(self, slug: str) -> Optional[Any]:
        """Find person using fuzzy matching"""
        try:
            info = self.slug_manager.extract_info_from_slug(slug)
            name = info['title']
            
            # Search by name similarity
            results = self.Person.query.filter(
                func.lower(self.Person.name).like(f"%{name.lower()}%")
            ).limit(10).all()
            
            if results:
                # Find best match
                from difflib import SequenceMatcher
                best_match = max(results, key=lambda x: SequenceMatcher(None, x.name.lower(), name.lower()).ratio())
                logger.info(f"Found fuzzy person match for '{slug}': {best_match.name}")
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fuzzy person search: {e}")
            return None
    
    def _should_fetch_from_external_strict(self, slug: str, title: str, year: Optional[int]) -> bool:
        """Determine if we should fetch from external APIs (strict criteria)"""
        try:
            current_year = datetime.now().year
            
            # Only fetch for recent content or if explicitly requested
            if year and (current_year - 2 <= year <= current_year + 3):
                return True
            
            # Only fetch for popular/trending keywords
            trending_keywords = ['marvel', 'dc', 'disney', 'netflix', 'sequel', 'remake', 'reboot']
            if any(keyword in title.lower() for keyword in trending_keywords):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _try_fetch_from_external(self, title: str, year: Optional[int], content_type: str) -> Optional[Any]:
        """Try to fetch content from external APIs"""
        try:
            # Search TMDB
            search_type = 'multi' if content_type == 'multi' else content_type
            search_results = self.tmdb_service.search_content(title, search_type, page=1)
            
            if search_results and search_results.get('results'):
                for result in search_results['results'][:3]:  # Only check top 3
                    if year:
                        # Verify year matches
                        result_date = result.get('release_date') or result.get('first_air_date')
                        if result_date:
                            try:
                                result_year = int(result_date[:4])
                                if abs(result_year - year) > 2:
                                    continue
                            except (ValueError, TypeError):
                                pass
                    
                    # Check if content already exists
                    existing = self.Content.query.filter_by(tmdb_id=result['id']).first()
                    if existing:
                        return existing
                    
                    # Create new content
                    detected_type = 'movie' if 'title' in result else 'tv'
                    saved_content = self._save_content_from_tmdb(result, detected_type)
                    
                    if saved_content:
                        logger.info(f"Fetched and saved content from TMDB: {saved_content.title}")
                        return saved_content
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching from external APIs: {e}")
            return None
    
    def _save_content_from_tmdb(self, tmdb_data: Dict, content_type: str) -> Optional[Any]:
        """Save content from TMDB data"""
        try:
            title = tmdb_data.get('title') or tmdb_data.get('name') or 'Unknown Title'
            original_title = tmdb_data.get('original_title') or tmdb_data.get('original_name')
            
            # Parse release date
            release_date = None
            year = None
            date_str = tmdb_data.get('release_date') or tmdb_data.get('first_air_date')
            if date_str:
                try:
                    release_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    year = release_date.year
                except Exception as e:
                    logger.warning(f"Error parsing date '{date_str}': {e}")
            
            # Parse genres
            genres = []
            if 'genres' in tmdb_data:
                genres = [genre['name'] for genre in tmdb_data['genres']]
            elif 'genre_ids' in tmdb_data:
                genres = self._map_genre_ids(tmdb_data['genre_ids'])
            
            # Generate slug
            slug = self.slug_manager.generate_unique_slug(
                self.db, self.Content, title, year, content_type,
                original_title=original_title, tmdb_id=tmdb_data['id']
            )
            
            # Create content
            content_data = {
                'slug': slug,
                'tmdb_id': tmdb_data['id'],
                'title': title,
                'original_title': original_title,
                'content_type': content_type,
                'genres': json.dumps(genres) if genres else None,
                'release_date': release_date,
                'rating': tmdb_data.get('vote_average'),
                'vote_count': tmdb_data.get('vote_count'),
                'popularity': tmdb_data.get('popularity'),
                'overview': tmdb_data.get('overview'),
                'poster_path': tmdb_data.get('poster_path'),
                'backdrop_path': tmdb_data.get('backdrop_path')
            }
            
            content = self.Content(**content_data)
            self.db.session.add(content)
            self.db.session.commit()
            
            return content
            
        except Exception as e:
            logger.error(f"Error saving content from TMDB: {e}")
            self.db.session.rollback()
            return None
    
    def _calculate_review_stats(self, content_id: int) -> Dict:
        """Calculate review statistics"""
        try:
            if not self.Review:
                return {}
            
            reviews = self.Review.query.filter_by(
                content_id=content_id,
                is_approved=True
            ).all()
            
            if not reviews:
                return {
                    'total_reviews': 0,
                    'average_rating': 0,
                    'rating_distribution': {},
                    'has_spoilers_count': 0
                }
            
            # Calculate statistics
            ratings = [r.rating for r in reviews if r.rating is not None]
            average_rating = sum(ratings) / len(ratings) if ratings else 0
            
            # Rating distribution
            distribution = {}
            for i in range(1, 11):
                distribution[str(i)] = sum(1 for r in ratings if int(r) == i)
            
            spoilers_count = sum(1 for r in reviews if r.has_spoilers)
            
            return {
                'total_reviews': len(reviews),
                'average_rating': round(average_rating, 1),
                'rating_distribution': distribution,
                'has_spoilers_count': spoilers_count,
                'total_helpful_votes': sum(r.helpful_count or 0 for r in reviews)
            }
            
        except Exception as e:
            logger.error(f"Error calculating review stats: {e}")
            return {}
    
    def _get_user_data(self, content_id: int, user_id: int) -> Dict:
        """Get user-specific data for content"""
        try:
            user_data = {
                'in_watchlist': False,
                'in_favorites': False,
                'user_rating': None,
                'watch_status': None,
                'has_reviewed': False
            }
            
            if not self.UserInteraction:
                return user_data
            
            # Get interactions
            interactions = self.UserInteraction.query.filter_by(
                user_id=user_id,
                content_id=content_id
            ).all()
            
            for interaction in interactions:
                if interaction.interaction_type == 'watchlist':
                    user_data['in_watchlist'] = True
                elif interaction.interaction_type == 'favorite':
                    user_data['in_favorites'] = True
                elif interaction.interaction_type == 'rating':
                    user_data['user_rating'] = interaction.rating
                elif interaction.interaction_type == 'watch_status':
                    if interaction.interaction_metadata:
                        user_data['watch_status'] = interaction.interaction_metadata.get('status')
            
            # Check if user has reviewed
            if self.Review:
                user_review = self.Review.query.filter_by(
                    user_id=user_id,
                    content_id=content_id
                ).first()
                user_data['has_reviewed'] = user_review is not None
            
            return user_data
            
        except Exception as e:
            logger.error(f"Error getting user data: {e}")
            return {
                'in_watchlist': False,
                'in_favorites': False,
                'user_rating': None,
                'watch_status': None,
                'has_reviewed': False
            }
    
    # Continue with remaining helper methods...
    def _generate_conservative_title_variations(self, title: str) -> List[str]:
        """Generate conservative title variations for search"""
        try:
            variations = []
            title_lower = title.lower().strip()
            
            if not title_lower:
                return []
            
            variations.append(title_lower)
            
            # Remove articles
            for article in ['^the\\s+', '^a\\s+', '^an\\s+']:
                modified = re.sub(article, '', title_lower).strip()
                if modified and modified != title_lower:
                    variations.append(modified)
            
            # Handle punctuation
            no_punct = re.sub(r'[^\w\s]', '', title_lower).strip()
            if no_punct != title_lower:
                variations.append(no_punct)
            
            return variations[:5]  # Limit variations
            
        except Exception as e:
            logger.error(f"Error generating title variations: {e}")
            return [title.lower().strip()]
    
    def _should_auto_approve_review(self, user: Any, review_data: Dict) -> bool:
        """Determine if review should be auto-approved"""
        try:
            if user and getattr(user, 'is_admin', False):
                return True
            
            review_text = review_data.get('review_text', '')
            rating = review_data.get('rating', 0)
            
            # Basic quality checks
            if len(review_text.strip()) >= 20 and rating and 1 <= rating <= 10:
                if user and self.Review:
                    # Check user's review history
                    approved_count = self.Review.query.filter_by(
                        user_id=user.id,
                        is_approved=True
                    ).count()
                    
                    if approved_count >= 1:  # Trust users with approved reviews
                        return True
                    
                    if len(review_text.strip()) >= 50:  # Longer reviews for new users
                        return True
            
            return False  # Default to manual approval
            
        except Exception as e:
            logger.error(f"Error in auto-approval logic: {e}")
            return False
    
    def _clear_content_caches(self, slug: str):
        """Clear all caches related to a content item"""
        try:
            # Clear details cache
            self.cache_manager.clear_content_cache(slug)
            
            # Clear reviews cache patterns
            cache_patterns = [
                f"details:reviews:{slug}:*",
                f"details:similar:*",
                f"details:content:{slug}*"
            ]
            
            for pattern in cache_patterns:
                try:
                    if hasattr(self.cache_manager.cache, 'delete_pattern'):
                        self.cache_manager.cache.delete_pattern(pattern)
                except Exception as e:
                    logger.warning(f"Failed to clear cache pattern {pattern}: {e}")
                    
        except Exception as e:
            logger.error(f"Error clearing content caches: {e}")
    
    def _get_gallery(self, content_id: int) -> Dict:
        """Get image gallery for content"""
        try:
            content = self.Content.query.get(content_id)
            if not content or not content.tmdb_id:
                return self._empty_gallery()
            
            # Get images from TMDB
            try:
                tmdb_data = self.tmdb_service.get_content_details(content.tmdb_id, content.content_type)
                if tmdb_data and 'images' in tmdb_data:
                    return self._process_tmdb_images(tmdb_data['images'])
            except Exception as e:
                logger.warning(f"Failed to get TMDB images: {e}")
            
            return self._empty_gallery()
            
        except Exception as e:
            logger.error(f"Error getting gallery: {e}")
            return self._empty_gallery()
    
    def _process_tmdb_images(self, images_data: Dict) -> Dict:
        """Process TMDB images data"""
        try:
            gallery = self._empty_gallery()
            
            # Process posters
            for img in images_data.get('posters', [])[:15]:
                gallery['posters'].append({
                    'url': self._format_image_url(img['file_path'], 'poster'),
                    'thumbnail': self._format_image_url(img['file_path'], 'thumbnail'),
                    'aspect_ratio': img.get('aspect_ratio'),
                    'width': img.get('width'),
                    'height': img.get('height')
                })
            
            # Process backdrops
            for img in images_data.get('backdrops', [])[:15]:
                gallery['backdrops'].append({
                    'url': self._format_image_url(img['file_path'], 'backdrop'),
                    'thumbnail': self._format_image_url(img['file_path'], 'still'),
                    'aspect_ratio': img.get('aspect_ratio'),
                    'width': img.get('width'),
                    'height': img.get('height')
                })
            
            return gallery
            
        except Exception as e:
            logger.error(f"Error processing TMDB images: {e}")
            return self._empty_gallery()
    
    def _get_streaming_info(self, content: Any) -> Optional[Dict]:
        """Get streaming information"""
        try:
            if not content.tmdb_id:
                return None
            
            # This would need to be implemented with watch providers API
            # For now, return None
            return None
            
        except Exception as e:
            logger.error(f"Error getting streaming info: {e}")
            return None
    
    def _get_seasons_episodes(self, content: Any, tmdb_data: Dict) -> Optional[Dict]:
        """Get seasons and episodes information"""
        try:
            if content.content_type not in ['tv', 'anime']:
                return None
            
            if not tmdb_data:
                return None
            
            seasons_data = {
                'season_count': tmdb_data.get('number_of_seasons', 0),
                'episode_count': tmdb_data.get('number_of_episodes', 0),
                'seasons': []
            }
            
            # Process seasons
            for season in tmdb_data.get('seasons', [])[:15]:
                if season['season_number'] == 0 and len(tmdb_data.get('seasons', [])) > 1:
                    continue  # Skip specials if there are regular seasons
                
                seasons_data['seasons'].append({
                    'season_number': season['season_number'],
                    'name': season['name'],
                    'episode_count': season['episode_count'],
                    'air_date': season.get('air_date'),
                    'overview': season.get('overview'),
                    'poster_path': self._format_image_url(season.get('poster_path'), 'poster')
                })
            
            return seasons_data
            
        except Exception as e:
            logger.error(f"Error getting seasons/episodes: {e}")
            return None
    
    def _get_database_health(self) -> Dict:
        """Get database health metrics"""
        try:
            health = {
                'status': 'healthy',
                'content_count': 0,
                'person_count': 0,
                'review_count': 0
            }
            
            if self.Content:
                health['content_count'] = self.Content.query.count()
            
            if self.Person:
                health['person_count'] = self.Person.query.count()
            
            if self.Review:
                health['review_count'] = self.Review.query.count()
            
            return health
            
        except Exception as e:
            logger.error(f"Error getting database health: {e}")
            return {'status': 'unhealthy', 'error': str(e)}
    
    def _get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        try:
            return {
                'cache_hit_rate': self.cache_manager.get_cache_stats().get('hit_rate', 0),
                'active_threads': 8,  # ThreadPoolExecutor max_workers
                'optimization_level': 'high'
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def _format_image_url(self, path: Optional[str], image_type: str = 'poster') -> Optional[str]:
        """Format image URL"""
        if not path:
            return None
        
        if path.startswith('http'):
            return path
        
        try:
            size_mappings = {
                'poster': 'w500',
                'backdrop': 'w1280',
                'profile': 'w185',
                'still': 'w780',
                'thumbnail': 'w185',
                'logo': 'w92'
            }
            
            size = size_mappings.get(image_type, 'w500')
            return f"https://image.tmdb.org/t/p/{size}{path}"
            
        except Exception as e:
            logger.error(f"Error formatting image URL: {e}")
            return None
    
    def _empty_cast_crew(self) -> Dict:
        """Return empty cast/crew structure"""
        return {
            'cast': [],
            'crew': {
                'directors': [],
                'writers': [],
                'producers': [],
                'other_crew': []
            }
        }
    
    def _empty_gallery(self) -> Dict:
        """Return empty gallery structure"""
        return {
            'posters': [],
            'backdrops': [],
            'stills': []
        }
    
    def _get_minimal_person_details(self, person: Any) -> Dict:
        """Get minimal person details as fallback"""
        try:
            return {
                'id': person.id,
                'slug': person.slug or f"person-{person.id}",
                'name': person.name,
                'biography': person.biography or '',
                'profile_url': self._format_image_url(person.profile_path, 'profile'),
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
    
    def _map_genre_ids(self, genre_ids: List[int]) -> List[str]:
        """Map TMDB genre IDs to names"""
        genre_map = {
            28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
            80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
            14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
            9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]
    
    def _update_person_from_tmdb(self, person: Any, tmdb_data: Dict):
        """Update person with TMDB data"""
        try:
            if not tmdb_data:
                return
            
            updated = False
            
            # Update missing fields
            fields_to_update = [
                ('biography', 'biography'),
                ('place_of_birth', 'place_of_birth'),
                ('also_known_as', 'also_known_as')
            ]
            
            for person_field, tmdb_field in fields_to_update:
                if not getattr(person, person_field, None) and tmdb_data.get(tmdb_field):
                    if person_field == 'also_known_as':
                        setattr(person, person_field, json.dumps(tmdb_data[tmdb_field]))
                    else:
                        setattr(person, person_field, tmdb_data[tmdb_field])
                    updated = True
            
            # Update dates
            for date_field in ['birthday', 'deathday']:
                if not getattr(person, date_field, None) and tmdb_data.get(date_field):
                    try:
                        date_obj = datetime.strptime(tmdb_data[date_field], '%Y-%m-%d').date()
                        setattr(person, date_field, date_obj)
                        updated = True
                    except (ValueError, TypeError):
                        pass
            
            # Update popularity if higher
            if tmdb_data.get('popularity') and tmdb_data['popularity'] > (person.popularity or 0):
                person.popularity = tmdb_data['popularity']
                updated = True
            
            if updated:
                self.db.session.commit()
                logger.info(f"Updated person {person.name} with TMDB data")
                
        except Exception as e:
            logger.error(f"Error updating person from TMDB: {e}")
            self.db.session.rollback()
    
    def _get_person_images(self, tmdb_data: Dict) -> List[Dict]:
        """Get person images"""
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
        """Get person social media links"""
        try:
            social = {}
            
            if tmdb_data.get('external_ids'):
                ids = tmdb_data['external_ids']
                
                social_mappings = {
                    'twitter_id': ('twitter', 'https://twitter.com/{}', '@{}'),
                    'instagram_id': ('instagram', 'https://instagram.com/{}', '@{}'),
                    'facebook_id': ('facebook', 'https://facebook.com/{}', '{}'),
                    'imdb_id': ('imdb', 'https://www.imdb.com/name/{}', '{}'),
                    'tiktok_id': ('tiktok', 'https://tiktok.com/@{}', '@{}'),
                    'youtube_id': ('youtube', 'https://youtube.com/channel/{}', '{}')
                }
                
                for tmdb_key, (platform, url_template, handle_template) in social_mappings.items():
                    if ids.get(tmdb_key):
                        social[platform] = {
                            'url': url_template.format(ids[tmdb_key]),
                            'handle': handle_template.format(ids[tmdb_key]),
                            'platform': platform.title()
                        }
            
            return social
            
        except Exception as e:
            logger.error(f"Error getting person social media: {e}")
            return {}
    
    def _build_enhanced_personal_info(self, person: Any, tmdb_data: Dict, filmography: Dict) -> Dict:
        """Build enhanced personal information"""
        try:
            personal_info = {
                'age': None,
                'zodiac_sign': None,
                'nationality': None,
                'career_span': None,
                'status': 'Active'
            }
            
            # Calculate age
            if person.birthday:
                try:
                    birth_date = person.birthday
                    if person.deathday:
                        end_date = person.deathday
                        personal_info['status'] = 'Deceased'
                        personal_info['age'] = end_date.year - birth_date.year
                        if end_date.month < birth_date.month or (end_date.month == birth_date.month and end_date.day < birth_date.day):
                            personal_info['age'] -= 1
                    else:
                        today = datetime.now().date()
                        personal_info['age'] = today.year - birth_date.year
                        if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
                            personal_info['age'] -= 1
                    
                    # Zodiac sign
                    personal_info['zodiac_sign'] = self._get_zodiac_sign(birth_date.month, birth_date.day)
                except Exception as e:
                    logger.warning(f"Error calculating age: {e}")
            
            # Nationality from place of birth
            if person.place_of_birth:
                try:
                    place_parts = person.place_of_birth.split(',')
                    if place_parts:
                        personal_info['nationality'] = place_parts[-1].strip()
                except Exception:
                    pass
            
            # Career span from filmography
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
    
    def _build_enhanced_career_highlights(self, filmography: Dict) -> Dict:
        """Build enhanced career highlights"""
        try:
            highlights = {
                'debut_work': None,
                'latest_work': None,
                'most_successful_decade': None,
                'notable_achievements': []
            }
            
            stats = filmography.get('statistics', {})
            
            # Find debut work
            if stats.get('debut_year'):
                debut_year = stats['debut_year']
                for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer']:
                    for work in filmography.get(role_type, []):
                        if work.get('year') == debut_year:
                            highlights['debut_work'] = work
                            break
                    if highlights['debut_work']:
                        break
            
            # Find latest work
            if stats.get('latest_year'):
                latest_year = stats['latest_year']
                for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer']:
                    for work in filmography.get(role_type, []):
                        if work.get('year') == latest_year:
                            highlights['latest_work'] = work
                            break
                    if highlights['latest_work']:
                        break
            
            return highlights
            
        except Exception as e:
            logger.error(f"Error building enhanced career highlights: {e}")
            return {}
    
    def _get_zodiac_sign(self, month: int, day: int) -> str:
        """Get zodiac sign from birth date"""
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
    
    def _calculate_total_works(self, filmography: Dict) -> int:
        """Calculate total works from filmography"""
        try:
            total = 0
            for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer']:
                total += len(filmography.get(role_type, []))
            return total
        except:
            return 0