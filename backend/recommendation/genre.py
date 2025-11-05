# recommendation/genre.py
import logging
import json
from flask import request, jsonify
from collections import Counter
import hashlib
import time

logger = logging.getLogger(__name__)

# Use existing language configurations from algorithms.py
CINEBRAIN_LANGUAGE_PRIORITY = {
    'first': ['telugu', 'english', 'hindi'],
    'second': ['malayalam', 'kannada', 'tamil'],
    'codes': {
        'telugu': 'te',
        'english': 'en',
        'hindi': 'hi',
        'malayalam': 'ml',
        'kannada': 'kn',
        'tamil': 'ta'
    }
}

CINEBRAIN_ANIME_GENRES = {
    'shonen': ['Action', 'Adventure', 'Martial Arts', 'School', 'Shounen'],
    'shojo': ['Romance', 'Drama', 'School', 'Slice of Life', 'Shoujo'],
    'seinen': ['Action', 'Drama', 'Thriller', 'Psychological', 'Seinen'],
    'josei': ['Romance', 'Drama', 'Slice of Life', 'Josei'],
    'kodomomuke': ['Kids', 'Family', 'Adventure', 'Comedy']
}

def init_genre_routes(app, db, models, services, cache):
    """Initialize genre, regional, and anime recommendation routes - uses existing services"""
    
    def get_genre_recommendations(genre):
        try:
            content_type = request.args.get('type', 'movie')
            limit = int(request.args.get('limit', 20))
            region = request.args.get('region')
            
            genre_ids = {
                'action': 28, 'adventure': 12, 'animation': 16, 'biography': -1,
                'comedy': 35, 'crime': 80, 'documentary': 99, 'drama': 18,
                'fantasy': 14, 'horror': 27, 'musical': 10402, 'mystery': 9648,
                'romance': 10749, 'sci-fi': 878, 'science fiction': 878, 'thriller': 53, 'western': 37
            }
            
            genre_id = genre_ids.get(genre.lower())
            recommendations = []
            
            # Use existing services
            TMDBService = services.get('TMDBService')
            content_service = services.get('ContentService')
            
            if not TMDBService or not content_service:
                return jsonify({'error': 'Required CineBrain services not available'}), 503
            
            if genre_id and genre_id != -1:
                genre_content = TMDBService.get_by_genre(genre_id, content_type, region=region)
                
                if genre_content:
                    for item in genre_content.get('results', [])[:limit]:
                        content = content_service.save_content_from_tmdb(item, content_type)
                        if content:
                            youtube_url = None
                            if content.youtube_trailer_id:
                                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                            
                            recommendations.append({
                                'id': content.id,
                                'slug': content.slug,
                                'title': content.title,
                                'content_type': content.content_type,
                                'genres': json.loads(content.genres or '[]'),
                                'rating': content.rating,
                                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                                'overview': content.overview[:150] + '...' if content.overview else '',
                                'youtube_trailer': youtube_url
                            })
            
            return jsonify({
                'recommendations': recommendations,
                'cinebrain_service': 'genre_recommendations'
            }), 200
            
        except Exception as e:
            logger.error(f"CineBrain genre recommendations error: {e}")
            return jsonify({'error': 'Failed to get CineBrain genre recommendations'}), 500

    def get_regional_recommendations(language):
        try:
            content_type = request.args.get('type', 'movie')
            limit = int(request.args.get('limit', 20))
            
            lang_code = CINEBRAIN_LANGUAGE_PRIORITY['codes'].get(language.lower())
            recommendations = []
            
            # Use existing services
            TMDBService = services.get('TMDBService')
            content_service = services.get('ContentService')
            
            if not TMDBService or not content_service:
                return jsonify({'error': 'Required CineBrain services not available'}), 503
            
            if lang_code:
                lang_content = TMDBService.get_language_specific(lang_code, content_type)
                if lang_content:
                    for item in lang_content.get('results', [])[:limit]:
                        content = content_service.save_content_from_tmdb(item, content_type)
                        if content:
                            youtube_url = None
                            if content.youtube_trailer_id:
                                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                            
                            recommendations.append({
                                'id': content.id,
                                'slug': content.slug,
                                'title': content.title,
                                'content_type': content.content_type,
                                'genres': json.loads(content.genres or '[]'),
                                'rating': content.rating,
                                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                                'overview': content.overview[:150] + '...' if content.overview else '',
                                'youtube_trailer': youtube_url
                            })
            
            return jsonify({
                'recommendations': recommendations,
                'cinebrain_service': 'regional_recommendations'
            }), 200
            
        except Exception as e:
            logger.error(f"CineBrain regional recommendations error: {e}")
            return jsonify({'error': 'Failed to get CineBrain regional recommendations'}), 500

    def get_anime_recommendations():
        try:
            genre = request.args.get('genre')
            limit = int(request.args.get('limit', 20))
            
            recommendations = []
            
            # Use existing services
            JikanService = services.get('JikanService')
            content_service = services.get('ContentService')
            
            if not JikanService or not content_service:
                return jsonify({'error': 'Required CineBrain services not available'}), 503
            
            if genre and genre.lower() in CINEBRAIN_ANIME_GENRES:
                genre_keywords = CINEBRAIN_ANIME_GENRES[genre.lower()]
                for keyword in genre_keywords[:2]:
                    anime_results = JikanService.get_anime_by_genre(keyword)
                    if anime_results:
                        for anime in anime_results.get('data', []):
                            if len(recommendations) >= limit:
                                break
                            content = content_service.save_anime_content(anime)
                            if content:
                                youtube_url = None
                                if content.youtube_trailer_id:
                                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                                
                                recommendations.append({
                                    'id': content.id,
                                    'slug': content.slug,
                                    'mal_id': content.mal_id,
                                    'title': content.title,
                                    'original_title': content.original_title,
                                    'content_type': content.content_type,
                                    'genres': json.loads(content.genres or '[]'),
                                    'anime_genres': json.loads(content.anime_genres or '[]'),
                                    'rating': content.rating,
                                    'poster_path': content.poster_path,
                                    'overview': content.overview[:150] + '...' if content.overview else '',
                                    'youtube_trailer': youtube_url
                                })
                        if len(recommendations) >= limit:
                            break
            else:
                # Use existing JikanService
                top_anime = JikanService.get_top_anime()
                if top_anime:
                    for anime in top_anime.get('data', [])[:limit]:
                        content = content_service.save_anime_content(anime)
                        if content:
                            youtube_url = None
                            if content.youtube_trailer_id:
                                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                            
                            recommendations.append({
                                'id': content.id,
                                'slug': content.slug,
                                'mal_id': content.mal_id,
                                'title': content.title,
                                'original_title': content.original_title,
                                'content_type': content.content_type,
                                'genres': json.loads(content.genres or '[]'),
                                'anime_genres': json.loads(content.anime_genres or '[]'),
                                'rating': content.rating,
                                'poster_path': content.poster_path,
                                'overview': content.overview[:150] + '...' if content.overview else '',
                                'youtube_trailer': youtube_url
                            })
            
            return jsonify({
                'recommendations': recommendations[:limit],
                'cinebrain_service': 'anime_recommendations'
            }), 200
            
        except Exception as e:
            logger.error(f"CineBrain anime recommendations error: {e}")
            return jsonify({'error': 'Failed to get CineBrain anime recommendations'}), 500

    def get_anonymous_recommendations():
        try:
            session_id = hashlib.md5(f"anon_{request.remote_addr}_{time.time()}".encode()).hexdigest()
            limit = int(request.args.get('limit', 20))
            
            # Use existing anonymous recommendation engine if available
            if hasattr(services.get('algorithms'), 'CineBrainAnonymousRecommendationEngine'):
                anonymous_engine = services.get('algorithms').CineBrainAnonymousRecommendationEngine
                recommendations = anonymous_engine.get_recommendations_for_anonymous(
                    session_id, request.remote_addr, limit
                )
            else:
                # Fallback to simple recommendations
                Content = models['Content']
                recommendations = _get_simple_anonymous_recommendations(Content, limit)
            
            result = []
            for content in recommendations:
                youtube_url = None
                if content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                result.append({
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'youtube_trailer': youtube_url
                })
            
            return jsonify({
                'recommendations': result,
                'cinebrain_service': 'anonymous_recommendations'
            }), 200
            
        except Exception as e:
            logger.error(f"CineBrain anonymous recommendations error: {e}")
            return jsonify({'error': 'Failed to get CineBrain recommendations'}), 500

    def get_admin_choice_recommendations():
        try:
            limit = int(request.args.get('limit', 20))
            rec_type = request.args.get('type', 'admin_choice')
            
            AdminRecommendation = models.get('AdminRecommendation')
            Content = models['Content']
            User = models['User']
            
            if not AdminRecommendation:
                return jsonify({
                    'recommendations': [],
                    'cinebrain_service': 'admin_recommendations'
                }), 200
            
            admin_recs = AdminRecommendation.query.filter_by(
                is_active=True,
                recommendation_type=rec_type
            ).order_by(AdminRecommendation.created_at.desc()).limit(limit).all()
            
            result = []
            for rec in admin_recs:
                content = Content.query.get(rec.content_id)
                admin = User.query.get(rec.admin_id)
                
                if content:
                    if not content.slug:
                        try:
                            content.ensure_slug()
                        except Exception as e:
                            logger.warning(f"CineBrain failed to ensure slug for admin rec content: {e}")
                            content.slug = f"cinebrain-content-{content.id}"
                    
                    youtube_url = None
                    if content.youtube_trailer_id:
                        youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                    
                    result.append({
                        'id': content.id,
                        'slug': content.slug,
                        'title': content.title,
                        'content_type': content.content_type,
                        'genres': json.loads(content.genres or '[]'),
                        'rating': content.rating,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                        'overview': content.overview[:150] + '...' if content.overview else '',
                        'youtube_trailer': youtube_url,
                        'admin_description': rec.description,
                        'admin_name': admin.username if admin else 'CineBrain Admin',
                        'recommended_at': rec.created_at.isoformat()
                    })
            
            return jsonify({
                'recommendations': result,
                'cinebrain_service': 'admin_recommendations'
            }), 200
            
        except Exception as e:
            logger.error(f"CineBrain public admin recommendations error: {e}")
            return jsonify({'error': 'Failed to get CineBrain admin recommendations'}), 500

    def _get_simple_anonymous_recommendations(Content, limit):
        """Simple fallback for anonymous recommendations"""
        try:
            # Get trending content
            trending = Content.query.filter_by(is_trending=True).limit(limit//2).all()
            
            # Get highly rated content
            highly_rated = Content.query.filter(
                Content.rating >= 7.5
            ).order_by(Content.rating.desc()).limit(limit//2).all()
            
            # Combine and remove duplicates
            all_content = trending + highly_rated
            seen_ids = set()
            unique_content = []
            
            for content in all_content:
                if content.id not in seen_ids:
                    seen_ids.add(content.id)
                    if not content.slug:
                        try:
                            content.ensure_slug()
                        except:
                            content.slug = f"cinebrain-content-{content.id}"
                    unique_content.append(content)
                    
                    if len(unique_content) >= limit:
                        break
            
            return unique_content
        except Exception as e:
            logger.error(f"Simple anonymous recommendations error: {e}")
            return []

    return {
        'get_genre_recommendations': get_genre_recommendations,
        'get_regional_recommendations': get_regional_recommendations,
        'get_anime_recommendations': get_anime_recommendations,
        'get_anonymous_recommendations': get_anonymous_recommendations,
        'get_admin_choice_recommendations': get_admin_choice_recommendations
    }