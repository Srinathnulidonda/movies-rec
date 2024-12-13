# recommendation/similar.py
import logging
import json
from flask import request, jsonify
from datetime import datetime
import hashlib
import time
from services.algorithms import RecommendationOrchestrator

logger = logging.getLogger(__name__)

def init_similar_routes(app, db, models, services, cache):
    """Initialize similar content routes - uses existing algorithms"""
    
    def get_similar_recommendations(content_id):
        try:
            limit = min(int(request.args.get('limit', 8)), 15)
            strict_mode = request.args.get('strict_mode', 'false').lower() == 'true'
            min_similarity = float(request.args.get('min_similarity', 0.3))
            
            cache_key = f"cinebrain:similar:{content_id}:{limit}:{strict_mode}:{min_similarity}"
            if cache:
                try:
                    cached_result = cache.get(cache_key)
                    if cached_result:
                        return jsonify(cached_result), 200
                except Exception as e:
                    logger.warning(f"CineBrain cache get error: {e}")
            
            Content = models['Content']
            base_content = Content.query.get(content_id)
            if not base_content:
                return jsonify({'error': 'CineBrain content not found'}), 404
            
            if not base_content.slug:
                try:
                    base_content.ensure_slug()
                    db.session.commit()
                except Exception as e:
                    logger.warning(f"CineBrain slug generation failed for base content: {e}")
                    base_content.slug = f"cinebrain-content-{base_content.id}"
            
            similar_content = []
            
            # Use existing RecommendationOrchestrator with UltraPowerfulSimilarityEngine
            try:
                recommendation_orchestrator = RecommendationOrchestrator()
                
                content_pool = Content.query.filter(
                    Content.id != content_id,
                    Content.content_type == base_content.content_type
                ).limit(200).all()
                
                ultra_results = recommendation_orchestrator.get_ultra_similar_content(
                    base_content_id=content_id,
                    content_pool=content_pool,
                    limit=limit,
                    strict_mode=strict_mode,
                    min_similarity=min_similarity
                )
                
                if ultra_results:
                    response = {
                        'base_content': {
                            'id': base_content.id,
                            'slug': base_content.slug or f"cinebrain-content-{base_content.id}",
                            'title': base_content.title,
                            'content_type': base_content.content_type,
                            'rating': base_content.rating
                        },
                        'similar_content': ultra_results,
                        'metadata': {
                            'algorithm': 'cinebrain_ultra_powerful_similarity_engine',
                            'total_results': len(ultra_results),
                            'similarity_threshold': min_similarity,
                            'strict_mode': strict_mode,
                            'timestamp': datetime.utcnow().isoformat(),
                            'cinebrain_service': 'similar_recommendations'
                        }
                    }
                    
                    if cache:
                        try:
                            cache.set(cache_key, response, timeout=900)
                        except Exception as e:
                            logger.warning(f"CineBrain caching failed: {e}")
                    
                    return jsonify(response), 200
                        
            except Exception as e:
                logger.warning(f"CineBrain ultra similarity engine error: {e}")
                # Fall back to simple similarity
            
            # Fallback to simple genre-based similarity
            try:
                try:
                    base_genres = json.loads(base_content.genres or '[]')
                except (json.JSONDecodeError, TypeError):
                    base_genres = []
                
                if base_genres:
                    primary_genre = base_genres[0]
                    
                    similar_items = Content.query.filter(
                        Content.id != content_id,
                        Content.content_type == base_content.content_type,
                        Content.genres.contains(primary_genre)
                    ).order_by(
                        Content.rating.desc()
                    ).limit(limit * 2).all()
                    
                    for item in similar_items[:limit]:
                        try:
                            if not item.slug:
                                item.slug = f"cinebrain-content-{item.id}"
                            
                            try:
                                item_genres = json.loads(item.genres or '[]')
                            except (json.JSONDecodeError, TypeError):
                                item_genres = []
                            
                            youtube_url = None
                            if item.youtube_trailer_id:
                                youtube_url = f"https://www.youtube.com/watch?v={item.youtube_trailer_id}"
                            
                            similar_content.append({
                                'id': item.id,
                                'slug': item.slug,
                                'title': item.title,
                                'poster_path': f"https://image.tmdb.org/t/p/w300{item.poster_path}" if item.poster_path and not item.poster_path.startswith('http') else item.poster_path,
                                'rating': item.rating,
                                'content_type': item.content_type,
                                'genres': item_genres,
                                'similarity_score': 0.8,
                                'match_type': 'cinebrain_genre_based',
                                'youtube_trailer': youtube_url
                            })
                            
                            if len(similar_content) >= limit:
                                break
                                
                        except Exception as e:
                            logger.warning(f"CineBrain error processing similar item {item.id}: {e}")
                            continue
                
                # Fallback if no genre matches
                if not similar_content:
                    fallback_items = Content.query.filter(
                        Content.id != content_id,
                        Content.content_type == base_content.content_type
                    ).order_by(
                        Content.popularity.desc()
                    ).limit(limit).all()
                    
                    for item in fallback_items:
                        if not item.slug:
                            item.slug = f"cinebrain-content-{item.id}"
                        
                        youtube_url = None
                        if item.youtube_trailer_id:
                            youtube_url = f"https://www.youtube.com/watch?v={item.youtube_trailer_id}"
                        
                        similar_content.append({
                            'id': item.id,
                            'slug': item.slug,
                            'title': item.title,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{item.poster_path}" if item.poster_path and not item.poster_path.startswith('http') else item.poster_path,
                            'rating': item.rating,
                            'content_type': item.content_type,
                            'similarity_score': 0.5,
                            'match_type': 'cinebrain_popularity_fallback',
                            'youtube_trailer': youtube_url
                        })
            
            except Exception as e:
                logger.error(f"CineBrain error in similarity calculation: {e}")
                similar_content = []
            
            # Record interaction
            try:
                session_id = hashlib.md5(f"anon_{request.remote_addr}_{time.time()}".encode()).hexdigest()
                
                AnonymousInteraction = models.get('AnonymousInteraction')
                if AnonymousInteraction:
                    interaction = AnonymousInteraction(
                        session_id=session_id,
                        content_id=content_id,
                        interaction_type='similar_view',
                        ip_address=request.remote_addr
                    )
                    db.session.add(interaction)
                    db.session.commit()
            except Exception as e:
                logger.warning(f"CineBrain interaction tracking failed: {e}")
            
            response = {
                'base_content': {
                    'id': base_content.id,
                    'slug': base_content.slug or f"cinebrain-content-{base_content.id}",
                    'title': base_content.title,
                    'content_type': base_content.content_type,
                    'rating': base_content.rating
                },
                'similar_content': similar_content,
                'metadata': {
                    'algorithm': 'cinebrain_optimized_genre_based',
                    'total_results': len(similar_content),
                    'similarity_threshold': min_similarity,
                    'timestamp': datetime.utcnow().isoformat(),
                    'cinebrain_service': 'similar_recommendations'
                }
            }
            
            if cache:
                try:
                    cache.set(cache_key, response, timeout=900)
                except Exception as e:
                    logger.warning(f"CineBrain caching failed: {e}")
            
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"CineBrain similar recommendations error: {e}")
            return jsonify({
                'error': 'Failed to get CineBrain similar recommendations',
                'similar_content': [],
                'metadata': {'error': str(e), 'cinebrain_service': 'similar_recommendations'}
            }), 500

    return {
        'get_similar_recommendations': get_similar_recommendations
    }
