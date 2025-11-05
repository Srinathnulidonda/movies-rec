# recommendation/trending.py
import logging
from flask import request, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict
import json
import time
from services.algorithms import RecommendationOrchestrator, EvaluationMetrics

logger = logging.getLogger(__name__)

def init_trending_routes(app, db, models, services, cache):
    """Initialize trending routes - uses existing algorithms and services"""
    
    def get_trending():
        try:
            category = request.args.get('category', 'all')
            limit = int(request.args.get('limit', 10))
            region = request.args.get('region', 'IN')
            apply_language_priority = request.args.get('language_priority', 'true').lower() == 'true'
            
            all_content = []
            
            # Get existing services
            TMDBService = services.get('TMDBService')
            JikanService = services.get('JikanService')
            content_service = services.get('ContentService')
            
            if not all([TMDBService, JikanService, content_service]):
                return jsonify({'error': 'Required CineBrain services not available'}), 503
            
            try:
                # Use existing TMDBService
                tmdb_movies = TMDBService.get_trending('movie', 'day')
                if tmdb_movies:
                    for item in tmdb_movies.get('results', []):
                        content = content_service.save_content_from_tmdb(item, 'movie')
                        if content:
                            all_content.append(content)
                
                tmdb_tv = TMDBService.get_trending('tv', 'day')
                if tmdb_tv:
                    for item in tmdb_tv.get('results', []):
                        content = content_service.save_content_from_tmdb(item, 'tv')
                        if content:
                            all_content.append(content)
            except Exception as e:
                logger.error(f"CineBrain TMDB fetch error: {e}")
            
            try:
                # Use existing JikanService
                top_anime = JikanService.get_top_anime()
                if top_anime:
                    for anime in top_anime.get('data', [])[:20]:
                        content = content_service.save_anime_content(anime)
                        if content:
                            all_content.append(content)
            except Exception as e:
                logger.error(f"CineBrain Jikan fetch error: {e}")
            
            # Get database trending
            Content = models['Content']
            db_trending = Content.query.filter_by(is_trending=True).limit(50).all()
            all_content.extend(db_trending)
            
            # Remove duplicates and ensure slugs
            seen_ids = set()
            unique_content = []
            for content in all_content:
                if content.id not in seen_ids:
                    seen_ids.add(content.id)
                    if not content.slug:
                        try:
                            content.ensure_slug()
                        except Exception as e:
                            logger.warning(f"CineBrain failed to ensure slug for content {content.id}: {e}")
                            content.slug = f"cinebrain-content-{content.id}"
                    unique_content.append(content)
            
            # Use existing RecommendationOrchestrator
            cinebrain_recommendation_orchestrator = RecommendationOrchestrator()
            categories = cinebrain_recommendation_orchestrator.get_trending_with_algorithms(
                unique_content,
                limit=limit,
                region=region,
                apply_language_priority=apply_language_priority
            )
            
            if category == 'all':
                response = {
                    'categories': categories,
                    'metadata': {
                        'total_content_analyzed': len(unique_content),
                        'region': region,
                        'language_priority_applied': apply_language_priority,
                        'algorithm': 'cinebrain_multi_level_ranking',
                        'timestamp': datetime.utcnow().isoformat(),
                        'cinebrain_service': 'trending'
                    }
                }
            else:
                category_map = {
                    'movies': 'trending_movies',
                    'tv_shows': 'trending_tv_shows',
                    'anime': 'trending_anime',
                    'nearby': 'popular_nearby',
                    'top10': 'top_10_today',
                    'critics': 'critics_choice'
                }
                
                selected_category = category_map.get(category, 'trending_movies')
                response = {
                    'category': category,
                    'recommendations': categories.get(selected_category, []),
                    'metadata': {
                        'total_content_analyzed': len(unique_content),
                        'region': region,
                        'language_priority_applied': apply_language_priority,
                        'algorithm': 'cinebrain_multi_level_ranking',
                        'timestamp': datetime.utcnow().isoformat(),
                        'cinebrain_service': 'trending'
                    }
                }
            
            # Add evaluation metrics if available
            if category != 'all' and selected_category in categories and categories[selected_category]:
                try:
                    content_items = categories[selected_category]
                    if content_items and len(content_items) > 0:
                        content_ids = []
                        for item in content_items:
                            if isinstance(item, dict) and 'id' in item:
                                content_ids.append(item['id'])
                        
                        if content_ids:
                            contents = Content.query.filter(Content.id.in_(content_ids)).all()
                            
                            response['metadata']['metrics'] = {
                                'diversity_score': round(EvaluationMetrics.diversity_score(contents), 3) if contents else 0,
                                'coverage_score': round(EvaluationMetrics.coverage_score(
                                    content_ids,
                                    Content.query.count()
                                ), 5) if Content.query.count() > 0 else 0
                            }
                except Exception as metric_error:
                    logger.warning(f"CineBrain metrics calculation error: {metric_error}")
            
            try:
                db.session.commit()
            except Exception as e:
                logger.warning(f"CineBrain failed to commit trending updates: {e}")
                db.session.rollback()
            
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"CineBrain trending recommendations error: {e}")
            return jsonify({'error': 'Failed to get CineBrain trending recommendations'}), 500

    return {
        'get_trending': get_trending
    }
