# recommendation/upcoming.py
import asyncio
import logging
from flask import request, jsonify
from services.upcoming import UpcomingContentService

logger = logging.getLogger(__name__)

def init_upcoming_routes(app, db, models, services, cache):
    """Initialize upcoming content routes - delegates to existing service"""
    
    async def get_upcoming_releases_async():
        try:
            region = request.args.get('region', 'IN')
            timezone_name = request.args.get('timezone', 'Asia/Kolkata')
            categories_param = request.args.get('categories', 'movies,tv,anime')
            use_cache = request.args.get('use_cache', 'true').lower() == 'true'
            include_analytics = request.args.get('include_analytics', 'true').lower() == 'true'
            
            categories = [cat.strip() for cat in categories_param.split(',')]
            
            if len(region) != 2:
                return jsonify({'error': 'Invalid region code for CineBrain'}), 400
            
            # Use existing UpcomingContentService directly
            service = UpcomingContentService(
                tmdb_api_key=app.config['TMDB_API_KEY'],
                cache_backend=cache,
                enable_analytics=include_analytics
            )
            
            try:
                results = await service.get_upcoming_releases(
                    region=region.upper(),
                    timezone_name=timezone_name,
                    categories=categories,
                    use_cache=use_cache,
                    include_analytics=include_analytics
                )
                
                return jsonify({
                    'success': True,
                    'data': results,
                    'cinebrain_telugu_priority': True,
                    'cinebrain_service': 'upcoming'
                }), 200
                
            finally:
                await service.close()
        
        except Exception as e:
            logger.error(f"CineBrain upcoming releases error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'cinebrain_service': 'upcoming'
            }), 500

    def get_upcoming_releases():
        try:
            region = request.args.get('region', 'IN')
            timezone_name = request.args.get('timezone', 'Asia/Kolkata')
            categories_param = request.args.get('categories', 'movies,tv,anime')
            use_cache = request.args.get('use_cache', 'true').lower() == 'true'
            include_analytics = request.args.get('include_analytics', 'true').lower() == 'true'
            
            categories = [cat.strip() for cat in categories_param.split(',')]
            
            if len(region) != 2:
                return jsonify({'error': 'Invalid region code for CineBrain'}), 400
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Use existing UpcomingContentService directly
                service = UpcomingContentService(
                    tmdb_api_key=app.config['TMDB_API_KEY'],
                    cache_backend=cache,
                    enable_analytics=include_analytics
                )
                
                results = loop.run_until_complete(
                    service.get_upcoming_releases(
                        region=region.upper(),
                        timezone_name=timezone_name,
                        categories=categories,
                        use_cache=use_cache,
                        include_analytics=include_analytics
                    )
                )
                
                loop.run_until_complete(service.close())
                
                return jsonify({
                    'success': True,
                    'data': results,
                    'cinebrain_telugu_priority': True,
                    'cinebrain_service': 'upcoming'
                }), 200
                
            finally:
                loop.close()
        
        except Exception as e:
            logger.error(f"CineBrain upcoming sync error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'cinebrain_service': 'upcoming'
            }), 500

    return {
        'get_upcoming_releases_async': get_upcoming_releases_async,
        'get_upcoming_releases': get_upcoming_releases
    }
