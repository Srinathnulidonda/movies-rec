# recommendation/critics_choice.py
import logging
from flask import request, jsonify
import jwt

logger = logging.getLogger(__name__)

def init_critics_choice_routes(app, db, models, services, cache):
    """Initialize critics choice routes - delegates to existing service"""
    
    def get_enhanced_critics_choice():
        try:
            content_type = request.args.get('type', 'all')
            limit = int(request.args.get('limit', 20))
            genre = request.args.get('genre')
            language = request.args.get('language')
            time_period = request.args.get('time_period', 'all')
            region = request.args.get('region', 'global')
            
            # Use existing critics choice service
            critics_choice_service = services.get('critics_choice_service')
            if not critics_choice_service:
                return jsonify({
                    'error': 'CineBrain Critics Choice service not available',
                    'recommendations': [],
                    'cinebrain_service': 'enhanced_critics_choice'
                }), 503
            
            recommendations = critics_choice_service.get_enhanced_critics_choice(
                content_type=content_type,
                limit=limit,
                genre=genre,
                language=language,
                time_period=time_period,
                region=region
            )
            
            return jsonify({
                'recommendations': recommendations['items'],
                'metadata': recommendations['metadata'],
                'cinebrain_service': 'enhanced_critics_choice'
            }), 200
            
        except Exception as e:
            logger.error(f"CineBrain enhanced critics choice endpoint error: {e}")
            return jsonify({
                'error': 'Failed to get CineBrain critics choice',
                'recommendations': [],
                'cinebrain_service': 'enhanced_critics_choice'
            }), 500

    def trigger_critics_refresh():
        try:
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'error': 'Authorization required'}), 401
            
            token = auth_header.split(' ')[1]
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            user_id = payload.get('user_id')
            
            user = models['User'].query.get(user_id)
            if not user or not user.is_admin:
                return jsonify({'error': 'Admin access required'}), 403
                
            # Use existing critics choice service
            critics_choice_service = services.get('critics_choice_service')
            if not critics_choice_service:
                return jsonify({'error': 'Critics Choice service not available'}), 503
                
            result = critics_choice_service.trigger_manual_refresh()
            return jsonify(result), 200
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        except Exception as e:
            logger.error(f"CineBrain critics refresh trigger error: {e}")
            return jsonify({'error': 'Failed to trigger refresh'}), 500

    def get_critics_status():
        try:
            # Use existing critics choice service
            critics_choice_service = services.get('critics_choice_service')
            if not critics_choice_service:
                return jsonify({'error': 'Critics Choice service not available'}), 503
                
            refresh_info = critics_choice_service.get_refresh_info()
            return jsonify({
                'success': True,
                'refresh_info': refresh_info,
                'cinebrain_service': 'critics_choice_status'
            }), 200
            
        except Exception as e:
            logger.error(f"CineBrain critics status error: {e}")
            return jsonify({'error': 'Failed to get status'}), 500

    return {
        'get_enhanced_critics_choice': get_enhanced_critics_choice,
        'trigger_critics_refresh': trigger_critics_refresh,
        'get_critics_status': get_critics_status
    }
