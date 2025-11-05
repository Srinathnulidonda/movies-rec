# recommendation/new_releases.py
import logging
from flask import request, jsonify
import jwt
import threading
import time

logger = logging.getLogger(__name__)

def init_new_releases_routes(app, db, models, services, cache):
    """Initialize new releases routes - delegates to existing service"""
    
    def get_new_releases():
        try:
            force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
            admin_override = request.args.get('admin_refresh', 'false').lower() == 'true'
            
            if admin_override:
                auth_header = request.headers.get('Authorization')
                if auth_header and auth_header.startswith('Bearer '):
                    token = auth_header.split(' ')[1]
                    try:
                        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
                        user_id = payload.get('user_id')
                        user = models['User'].query.get(user_id)
                        if user and user.is_admin:
                            force_refresh = True
                            logger.info(f"CineBrain admin {user.username} triggered new releases refresh")
                    except:
                        pass
            
            # Use existing service
            cinebrain_new_releases_service = services.get('new_releases_service')
            if cinebrain_new_releases_service:
                data = cinebrain_new_releases_service.get_new_releases(force_refresh=force_refresh)
                
                response_data = {
                    'success': True,
                    'cinebrain_service': 'new_releases',
                    'data': data,
                    'stats': cinebrain_new_releases_service.get_stats()
                }
                
                return jsonify(response_data), 200
            else:
                logger.error("CineBrain new releases service not available")
                return jsonify({
                    'success': False,
                    'error': 'CineBrain new releases service unavailable',
                    'cinebrain_service': 'new_releases'
                }), 503
                
        except Exception as e:
            logger.error(f"CineBrain new releases endpoint error: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get CineBrain new releases',
                'cinebrain_service': 'new_releases'
            }), 500

    def get_new_releases_stats():
        try:
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'error': 'CineBrain authentication required'}), 401
            
            token = auth_header.split(' ')[1]
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            user_id = payload.get('user_id')
            
            user = models['User'].query.get(user_id)
            if not user or not user.is_admin:
                return jsonify({'error': 'CineBrain admin access required'}), 403
                
            # Use existing service
            cinebrain_new_releases_service = services.get('new_releases_service')
            if cinebrain_new_releases_service:
                stats = cinebrain_new_releases_service.get_stats()
                return jsonify({
                    'success': True,
                    'cinebrain_service': 'new_releases',
                    'stats': stats
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': 'CineBrain service not available'
                }), 503
                
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'CineBrain token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid CineBrain token'}), 401
        except Exception as e:
            logger.error(f"CineBrain new releases stats error: {e}")
            return jsonify({'error': 'Failed to get CineBrain stats'}), 500

    def trigger_new_releases_refresh():
        try:
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'error': 'CineBrain authentication required'}), 401
            
            token = auth_header.split(' ')[1]
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            user_id = payload.get('user_id')
            
            user = models['User'].query.get(user_id)
            if not user or not user.is_admin:
                return jsonify({'error': 'CineBrain admin access required'}), 403
                
            # Use existing service
            cinebrain_new_releases_service = services.get('new_releases_service')
            if cinebrain_new_releases_service:
                threading.Thread(
                    target=cinebrain_new_releases_service.refresh_new_releases,
                    daemon=True,
                    name=f'CineBrainManualRefresh_{int(time.time())}'
                ).start()
                
                return jsonify({
                    'success': True,
                    'message': 'CineBrain new releases refresh triggered',
                    'cinebrain_service': 'new_releases'
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': 'CineBrain service not available'
                }), 503
                
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'CineBrain token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid CineBrain token'}), 401
        except Exception as e:
            logger.error(f"CineBrain manual refresh error: {e}")
            return jsonify({'error': 'Failed to trigger CineBrain refresh'}), 500

    def update_new_releases_config():
        try:
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'error': 'CineBrain authentication required'}), 401
            
            token = auth_header.split(' ')[1]
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            user_id = payload.get('user_id')
            
            user = models['User'].query.get(user_id)
            if not user or not user.is_admin:
                return jsonify({'error': 'CineBrain admin access required'}), 403
                
            # Use existing service
            cinebrain_new_releases_service = services.get('new_releases_service')
            if not cinebrain_new_releases_service:
                return jsonify({
                    'success': False,
                    'error': 'CineBrain service not available'
                }), 503
                
            config_updates = request.json
            allowed_updates = {
                'language_priorities', 'refresh_interval_minutes', 
                'date_range_days', 'max_items_per_category'
            }
            
            valid_updates = {k: v for k, v in config_updates.items() if k in allowed_updates}
            
            if valid_updates:
                cinebrain_new_releases_service.update_config(**valid_updates)
                return jsonify({
                    'success': True,
                    'message': 'CineBrain configuration updated',
                    'updates': valid_updates,
                    'cinebrain_service': 'new_releases'
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': 'No valid CineBrain configuration updates provided'
                }), 400
                
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'CineBrain token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid CineBrain token'}), 401
        except Exception as e:
            logger.error(f"CineBrain config update error: {e}")
            return jsonify({'error': 'Failed to update CineBrain configuration'}), 500

    return {
        'get_new_releases': get_new_releases,
        'get_new_releases_stats': get_new_releases_stats,
        'trigger_new_releases_refresh': trigger_new_releases_refresh,
        'update_new_releases_config': update_new_releases_config
    }

