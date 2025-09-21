#backend/services/users.py
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import logging
import jwt
from functools import wraps
import os
import numpy as np

users_bp = Blueprint('users', __name__)

logger = logging.getLogger(__name__)

db = None
User = None
Content = None
UserInteraction = None
http_session = None
app = None
cache = None

ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-aksq.onrender.com')

def init_users(flask_app, database, models, services):
    global db, User, Content, UserInteraction
    global http_session, app, cache
    
    app = flask_app
    db = database
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    
    http_session = services['http_session']
    cache = services['cache']

class EnhancedMLServiceClient:
    
    @staticmethod
    def call_ml_service(endpoint, params=None, timeout=15, use_cache=True):
        try:
            if not ML_SERVICE_URL:
                logger.warning("ML service URL not configured")
                return None
            
            cache_key = f"ml_get:{endpoint}:{json.dumps(params, sort_keys=True)}"
            
            if use_cache and cache:
                cached_result = cache.get(cache_key)
                if cached_result:
                    logger.info(f"ML service cache hit for {endpoint}")
                    return cached_result
            
            url = f"{ML_SERVICE_URL}{endpoint}"
            response = http_session.get(url, params=params, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                if use_cache and cache:
                    cache.set(cache_key, result, timeout=1800)
                logger.info(f"ML service successful response for {endpoint}")
                return result
            else:
                logger.warning(f"ML service returned {response.status_code} for {endpoint}")
                return None
                
        except Exception as e:
            logger.warning(f"ML service call failed for {endpoint}: {e}")
            return None
    
    @staticmethod
    def call_ml_service_post(endpoint, data=None, timeout=30, use_cache=True):
        try:
            if not ML_SERVICE_URL:
                logger.warning("ML service URL not configured")
                return None
            
            cache_key = f"ml_post:{endpoint}:{json.dumps(data, sort_keys=True)}"
            
            if use_cache and cache:
                cached_result = cache.get(cache_key)
                if cached_result:
                    logger.info(f"ML service POST cache hit for {endpoint}")
                    return cached_result
            
            url = f"{ML_SERVICE_URL}{endpoint}"
            headers = {'Content-Type': 'application/json'}
            response = http_session.post(url, json=data, headers=headers, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                if use_cache and cache:
                    cache.set(cache_key, result, timeout=1200)
                logger.info(f"ML service POST successful response for {endpoint}")
                return result
            else:
                logger.warning(f"ML service POST returned {response.status_code} for {endpoint}")
                logger.warning(f"Response content: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.warning(f"ML service POST call failed for {endpoint}: {e}")
            return None
    
    @staticmethod
    def process_ml_recommendations(ml_response, limit=20):
        try:
            if not ml_response:
                logger.warning("Empty ML response received")
                return []
            
            if 'recommendations' not in ml_response:
                logger.warning("No recommendations in ML response")
                return []
            
            recommendations = []
            ml_recs = ml_response['recommendations'][:limit]
            
            content_ids = []
            for rec in ml_recs:
                if isinstance(rec, dict) and 'content_id' in rec:
                    content_ids.append(rec['content_id'])
                elif isinstance(rec, int):
                    content_ids.append(rec)
            
            if not content_ids:
                logger.warning("No valid content IDs in ML recommendations")
                return []
            
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            content_dict = {content.id: content for content in contents}
            
            for i, rec in enumerate(ml_recs):
                content_id = rec['content_id'] if isinstance(rec, dict) else rec
                content = content_dict.get(content_id)
                
                if content:
                    if not content.slug:
                        try:
                            content.ensure_slug()
                        except:
                            content.slug = f"content-{content.id}"
                    
                    content_data = {
                        'content': content,
                        'ml_score': rec.get('score', 0) if isinstance(rec, dict) else 0,
                        'ml_reason': rec.get('reason', 'ML recommendation') if isinstance(rec, dict) else 'ML recommendation',
                        'ml_rank': i + 1,
                        'confidence': rec.get('confidence', 0.5) if isinstance(rec, dict) else 0.5,
                        'content_type': rec.get('content_type', content.content_type) if isinstance(rec, dict) else content.content_type
                    }
                    recommendations.append(content_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error processing ML recommendations: {e}")
            return []
    
    @staticmethod
    def get_comprehensive_user_data(user_id):
        try:
            user = User.query.get(user_id)
            if not user:
                return None
            
            interactions = UserInteraction.query.filter_by(user_id=user_id).order_by(
                UserInteraction.timestamp.desc()
            ).all()
            
            try:
                preferred_languages = json.loads(user.preferred_languages) if user.preferred_languages else []
            except:
                preferred_languages = []
            
            try:
                preferred_genres = json.loads(user.preferred_genres) if user.preferred_genres else []
            except:
                preferred_genres = []
            
            interaction_data = []
            for interaction in interactions:
                interaction_dict = {
                    'content_id': interaction.content_id,
                    'interaction_type': interaction.interaction_type,
                    'timestamp': interaction.timestamp.isoformat(),
                    'rating': interaction.rating
                }
                
                if interaction.interaction_metadata:
                    try:
                        metadata = json.loads(interaction.interaction_metadata) if isinstance(interaction.interaction_metadata, str) else interaction.interaction_metadata
                        interaction_dict.update(metadata)
                    except:
                        pass
                
                interaction_data.append(interaction_dict)
            
            user_data = {
                'user_id': user_id,
                'username': user.username,
                'preferred_languages': preferred_languages,
                'preferred_genres': preferred_genres,
                'location': user.location,
                'account_created': user.created_at.isoformat(),
                'last_active': user.last_active.isoformat() if user.last_active else None,
                'interactions': interaction_data,
                'total_interactions': len(interaction_data),
                'interaction_summary': EnhancedMLServiceClient._generate_interaction_summary(interaction_data)
            }
            
            return user_data
            
        except Exception as e:
            logger.error(f"Error getting comprehensive user data: {e}")
            return None
    
    @staticmethod
    def _generate_interaction_summary(interactions):
        try:
            summary = {
                'total_count': len(interactions),
                'by_type': {},
                'rating_stats': {
                    'count': 0,
                    'average': 0,
                    'distribution': {}
                },
                'recent_activity': 0,
                'content_types_interacted': set(),
                'top_interaction_types': []
            }
            
            now = datetime.utcnow()
            ratings = []
            
            for interaction in interactions:
                interaction_type = interaction.get('interaction_type', 'unknown')
                summary['by_type'][interaction_type] = summary['by_type'].get(interaction_type, 0) + 1
                
                if interaction.get('rating'):
                    ratings.append(interaction['rating'])
                    rating_key = str(int(interaction['rating']))
                    summary['rating_stats']['distribution'][rating_key] = summary['rating_stats']['distribution'].get(rating_key, 0) + 1
                
                try:
                    interaction_time = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00'))
                    if (now - interaction_time).days <= 7:
                        summary['recent_activity'] += 1
                except:
                    pass
            
            if ratings:
                summary['rating_stats']['count'] = len(ratings)
                summary['rating_stats']['average'] = sum(ratings) / len(ratings)
            
            summary['top_interaction_types'] = sorted(
                summary['by_type'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating interaction summary: {e}")
            return {}
    
    @staticmethod
    def get_personalized_recommendations(user_id, limit=20, content_types=None):
        try:
            user_data = EnhancedMLServiceClient.get_comprehensive_user_data(user_id)
            if not user_data:
                logger.warning(f"No user data found for user {user_id}")
                return []
            
            if content_types:
                user_data['preferred_content_types'] = content_types
            
            endpoint = "/api/recommendations"
            params = {'limit': limit}
            
            ml_response = EnhancedMLServiceClient.call_ml_service_post(
                endpoint, user_data, timeout=30, use_cache=True
            )
            
            if ml_response and ml_response.get('recommendations'):
                recommendations = EnhancedMLServiceClient.process_ml_recommendations(ml_response, limit)
                
                if recommendations:
                    logger.info(f"Successfully got {len(recommendations)} ML recommendations for user {user_id}")
                    return recommendations
                else:
                    logger.warning(f"ML service returned empty recommendations for user {user_id}")
            else:
                logger.warning(f"ML service failed to return recommendations for user {user_id}")
            
            return EnhancedMLServiceClient._get_fallback_recommendations(user_data, limit)
        
        except Exception as e:
            logger.error(f"Error getting personalized recommendations for user {user_id}: {e}")
            return []
    
    @staticmethod
    def _get_fallback_recommendations(user_data, limit=20):
        try:
            preferred_genres = user_data.get('preferred_genres', [])
            preferred_languages = user_data.get('preferred_languages', [])
            
            query = Content.query
            
            if preferred_genres:
                genre_conditions = []
                for genre in preferred_genres:
                    genre_conditions.append(Content.genres.contains(genre))
                if genre_conditions:
                    query = query.filter(db.or_(*genre_conditions))
            
            if preferred_languages:
                lang_conditions = []
                for lang in preferred_languages:
                    lang_conditions.append(Content.languages.contains(lang))
                if lang_conditions:
                    query = query.filter(db.or_(*lang_conditions))
            
            fallback_content = query.order_by(
                Content.rating.desc(),
                Content.popularity.desc()
            ).limit(limit).all()
            
            recommendations = []
            for i, content in enumerate(fallback_content):
                if not content.slug:
                    try:
                        content.ensure_slug()
                    except:
                        content.slug = f"content-{content.id}"
                
                rec = {
                    'content': content,
                    'ml_score': 0.7 - (i * 0.01),
                    'ml_reason': 'Based on your preferences (fallback)',
                    'ml_rank': i + 1,
                    'confidence': 0.6,
                    'content_type': content.content_type
                }
                recommendations.append(rec)
            
            logger.info(f"Generated {len(recommendations)} fallback recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating fallback recommendations: {e}")
            return []
    
    @staticmethod
    def get_similar_content(content_id, limit=10):
        try:
            endpoint = f"/api/similar/{content_id}"
            params = {'limit': limit}
            ml_response = EnhancedMLServiceClient.call_ml_service(endpoint, params, timeout=15)
            
            if ml_response and ml_response.get('recommendations'):
                return EnhancedMLServiceClient.process_ml_recommendations(ml_response, limit)
            else:
                return EnhancedMLServiceClient._get_fallback_similar_content(content_id, limit)
        
        except Exception as e:
            logger.error(f"Error getting similar content for {content_id}: {e}")
            return []
    
    @staticmethod
    def _get_fallback_similar_content(content_id, limit=10):
        try:
            base_content = Content.query.get(content_id)
            if not base_content:
                return []
            
            try:
                base_genres = json.loads(base_content.genres) if base_content.genres else []
            except:
                base_genres = []
            
            similar_content = Content.query.filter(
                Content.id != content_id,
                Content.content_type == base_content.content_type
            ).order_by(Content.rating.desc()).limit(limit * 2).all()
            
            recommendations = []
            for content in similar_content:
                try:
                    content_genres = json.loads(content.genres) if content.genres else []
                    
                    if base_genres and content_genres:
                        common_genres = set(base_genres).intersection(set(content_genres))
                        if common_genres:
                            similarity_score = len(common_genres) / len(set(base_genres).union(set(content_genres)))
                            
                            if similarity_score > 0.2:
                                if not content.slug:
                                    try:
                                        content.ensure_slug()
                                    except:
                                        content.slug = f"content-{content.id}"
                                
                                rec = {
                                    'content': content,
                                    'ml_score': similarity_score,
                                    'ml_reason': f'Similar content (fallback)',
                                    'ml_rank': len(recommendations) + 1,
                                    'confidence': 0.6,
                                    'content_type': content.content_type
                                }
                                recommendations.append(rec)
                                
                                if len(recommendations) >= limit:
                                    break
                except:
                    continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating fallback similar content: {e}")
            return []
    
    @staticmethod
    def get_trending_recommendations(limit=20, region='IN'):
        try:
            endpoint = "/api/trending"
            params = {'limit': limit, 'region': region}
            ml_response = EnhancedMLServiceClient.call_ml_service(endpoint, params, timeout=15)
            
            if ml_response and ml_response.get('recommendations'):
                return EnhancedMLServiceClient.process_ml_recommendations(ml_response, limit)
            else:
                return EnhancedMLServiceClient._get_fallback_trending(limit)
        
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    @staticmethod
    def _get_fallback_trending(limit=20):
        try:
            trending_content = Content.query.filter(
                db.or_(
                    Content.is_trending == True,
                    Content.is_new_release == True,
                    Content.popularity > 50.0
                )
            ).order_by(
                Content.popularity.desc(),
                Content.rating.desc()
            ).limit(limit).all()
            
            recommendations = []
            for i, content in enumerate(trending_content):
                if not content.slug:
                    try:
                        content.ensure_slug()
                    except:
                        content.slug = f"content-{content.id}"
                
                rec = {
                    'content': content,
                    'ml_score': 0.8 - (i * 0.01),
                    'ml_reason': 'Trending content (fallback)',
                    'ml_rank': i + 1,
                    'confidence': 0.7,
                    'content_type': content.content_type
                }
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating fallback trending: {e}")
            return []

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': 'Invalid token'}), 401
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

@users_bp.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        required_fields = ['username', 'email', 'password']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=generate_password_hash(data['password']),
            preferred_languages=json.dumps(data.get('preferred_languages', [])),
            preferred_genres=json.dumps(data.get('preferred_genres', [])),
            location=data.get('location', '')
        )
        
        db.session.add(user)
        db.session.commit()
        
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.secret_key, algorithm='HS256')
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]')
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@users_bp.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Missing username or password'}), 400
        
        user = User.query.filter_by(username=data['username']).first()
        
        if not user or not check_password_hash(user.password_hash, data['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.secret_key, algorithm='HS256')
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]'),
                'location': user.location
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@users_bp.route('/api/interactions', methods=['POST'])
@require_auth
def record_interaction(current_user):
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'interaction_type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        content_id = data['content_id']
        interaction_type = data['interaction_type']
        
        if interaction_type == 'remove_watchlist':
            interaction = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=content_id,
                interaction_type='watchlist'
            ).first()
            
            if interaction:
                db.session.delete(interaction)
                db.session.commit()
                
                cache_key = f"personalized_v2:{current_user.id}:20"
                if cache:
                    cache.delete(cache_key)
                
                return jsonify({'message': 'Removed from watchlist'}), 200
            else:
                return jsonify({'message': 'Content not in watchlist'}), 404
        
        if interaction_type == 'watchlist':
            existing = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=content_id,
                interaction_type='watchlist'
            ).first()
            
            if existing:
                return jsonify({'message': 'Already in watchlist'}), 200
        
        metadata = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_agent': request.headers.get('User-Agent', ''),
            'ip_address': request.remote_addr
        }
        
        if 'search_query' in data:
            metadata['search_query'] = data['search_query']
        
        if 'recommendation_source' in data:
            metadata['recommendation_source'] = data['recommendation_source']
        
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type=interaction_type,
            rating=data.get('rating'),
            interaction_metadata=json.dumps(metadata)
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        cache_keys_to_invalidate = [
            f"personalized_v2:{current_user.id}:20",
            f"personalized_v2:{current_user.id}:10",
            f"similar_v2:{content_id}:10"
        ]
        
        if cache:
            for key in cache_keys_to_invalidate:
                cache.delete(key)
        
        current_user.last_active = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Interaction recorded successfully',
            'interaction_id': interaction.id
        }), 201
        
    except Exception as e:
        logger.error(f"Interaction recording error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record interaction'}), 500

@users_bp.route('/api/user/watchlist', methods=['GET'])
@require_auth
def get_watchlist(current_user):
    try:
        watchlist_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='watchlist'
        ).order_by(UserInteraction.timestamp.desc()).all()
        
        content_ids = [interaction.content_id for interaction in watchlist_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        content_dict = {content.id: content for content in contents}
        
        result = []
        for interaction in watchlist_interactions:
            content = content_dict.get(interaction.content_id)
            if content:
                if not content.slug:
                    try:
                        content.ensure_slug()
                    except:
                        content.slug = f"content-{content.id}"
                
                youtube_url = None
                if content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                try:
                    genres = json.loads(content.genres or '[]')
                except:
                    genres = []
                
                result.append({
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': genres,
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'youtube_trailer': youtube_url,
                    'added_to_watchlist': interaction.timestamp.isoformat()
                })
        
        return jsonify({
            'watchlist': result,
            'total_items': len(result)
        }), 200
        
    except Exception as e:
        logger.error(f"Watchlist error: {e}")
        return jsonify({'error': 'Failed to get watchlist'}), 500

@users_bp.route('/api/user/watchlist/<int:content_id>', methods=['DELETE'])
@require_auth
def remove_from_watchlist(current_user, content_id):
    try:
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='watchlist'
        ).first()
        
        if interaction:
            db.session.delete(interaction)
            db.session.commit()
            
            if cache:
                cache_key = f"personalized_v2:{current_user.id}:20"
                cache.delete(cache_key)
            
            return jsonify({'message': 'Removed from watchlist'}), 200
        else:
            return jsonify({'message': 'Content not in watchlist'}), 404
            
    except Exception as e:
        logger.error(f"Remove from watchlist error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to remove from watchlist'}), 500

@users_bp.route('/api/user/watchlist/<int:content_id>', methods=['GET'])
@require_auth
def check_watchlist_status(current_user, content_id):
    try:
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='watchlist'
        ).first()
        
        return jsonify({
            'in_watchlist': interaction is not None,
            'added_date': interaction.timestamp.isoformat() if interaction else None
        }), 200
        
    except Exception as e:
        logger.error(f"Check watchlist status error: {e}")
        return jsonify({'error': 'Failed to check watchlist status'}), 500

@users_bp.route('/api/user/favorites', methods=['GET'])
@require_auth
def get_favorites(current_user):
    try:
        favorite_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='favorite'
        ).order_by(UserInteraction.timestamp.desc()).all()
        
        content_ids = [interaction.content_id for interaction in favorite_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        content_dict = {content.id: content for content in contents}
        
        result = []
        for interaction in favorite_interactions:
            content = content_dict.get(interaction.content_id)
            if content:
                if not content.slug:
                    try:
                        content.ensure_slug()
                    except:
                        content.slug = f"content-{content.id}"
                
                youtube_url = None
                if content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                try:
                    genres = json.loads(content.genres or '[]')
                except:
                    genres = []
                
                result.append({
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': genres,
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'youtube_trailer': youtube_url,
                    'favorited_date': interaction.timestamp.isoformat()
                })
        
        return jsonify({
            'favorites': result,
            'total_items': len(result)
        }), 200
        
    except Exception as e:
        logger.error(f"Favorites error: {e}")
        return jsonify({'error': 'Failed to get favorites'}), 500

@users_bp.route('/api/recommendations/personalized', methods=['GET'])
@require_auth
def get_personalized_recommendations(current_user):
    try:
        limit = min(int(request.args.get('limit', 20)), 50)
        content_types = request.args.getlist('content_type')
        
        ml_recommendations = EnhancedMLServiceClient.get_personalized_recommendations(
            current_user.id, limit, content_types if content_types else None
        )
        
        if ml_recommendations:
            result = []
            for rec in ml_recommendations:
                content = rec['content']
                
                youtube_url = None
                if content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                try:
                    genres = json.loads(content.genres or '[]')
                except:
                    genres = []
                
                result.append({
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': genres,
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'youtube_trailer': youtube_url,
                    'ml_score': rec['ml_score'],
                    'ml_reason': rec['ml_reason'],
                    'ml_rank': rec['ml_rank'],
                    'confidence': rec['confidence'],
                    'recommendation_source': 'ml_personalized'
                })
            
            user_data = EnhancedMLServiceClient.get_comprehensive_user_data(current_user.id)
            interaction_summary = user_data.get('interaction_summary', {}) if user_data else {}
            
            return jsonify({
                'recommendations': result,
                'total_recommendations': len(result),
                'user_stats': {
                    'total_interactions': interaction_summary.get('total_count', 0),
                    'recent_activity': interaction_summary.get('recent_activity', 0),
                    'rating_average': round(interaction_summary.get('rating_stats', {}).get('average', 0), 1),
                    'top_interaction_types': interaction_summary.get('top_interaction_types', [])[:3]
                },
                'source': 'ml_service_enhanced',
                'algorithm': 'advanced_hybrid_personalized'
            }), 200
        
        return jsonify({
            'recommendations': [], 
            'source': 'ml_service_unavailable',
            'message': 'ML service temporarily unavailable'
        }), 200
        
    except Exception as e:
        logger.error(f"Personalized recommendations error: {e}")
        return jsonify({
            'recommendations': [], 
            'error': 'Failed to get personalized recommendations',
            'source': 'error'
        }), 500

@users_bp.route('/api/recommendations/ml-similar/<int:content_id>', methods=['GET'])
def get_ml_similar_recommendations(content_id):
    try:
        limit = min(int(request.args.get('limit', 10)), 20)
        
        ml_recommendations = EnhancedMLServiceClient.get_similar_content(content_id, limit)
        
        if ml_recommendations:
            result = []
            for rec in ml_recommendations:
                content = rec['content']
                
                youtube_url = None
                if content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                try:
                    genres = json.loads(content.genres or '[]')
                except:
                    genres = []
                
                result.append({
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': genres,
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'youtube_trailer': youtube_url,
                    'ml_score': rec['ml_score'],
                    'ml_reason': rec['ml_reason'],
                    'similarity_score': rec['ml_score'],
                    'match_type': 'ml_content_similarity'
                })
            
            return jsonify({
                'similar_content': result,
                'source': 'ml_service_enhanced',
                'algorithm': 'advanced_content_similarity',
                'base_content_id': content_id
            }), 200
        
        return jsonify({
            'similar_content': [],
            'source': 'ml_service_unavailable',
            'message': 'ML service temporarily unavailable'
        }), 200
        
    except Exception as e:
        logger.error(f"ML similar recommendations error: {e}")
        return jsonify({
            'similar_content': [],
            'error': 'Failed to get similar recommendations'
        }), 500

@users_bp.route('/api/recommendations/ml-trending', methods=['GET'])
def get_ml_trending_recommendations():
    try:
        limit = min(int(request.args.get('limit', 20)), 50)
        region = request.args.get('region', 'IN')
        
        ml_recommendations = EnhancedMLServiceClient.get_trending_recommendations(limit, region)
        
        if ml_recommendations:
            result = []
            for rec in ml_recommendations:
                content = rec['content']
                
                youtube_url = None
                if content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                try:
                    genres = json.loads(content.genres or '[]')
                except:
                    genres = []
                
                result.append({
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': genres,
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'youtube_trailer': youtube_url,
                    'ml_score': rec['ml_score'],
                    'ml_reason': rec['ml_reason'],
                    'trending_score': rec['ml_score'],
                    'confidence': rec['confidence']
                })
            
            return jsonify({
                'recommendations': result,
                'source': 'ml_service_enhanced',
                'region': region,
                'algorithm': 'advanced_trending_analysis'
            }), 200
        
        return jsonify({
            'recommendations': [],
            'source': 'ml_service_unavailable',
            'message': 'ML service temporarily unavailable'
        }), 200
        
    except Exception as e:
        logger.error(f"ML trending recommendations error: {e}")
        return jsonify({
            'recommendations': [],
            'error': 'Failed to get trending recommendations'
        }), 500

@users_bp.route('/api/user/profile/analysis', methods=['GET'])
@require_auth
def get_user_profile_analysis(current_user):
    try:
        user_data = EnhancedMLServiceClient.get_comprehensive_user_data(current_user.id)
        if not user_data:
            return jsonify({'error': 'User data not found'}), 404
        
        interaction_summary = user_data.get('interaction_summary', {})
        
        analysis = {
            'user_id': current_user.id,
            'username': current_user.username,
            'account_age_days': (datetime.utcnow() - current_user.created_at).days,
            'total_interactions': interaction_summary.get('total_count', 0),
            'recent_activity_week': interaction_summary.get('recent_activity', 0),
            'interaction_breakdown': interaction_summary.get('by_type', {}),
            'rating_behavior': interaction_summary.get('rating_stats', {}),
            'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
            'preferred_genres': json.loads(current_user.preferred_genres or '[]'),
            'engagement_level': 'high' if interaction_summary.get('total_count', 0) > 20 else 'medium' if interaction_summary.get('total_count', 0) > 5 else 'low',
            'recommendation_readiness': 'ready' if interaction_summary.get('total_count', 0) >= 3 else 'building_profile'
        }
        
        endpoint = "/api/analytics/user-profile"
        ml_analysis = EnhancedMLServiceClient.call_ml_service_post(
            endpoint, user_data, timeout=20, use_cache=True
        )
        
        if ml_analysis and 'insights' in ml_analysis:
            analysis['ml_insights'] = ml_analysis['insights']
            analysis['ml_profile'] = ml_analysis.get('user_profile', {})
        
        return jsonify({
            'profile_analysis': analysis,
            'data_source': 'enhanced_with_ml' if ml_analysis else 'basic_analysis'
        }), 200
        
    except Exception as e:
        logger.error(f"User profile analysis error: {e}")
        return jsonify({'error': 'Failed to analyze user profile'}), 500

@users_bp.route('/api/ml/health', methods=['GET'])
def check_ml_service_health():
    try:
        health_response = EnhancedMLServiceClient.call_ml_service('/health', use_cache=False, timeout=10)
        
        if health_response:
            return jsonify({
                'ml_service_status': 'healthy',
                'ml_service_url': ML_SERVICE_URL,
                'response': health_response,
                'features_available': health_response.get('features', {}),
                'libraries': health_response.get('libraries', {}),
                'last_checked': datetime.utcnow().isoformat()
            }), 200
        else:
            return jsonify({
                'ml_service_status': 'unhealthy',
                'ml_service_url': ML_SERVICE_URL,
                'error': 'No response from ML service',
                'fallback_available': True,
                'last_checked': datetime.utcnow().isoformat()
            }), 503
            
    except Exception as e:
        logger.error(f"ML service health check error: {e}")
        return jsonify({
            'ml_service_status': 'error',
            'ml_service_url': ML_SERVICE_URL,
            'error': str(e),
            'fallback_available': True,
            'last_checked': datetime.utcnow().isoformat()
        }), 503