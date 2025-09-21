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

class AdvancedMLServiceClient:
    
    @staticmethod
    def call_ml_service(endpoint, params=None, timeout=20, use_cache=True):
        try:
            if not ML_SERVICE_URL:
                logger.warning("ML service URL not configured")
                return None
            
            cache_key = f"ml_advanced_get:{endpoint}:{json.dumps(params, sort_keys=True)}"
            
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
                logger.warning(f"Response: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.warning(f"ML service call failed for {endpoint}: {e}")
            return None
    
    @staticmethod
    def call_ml_service_post(endpoint, data=None, timeout=35, use_cache=True):
        try:
            if not ML_SERVICE_URL:
                logger.warning("ML service URL not configured")
                return None
            
            if data:
                data = AdvancedMLServiceClient._ensure_json_serializable(data)
            
            cache_key = f"ml_advanced_post:{endpoint}:{json.dumps(data, sort_keys=True)}"
            
            if use_cache and cache:
                cached_result = cache.get(cache_key)
                if cached_result:
                    logger.info(f"ML service POST cache hit for {endpoint}")
                    return cached_result
            
            url = f"{ML_SERVICE_URL}{endpoint}"
            headers = {'Content-Type': 'application/json'}
            
            try:
                json_data = json.dumps(data)
                logger.debug(f"Sending data to ML service: {len(json_data)} characters")
            except (TypeError, ValueError) as e:
                logger.error(f"JSON serialization error: {e}")
                return None
            
            response = http_session.post(url, json=data, headers=headers, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                if use_cache and cache:
                    cache.set(cache_key, result, timeout=1200)
                logger.info(f"ML service POST successful response for {endpoint}")
                return result
            else:
                logger.warning(f"ML service POST returned {response.status_code} for {endpoint}")
                logger.warning(f"Response content: {response.text[:300]}")
                return None
                
        except Exception as e:
            logger.warning(f"ML service POST call failed for {endpoint}: {e}")
            return None
    
    @staticmethod
    def _ensure_json_serializable(obj):
        if isinstance(obj, dict):
            return {key: AdvancedMLServiceClient._ensure_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [AdvancedMLServiceClient._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, (datetime, )):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    @staticmethod
    def process_advanced_ml_recommendations(ml_response, limit=20):
        try:
            if not ml_response:
                logger.warning("Empty ML response received")
                return []
            
            if 'recommendations' not in ml_response:
                logger.warning("No recommendations in ML response")
                return []
            
            recommendations = []
            ml_recs = ml_response['recommendations'][:limit]
            metadata = ml_response.get('metadata', {})
            
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
                        'ml_reason': rec.get('reason', 'AI recommendation') if isinstance(rec, dict) else 'AI recommendation',
                        'ml_rank': rec.get('rank', i + 1) if isinstance(rec, dict) else i + 1,
                        'confidence': rec.get('confidence', 0.5) if isinstance(rec, dict) else 0.5,
                        'content_type': rec.get('content_type', content.content_type) if isinstance(rec, dict) else content.content_type,
                        'personalization_factors': rec.get('personalization_factors', {}) if isinstance(rec, dict) else {},
                        'metadata': metadata
                    }
                    recommendations.append(content_data)
            
            logger.info(f"Processed {len(recommendations)} advanced ML recommendations")
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
            ).limit(1000).all()
            
            try:
                preferred_languages = json.loads(user.preferred_languages) if user.preferred_languages else []
            except:
                preferred_languages = []
            
            try:
                preferred_genres = json.loads(user.preferred_genres) if user.preferred_genres else []
            except:
                preferred_genres = []
            
            interaction_data = []
            content_types_seen = set()
            
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
                        for key, value in metadata.items():
                            if isinstance(value, set):
                                metadata[key] = list(value)
                        interaction_dict.update(metadata)
                    except:
                        pass
                
                content = Content.query.get(interaction.content_id)
                if content:
                    interaction_dict['content_type'] = content.content_type
                    content_types_seen.add(content.content_type)
                
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
                'content_types_experienced': list(content_types_seen),
                'interaction_summary': AdvancedMLServiceClient._generate_interaction_summary(interaction_data)
            }
            
            user_data = AdvancedMLServiceClient._ensure_json_serializable(user_data)
            
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
                    'distribution': {},
                    'high_ratings': 0
                },
                'recent_activity': 0,
                'content_types_interacted': [],
                'top_interaction_types': [],
                'engagement_score': 0,
                'preference_indicators': {}
            }
            
            now = datetime.utcnow()
            ratings = []
            content_types_seen = set()
            high_engagement_interactions = 0
            
            for interaction in interactions:
                interaction_type = interaction.get('interaction_type', 'unknown')
                summary['by_type'][interaction_type] = summary['by_type'].get(interaction_type, 0) + 1
                
                if interaction.get('rating'):
                    rating = interaction['rating']
                    ratings.append(rating)
                    rating_key = str(int(rating))
                    summary['rating_stats']['distribution'][rating_key] = summary['rating_stats']['distribution'].get(rating_key, 0) + 1
                    if rating >= 4:
                        summary['rating_stats']['high_ratings'] += 1
                
                try:
                    interaction_time = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00'))
                    if (now - interaction_time).days <= 7:
                        summary['recent_activity'] += 1
                except:
                    pass
                
                content_type = interaction.get('content_type', 'unknown')
                content_types_seen.add(content_type)
                
                if interaction_type in ['like', 'favorite', 'watchlist']:
                    high_engagement_interactions += 1
            
            if ratings:
                summary['rating_stats']['count'] = len(ratings)
                summary['rating_stats']['average'] = sum(ratings) / len(ratings)
            
            summary['top_interaction_types'] = sorted(
                summary['by_type'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            summary['content_types_interacted'] = list(content_types_seen)
            summary['engagement_score'] = high_engagement_interactions / len(interactions) if interactions else 0
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating interaction summary: {e}")
            return {
                'total_count': 0,
                'by_type': {},
                'rating_stats': {'count': 0, 'average': 0, 'distribution': {}, 'high_ratings': 0},
                'recent_activity': 0,
                'content_types_interacted': [],
                'top_interaction_types': [],
                'engagement_score': 0,
                'preference_indicators': {}
            }
    
    @staticmethod
    def get_advanced_personalized_recommendations(user_id, limit=20, content_types=None):
        try:
            user_data = AdvancedMLServiceClient.get_comprehensive_user_data(user_id)
            if not user_data:
                logger.warning(f"No user data found for user {user_id}")
                return []
            
            if content_types:
                user_data['preferred_content_types'] = content_types
            
            endpoint = "/api/recommendations"
            params = {'limit': limit}
            
            ml_response = AdvancedMLServiceClient.call_ml_service_post(
                endpoint, user_data, timeout=35, use_cache=True
            )
            
            if ml_response and ml_response.get('recommendations'):
                recommendations = AdvancedMLServiceClient.process_advanced_ml_recommendations(ml_response, limit)
                
                if recommendations:
                    logger.info(f"Successfully got {len(recommendations)} advanced ML recommendations for user {user_id}")
                    return recommendations
                else:
                    logger.warning(f"ML service returned empty recommendations for user {user_id}")
            else:
                logger.warning(f"ML service failed to return recommendations for user {user_id}")
            
            return AdvancedMLServiceClient._get_intelligent_fallback_recommendations(user_data, limit)
        
        except Exception as e:
            logger.error(f"Error getting advanced personalized recommendations for user {user_id}: {e}")
            return []
    
    @staticmethod
    def _get_intelligent_fallback_recommendations(user_data, limit=20):
        try:
            preferred_genres = user_data.get('preferred_genres', [])
            preferred_languages = user_data.get('preferred_languages', [])
            interaction_summary = user_data.get('interaction_summary', {})
            content_types_experienced = user_data.get('content_types_experienced', [])
            
            query = Content.query
            
            filters_applied = []
            
            if preferred_genres:
                genre_conditions = []
                for genre in preferred_genres:
                    genre_conditions.append(Content.genres.contains(genre))
                if genre_conditions:
                    query = query.filter(db.or_(*genre_conditions))
                    filters_applied.append(f"genres: {', '.join(preferred_genres[:3])}")
            
            if preferred_languages:
                lang_conditions = []
                for lang in preferred_languages:
                    lang_conditions.append(Content.languages.contains(lang))
                if lang_conditions:
                    query = query.filter(db.or_(*lang_conditions))
                    filters_applied.append(f"languages: {', '.join(preferred_languages[:2])}")
            
            if content_types_experienced:
                query = query.filter(Content.content_type.in_(content_types_experienced))
                filters_applied.append(f"content types: {', '.join(content_types_experienced)}")
            
            engagement_score = interaction_summary.get('engagement_score', 0)
            if engagement_score > 0.3:
                query = query.filter(Content.rating >= 7.0)
                filters_applied.append("high quality (7.0+ rating)")
            else:
                query = query.filter(Content.rating >= 6.0)
                filters_applied.append("good quality (6.0+ rating)")
            
            fallback_content = query.order_by(
                Content.popularity.desc(),
                Content.rating.desc(),
                Content.vote_count.desc()
            ).limit(limit * 2).all()
            
            if not fallback_content:
                fallback_content = Content.query.filter(
                    Content.rating >= 6.5
                ).order_by(
                    Content.popularity.desc()
                ).limit(limit).all()
                filters_applied = ["popular content (fallback)"]
            
            recommendations = []
            for i, content in enumerate(fallback_content[:limit]):
                if not content.slug:
                    try:
                        content.ensure_slug()
                    except:
                        content.slug = f"content-{content.id}"
                
                base_score = 0.8 - (i * 0.01)
                reason_parts = ["Based on your preferences"]
                if filters_applied:
                    reason_parts.append(f"({', '.join(filters_applied[:2])})")
                
                rec = {
                    'content': content,
                    'ml_score': base_score,
                    'ml_reason': ' '.join(reason_parts),
                    'ml_rank': i + 1,
                    'confidence': 0.7,
                    'content_type': content.content_type,
                    'personalization_factors': {
                        'preference_match': True,
                        'quality_filter': True,
                        'popularity_boost': content.popularity > 50,
                        'fallback_applied': True
                    },
                    'metadata': {
                        'algorithm': 'intelligent_fallback',
                        'filters_applied': filters_applied
                    }
                }
                recommendations.append(rec)
            
            logger.info(f"Generated {len(recommendations)} intelligent fallback recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating intelligent fallback recommendations: {e}")
            return []
    
    @staticmethod
    def get_advanced_similar_content(content_id, limit=10):
        try:
            endpoint = f"/api/similar/{content_id}"
            params = {'limit': limit}
            ml_response = AdvancedMLServiceClient.call_ml_service(endpoint, params, timeout=20)
            
            if ml_response and ml_response.get('recommendations'):
                return AdvancedMLServiceClient.process_advanced_ml_recommendations(ml_response, limit)
            else:
                return AdvancedMLServiceClient._get_advanced_fallback_similar_content(content_id, limit)
        
        except Exception as e:
            logger.error(f"Error getting advanced similar content for {content_id}: {e}")
            return []
    
    @staticmethod
    def _get_advanced_fallback_similar_content(content_id, limit=10):
        try:
            base_content = Content.query.get(content_id)
            if not base_content:
                return []
            
            try:
                base_genres = json.loads(base_content.genres) if base_content.genres else []
            except:
                base_genres = []
            
            try:
                base_languages = json.loads(base_content.languages) if base_content.languages else []
            except:
                base_languages = []
            
            similar_content = Content.query.filter(
                Content.id != content_id,
                Content.content_type == base_content.content_type
            ).order_by(Content.rating.desc(), Content.popularity.desc()).limit(limit * 3).all()
            
            recommendations = []
            for content in similar_content:
                try:
                    content_genres = json.loads(content.genres) if content.genres else []
                    content_languages = json.loads(content.languages) if content.languages else []
                    
                    similarity_score = 0.0
                    similarity_factors = []
                    
                    if base_genres and content_genres:
                        common_genres = set(base_genres).intersection(set(content_genres))
                        if common_genres:
                            genre_similarity = len(common_genres) / len(set(base_genres).union(set(content_genres)))
                            similarity_score += genre_similarity * 0.5
                            similarity_factors.append(f"shared genres: {', '.join(list(common_genres)[:2])}")
                    
                    if base_languages and content_languages:
                        common_languages = set(base_languages).intersection(set(content_languages))
                        if common_languages:
                            lang_similarity = len(common_languages) / len(set(base_languages).union(set(content_languages)))
                            similarity_score += lang_similarity * 0.3
                            similarity_factors.append(f"same language")
                    
                    if base_content.rating and content.rating:
                        rating_similarity = 1.0 - abs(base_content.rating - content.rating) / 10.0
                        similarity_score += rating_similarity * 0.2
                        if abs(base_content.rating - content.rating) <= 1.0:
                            similarity_factors.append("similar quality")
                    
                    if similarity_score > 0.3:
                        if not content.slug:
                            try:
                                content.ensure_slug()
                            except:
                                content.slug = f"content-{content.id}"
                        
                        reason = f"Similar content ({', '.join(similarity_factors[:2])})" if similarity_factors else "Similar content"
                        
                        rec = {
                            'content': content,
                            'ml_score': similarity_score,
                            'ml_reason': reason,
                            'ml_rank': len(recommendations) + 1,
                            'confidence': 0.7,
                            'content_type': content.content_type,
                            'personalization_factors': {
                                'genre_match': bool(set(base_genres).intersection(set(content_genres)) if base_genres and content_genres else False),
                                'language_match': bool(set(base_languages).intersection(set(content_languages)) if base_languages and content_languages else False),
                                'quality_match': abs(base_content.rating - content.rating) <= 1.0 if base_content.rating and content.rating else False
                            }
                        }
                        recommendations.append(rec)
                        
                        if len(recommendations) >= limit:
                            break
                except Exception as e:
                    logger.warning(f"Error processing similar content {content.id}: {e}")
                    continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating advanced fallback similar content: {e}")
            return []
    
    @staticmethod
    def get_advanced_trending_recommendations(limit=20, region='IN'):
        try:
            endpoint = "/api/trending"
            params = {'limit': limit, 'region': region}
            ml_response = AdvancedMLServiceClient.call_ml_service(endpoint, params, timeout=20)
            
            if ml_response and ml_response.get('recommendations'):
                return AdvancedMLServiceClient.process_advanced_ml_recommendations(ml_response, limit)
            else:
                return AdvancedMLServiceClient._get_advanced_fallback_trending(limit, region)
        
        except Exception as e:
            logger.error(f"Error getting advanced trending recommendations: {e}")
            return []
    
    @staticmethod
    def _get_advanced_fallback_trending(limit=20, region='IN'):
        try:
            trending_content = Content.query.filter(
                db.or_(
                    Content.is_trending == True,
                    Content.is_new_release == True,
                    Content.popularity > 70.0,
                    db.and_(Content.rating > 7.5, Content.vote_count > 500)
                )
            ).order_by(
                Content.popularity.desc(),
                Content.rating.desc(),
                Content.vote_count.desc()
            ).limit(limit * 2).all()
            
            recommendations = []
            for i, content in enumerate(trending_content[:limit]):
                if not content.slug:
                    try:
                        content.ensure_slug()
                    except:
                        content.slug = f"content-{content.id}"
                
                score = 0.8 - (i * 0.01)
                
                trending_factors = []
                if content.is_trending:
                    trending_factors.append("trending")
                    score += 0.1
                if content.is_new_release:
                    trending_factors.append("new release")
                    score += 0.05
                if content.popularity > 70:
                    trending_factors.append("popular")
                if content.rating > 7.5:
                    trending_factors.append("highly rated")
                
                reason = f"Trending content ({', '.join(trending_factors[:2])})" if trending_factors else "Trending content"
                
                rec = {
                    'content': content,
                    'ml_score': score,
                    'ml_reason': reason,
                    'ml_rank': i + 1,
                    'confidence': 0.8,
                    'content_type': content.content_type,
                    'personalization_factors': {
                        'trending': content.is_trending,
                        'new_release': content.is_new_release,
                        'popular': content.popularity > 70,
                        'high_quality': content.rating > 7.5
                    }
                }
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating advanced fallback trending: {e}")
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
            preferred_languages=json.dumps(data.get('preferred_languages', ['english', 'telugu'])),
            preferred_genres=json.dumps(data.get('preferred_genres', ['Action', 'Drama', 'Comedy'])),
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
                
                cache_keys_to_invalidate = [
                    f"ml_advanced_post:/api/recommendations:{current_user.id}",
                    f"advanced_personalized:{current_user.id}:20",
                    f"advanced_personalized:{current_user.id}:10"
                ]
                
                if cache:
                    for key_pattern in cache_keys_to_invalidate:
                        cache.delete(key_pattern)
                
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
            'ip_address': request.remote_addr,
            'interaction_context': 'web_interface'
        }
        
        if 'search_query' in data:
            metadata['search_query'] = data['search_query']
        
        if 'recommendation_source' in data:
            metadata['recommendation_source'] = data['recommendation_source']
        
        if 'content_context' in data:
            metadata['content_context'] = data['content_context']
        
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
            f"ml_advanced_post:/api/recommendations:{current_user.id}",
            f"advanced_personalized:{current_user.id}:20",
            f"advanced_personalized:{current_user.id}:10",
            f"advanced_similar:{content_id}:10"
        ]
        
        if cache:
            for key_pattern in cache_keys_to_invalidate:
                cache.delete(key_pattern)
        
        current_user.last_active = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Interaction recorded successfully',
            'interaction_id': interaction.id,
            'cache_invalidated': True
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
                cache_keys = [
                    f"ml_advanced_post:/api/recommendations:{current_user.id}",
                    f"advanced_personalized:{current_user.id}:20"
                ]
                for key in cache_keys:
                    cache.delete(key)
            
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
        
        ml_recommendations = AdvancedMLServiceClient.get_advanced_personalized_recommendations(
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
                    'personalization_factors': rec['personalization_factors'],
                    'recommendation_source': 'advanced_ml_personalized'
                })
            
            user_data = AdvancedMLServiceClient.get_comprehensive_user_data(current_user.id)
            interaction_summary = user_data.get('interaction_summary', {}) if user_data else {}
            
            metadata = ml_recommendations[0].get('metadata', {}) if ml_recommendations else {}
            
            return jsonify({
                'recommendations': result,
                'total_recommendations': len(result),
                'user_stats': {
                    'total_interactions': interaction_summary.get('total_count', 0),
                    'recent_activity': interaction_summary.get('recent_activity', 0),
                    'rating_average': round(interaction_summary.get('rating_stats', {}).get('average', 0), 1),
                    'engagement_score': round(interaction_summary.get('engagement_score', 0), 2),
                    'top_interaction_types': interaction_summary.get('top_interaction_types', [])[:3]
                },
                'source': 'advanced_ml_service',
                'algorithm': metadata.get('algorithm', 'advanced_hybrid_neural_personalized'),
                'personalization_score': metadata.get('personalization_score', 0),
                'advanced_features': {
                    'neural_enhancement': metadata.get('neural_enhancement', False),
                    'content_breadth': metadata.get('content_breadth', 0),
                    'preference_strength': metadata.get('preference_strength', 0),
                    'discovery_efficiency': metadata.get('discovery_efficiency', 0)
                }
            }), 200
        
        return jsonify({
            'recommendations': [], 
            'source': 'ml_service_unavailable',
            'message': 'Advanced ML service temporarily unavailable'
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
        
        ml_recommendations = AdvancedMLServiceClient.get_advanced_similar_content(content_id, limit)
        
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
                    'match_type': 'advanced_ml_similarity',
                    'personalization_factors': rec['personalization_factors']
                })
            
            return jsonify({
                'similar_content': result,
                'source': 'advanced_ml_service',
                'algorithm': 'advanced_multi_factor_similarity',
                'base_content_id': content_id
            }), 200
        
        return jsonify({
            'similar_content': [],
            'source': 'ml_service_unavailable',
            'message': 'Advanced ML service temporarily unavailable'
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
        
        ml_recommendations = AdvancedMLServiceClient.get_advanced_trending_recommendations(limit, region)
        
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
                    'confidence': rec['confidence'],
                    'personalization_factors': rec['personalization_factors']
                })
            
            return jsonify({
                'recommendations': result,
                'source': 'advanced_ml_service',
                'region': region,
                'algorithm': 'advanced_multi_factor_trending'
            }), 200
        
        return jsonify({
            'recommendations': [],
            'source': 'ml_service_unavailable',
            'message': 'Advanced ML service temporarily unavailable'
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
        user_data = AdvancedMLServiceClient.get_comprehensive_user_data(current_user.id)
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
            'engagement_score': interaction_summary.get('engagement_score', 0),
            'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
            'preferred_genres': json.loads(current_user.preferred_genres or '[]'),
            'content_types_experienced': user_data.get('content_types_experienced', []),
            'engagement_level': 'high' if interaction_summary.get('engagement_score', 0) > 0.4 else 'medium' if interaction_summary.get('engagement_score', 0) > 0.2 else 'low',
            'recommendation_readiness': 'advanced' if interaction_summary.get('total_count', 0) >= 10 else 'ready' if interaction_summary.get('total_count', 0) >= 3 else 'building_profile'
        }
        
        endpoint = "/api/analytics/user-profile"
        ml_analysis = AdvancedMLServiceClient.call_ml_service_post(
            endpoint, user_data, timeout=25, use_cache=True
        )
        
        if ml_analysis and 'insights' in ml_analysis:
            analysis['ml_insights'] = ml_analysis['insights']
            analysis['ml_profile'] = ml_analysis.get('user_profile', {})
        
        return jsonify({
            'profile_analysis': analysis,
            'data_source': 'advanced_with_ml' if ml_analysis else 'basic_analysis',
            'advanced_features': {
                'neural_profiling': bool(ml_analysis),
                'behavioral_analysis': True,
                'personalization_ready': analysis['recommendation_readiness'] in ['ready', 'advanced']
            }
        }), 200
        
    except Exception as e:
        logger.error(f"User profile analysis error: {e}")
        return jsonify({'error': 'Failed to analyze user profile'}), 500

@users_bp.route('/api/ml/health', methods=['GET'])
def check_ml_service_health():
    try:
        health_response = AdvancedMLServiceClient.call_ml_service('/health', use_cache=False, timeout=15)
        
        if health_response:
            return jsonify({
                'ml_service_status': 'healthy',
                'ml_service_url': ML_SERVICE_URL,
                'response': health_response,
                'features_available': health_response.get('features', {}),
                'algorithms': health_response.get('algorithms', {}),
                'libraries': health_response.get('libraries', {}),
                'advanced_capabilities': {
                    'neural_collaborative_filtering': health_response.get('features', {}).get('neural_collaborative_filtering', False),
                    'matrix_factorization': health_response.get('features', {}).get('matrix_factorization', False),
                    'advanced_personalization': health_response.get('features', {}).get('advanced_personalization', False),
                    'contextual_recommendations': health_response.get('features', {}).get('contextual_popularity_scoring', False)
                },
                'last_checked': datetime.utcnow().isoformat()
            }), 200
        else:
            return jsonify({
                'ml_service_status': 'unhealthy',
                'ml_service_url': ML_SERVICE_URL,
                'error': 'No response from advanced ML service',
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