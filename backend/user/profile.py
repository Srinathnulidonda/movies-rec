# backend/user/profile.py

from flask import request, jsonify
from datetime import datetime
import json
import logging
from .utils import require_auth, db, User, UserInteraction, Content, get_enhanced_user_stats, recommendation_engine, profile_analyzer, cache_get, cache_set, cache_delete, get_cache_key

logger = logging.getLogger(__name__)

@require_auth
def get_user_profile(current_user):
    try:
        cache_key = get_cache_key('user_profile', current_user.id)
        cached_profile = cache_get(cache_key)
        
        if cached_profile:
            return jsonify(cached_profile), 200
        
        stats = get_enhanced_user_stats(current_user.id)
        
        recent_interactions = []
        if UserInteraction:
            try:
                recent = UserInteraction.query.filter_by(
                    user_id=current_user.id
                ).order_by(UserInteraction.timestamp.desc()).limit(10).all()
                
                for interaction in recent:
                    content = Content.query.get(interaction.content_id) if Content else None
                    recent_interactions.append({
                        'id': interaction.id,
                        'interaction_type': interaction.interaction_type,
                        'timestamp': interaction.timestamp.isoformat(),
                        'rating': interaction.rating,
                        'content': {
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                            'slug': content.slug
                        } if content else None
                    })
            except Exception as e:
                logger.warning(f"Could not get CineBrain recent activity: {e}")
        
        rec_effectiveness = {}
        profile_insights = {}
        
        try:
            if recommendation_engine:
                rec_effectiveness = recommendation_engine.get_user_recommendation_metrics(current_user.id)
        except Exception as e:
            logger.warning(f"Could not get CineBrain recommendation effectiveness: {e}")
        
        try:
            if profile_analyzer:
                profile_insights = profile_analyzer.build_user_profile(current_user.id)
        except Exception as e:
            logger.warning(f"Could not get CineBrain profile insights: {e}")
        
        profile_fields = {
            'preferred_languages': current_user.preferred_languages,
            'preferred_genres': current_user.preferred_genres,
            'location': current_user.location,
            'avatar_url': current_user.avatar_url
        }
        
        completed_fields = [field for field, value in profile_fields.items() if value]
        completion_score = min(100, len(completed_fields) * 25)
        missing_fields = [field for field, value in profile_fields.items() if not value]
        
        ai_insights = {}
        if profile_insights:
            ai_insights = {
                'cinematic_dna': profile_insights.get('cinematic_dna', {}),
                'user_segment': profile_insights.get('user_segment', 'new_user'),
                'profile_confidence': profile_insights.get('confidence_score', 0),
                'diversity_score': profile_insights.get('diversity_score', 0),
                'engagement_level': profile_insights.get('engagement_metrics', {}).get('engagement_score', 0),
                'behavioral_patterns': profile_insights.get('behavioral_patterns', {}),
                'preference_evolution': profile_insights.get('temporal_patterns', {}),
                'recommendation_accuracy': rec_effectiveness.get('accuracy_score', 0) if rec_effectiveness else 0
            }
        
        profile_data = {
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'is_admin': current_user.is_admin,
                'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
                'preferred_genres': json.loads(current_user.preferred_genres or '[]'),
                'location': current_user.location,
                'avatar_url': current_user.avatar_url,
                'created_at': current_user.created_at.isoformat(),
                'last_active': current_user.last_active.isoformat() if current_user.last_active else None
            },
            'stats': stats,
            'recent_activity': recent_interactions,
            'recommendation_effectiveness': rec_effectiveness,
            'ai_insights': ai_insights,
            'profile_completion': {
                'score': completion_score,
                'missing_fields': missing_fields,
                'suggestions': [
                    'Add preferred languages to get better CineBrain recommendations',
                    'Select favorite genres to improve CineBrain content discovery',
                    'Add your location for regional CineBrain content suggestions',
                    'Upload an avatar to personalize your CineBrain profile'
                ][:len(missing_fields)]
            },
            'personalization_status': {
                'is_ready': ai_insights.get('profile_confidence', 0) > 0.3,
                'confidence_level': ai_insights.get('profile_confidence', 0),
                'recommendations_available': bool(rec_effectiveness),
                'profile_strength': 'strong' if ai_insights.get('profile_confidence', 0) > 0.7 else 'developing'
            }
        }
        
        cache_set(cache_key, profile_data, timeout=600)
        
        return jsonify(profile_data), 200
        
    except Exception as e:
        logger.error(f"CineBrain profile error: {e}")
        return jsonify({'error': 'Failed to get CineBrain user profile'}), 500

@require_auth
def update_user_profile(current_user):
    try:
        data = request.get_json()
        
        updated_fields = []
        
        if 'preferred_languages' in data:
            current_user.preferred_languages = json.dumps(data['preferred_languages'])
            updated_fields.append('preferred_languages')
        
        if 'preferred_genres' in data:
            current_user.preferred_genres = json.dumps(data['preferred_genres'])
            updated_fields.append('preferred_genres')
        
        if 'location' in data:
            current_user.location = data['location']
            updated_fields.append('location')
        
        if 'avatar_url' in data:
            current_user.avatar_url = data['avatar_url']
            updated_fields.append('avatar_url')
        
        db.session.commit()
        
        cache_key = get_cache_key('user_profile', current_user.id)
        cache_delete(cache_key)
        
        if recommendation_engine and updated_fields:
            try:
                recommendation_engine.update_user_feedback(
                    current_user.id,
                    0,
                    'profile_update',
                    {
                        'updated_fields': updated_fields,
                        'data': data
                    }
                )
                logger.info(f"Updated CineBrain recommendation engine for user {current_user.id}")
            except Exception as e:
                logger.warning(f"Failed to update CineBrain recommendation engine: {e}")
        
        if profile_analyzer and updated_fields:
            try:
                profile_analyzer.update_profile_realtime(
                    current_user.id,
                    {
                        'interaction_type': 'profile_update',
                        'metadata': {
                            'updated_fields': updated_fields,
                            'data': data
                        }
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update CineBrain profile analyzer: {e}")
        
        return jsonify({
            'success': True,
            'message': f'CineBrain profile updated successfully. Updated: {", ".join(updated_fields)}',
            'updated_fields': updated_fields,
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
                'preferred_genres': json.loads(current_user.preferred_genres or '[]'),
                'location': current_user.location,
                'avatar_url': current_user.avatar_url
            },
            'recommendations_refreshed': bool(recommendation_engine),
            'profile_analyzed': bool(profile_analyzer)
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain profile update error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update CineBrain profile'}), 500

def get_public_profile(username):
    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        public_stats = get_enhanced_user_stats(user.id)
        
        return jsonify({
            'user': {
                'id': user.id,
                'username': user.username,
                'avatar_url': user.avatar_url,
                'created_at': user.created_at.isoformat(),
                'last_active': user.last_active.isoformat() if user.last_active else None
            },
            'public_stats': {
                'total_interactions': public_stats.get('total_interactions', 0),
                'favorites': public_stats.get('favorites', 0),
                'ratings_given': public_stats.get('ratings_given', 0),
                'user_segment': public_stats.get('user_segment', 'new_user'),
                'engagement_level': public_stats.get('engagement_metrics', {}).get('engagement_score', 0)
            }
        }), 200
    except Exception as e:
        logger.error(f"Error getting public profile: {e}")
        return jsonify({'error': 'Failed to get profile'}), 500

@require_auth
def update_user_preferences(current_user):
    try:
        data = request.get_json()
        
        if 'preferred_languages' in data:
            current_user.preferred_languages = json.dumps(data['preferred_languages'])
        
        if 'preferred_genres' in data:
            current_user.preferred_genres = json.dumps(data['preferred_genres'])
        
        db.session.commit()
        
        cache_key = get_cache_key('user_profile', current_user.id)
        cache_delete(cache_key)
        
        if recommendation_engine:
            try:
                recommendation_engine.update_user_feedback(
                    current_user.id,
                    0,
                    'preference_update',
                    {
                        'updated_languages': data.get('preferred_languages'),
                        'updated_genres': data.get('preferred_genres'),
                        'source': 'explicit_preference_update'
                    }
                )
                logger.info(f"Successfully updated CineBrain preferences for user {current_user.id}")
            except Exception as e:
                logger.warning(f"Failed to update CineBrain recommendation engine: {e}")
        
        if profile_analyzer:
            try:
                profile_analyzer.update_profile_realtime(
                    current_user.id,
                    {
                        'interaction_type': 'preference_update',
                        'metadata': {
                            'updated_languages': data.get('preferred_languages'),
                            'updated_genres': data.get('preferred_genres'),
                            'source': 'explicit_preference_update'
                        }
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update CineBrain profile analyzer: {e}")
        
        return jsonify({
            'success': True,
            'message': 'CineBrain preferences updated successfully',
            'user': {
                'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
                'preferred_genres': json.loads(current_user.preferred_genres or '[]')
            },
            'recommendation_refresh': 'triggered',
            'profile_analysis': 'updated'
        }), 200
        
    except Exception as e:
        logger.error(f"Update CineBrain preferences error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update CineBrain preferences'}), 500

@require_auth
def get_profile_analytics(current_user):
    try:
        if not profile_analyzer:
            return jsonify({'error': 'Profile analytics not available'}), 503
        
        profile = profile_analyzer.build_user_profile(current_user.id)
        
        analytics = {
            'user_segment': profile.get('user_segment', 'new_user'),
            'profile_strength': {
                'completeness': profile.get('profile_completeness', 0),
                'confidence': profile.get('confidence_score', 0),
                'diversity': profile.get('diversity_score', 0)
            },
            'cinematic_dna': profile.get('cinematic_dna', {}),
            'behavioral_insights': profile.get('behavioral_patterns', {}),
            'engagement_metrics': profile.get('engagement_metrics', {}),
            'recommendation_context': profile.get('recommendation_context', {}),
            'temporal_patterns': profile.get('temporal_patterns', {}),
            'preference_clusters': profile.get('preference_clusters', {})
        }
        
        return jsonify({
            'success': True,
            'analytics': analytics,
            'generated_at': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Profile analytics error: {e}")
        return jsonify({'error': 'Failed to get profile analytics'}), 500