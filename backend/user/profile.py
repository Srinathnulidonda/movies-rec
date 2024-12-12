# user/profile.py
from flask import request, jsonify
from datetime import datetime
import json
import logging
from .utils import require_auth, db, User, UserInteraction, Content, get_enhanced_user_stats, recommendation_engine

logger = logging.getLogger(__name__)

@require_auth
def get_user_profile(current_user):
    """Get comprehensive user profile"""
    try:
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
        try:
            if recommendation_engine:
                rec_effectiveness = recommendation_engine.get_user_recommendation_metrics(current_user.id)
        except Exception as e:
            logger.warning(f"Could not get CineBrain recommendation effectiveness: {e}")
        
        profile_fields = {
            'preferred_languages': current_user.preferred_languages,
            'preferred_genres': current_user.preferred_genres,
            'location': current_user.location,
            'avatar_url': current_user.avatar_url
        }
        
        completed_fields = [field for field, value in profile_fields.items() if value]
        completion_score = min(100, len(completed_fields) * 25)
        missing_fields = [field for field, value in profile_fields.items() if not value]
        
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
            'profile_completion': {
                'score': completion_score,
                'missing_fields': missing_fields,
                'suggestions': [
                    'Add preferred languages to get better CineBrain recommendations',
                    'Select favorite genres to improve CineBrain content discovery',
                    'Add your location for regional CineBrain content suggestions',
                    'Upload an avatar to personalize your CineBrain profile'
                ][:len(missing_fields)]
            }
        }
        
        return jsonify(profile_data), 200
        
    except Exception as e:
        logger.error(f"CineBrain profile error: {e}")
        return jsonify({'error': 'Failed to get CineBrain user profile'}), 500

@require_auth
def update_user_profile(current_user):
    """Update user profile"""
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
        
        if recommendation_engine and updated_fields:
            try:
                recommendation_engine.update_user_preferences_realtime(
                    current_user.id,
                    {
                        'interaction_type': 'profile_update',
                        'metadata': {
                            'updated_fields': updated_fields,
                            'data': data
                        }
                    }
                )
                logger.info(f"Updated CineBrain recommendation engine for user {current_user.id}")
            except Exception as e:
                logger.warning(f"Failed to update CineBrain recommendation engine: {e}")
        
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
            }
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain profile update error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update CineBrain profile'}), 500

def get_public_profile(username):
    """Get public profile for a username"""
    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'user': {
                'id': user.id,
                'username': user.username,
                'avatar_url': user.avatar_url,
                'created_at': user.created_at.isoformat(),
                'last_active': user.last_active.isoformat() if user.last_active else None
            }
        }), 200
    except Exception as e:
        logger.error(f"Error getting public profile: {e}")
        return jsonify({'error': 'Failed to get profile'}), 500

@require_auth
def update_user_preferences(current_user):
    """Update user preferences for personalization"""
    try:
        data = request.get_json()
        
        if 'preferred_languages' in data:
            current_user.preferred_languages = json.dumps(data['preferred_languages'])
        
        if 'preferred_genres' in data:
            current_user.preferred_genres = json.dumps(data['preferred_genres'])
        
        db.session.commit()
        
        if recommendation_engine:
            try:
                recommendation_engine.update_user_preferences_realtime(
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
                logger.info(f"Successfully updated CineBrain preferences for user {current_user.id}")
            except Exception as e:
                logger.warning(f"Failed to update CineBrain recommendation engine: {e}")
        
        return jsonify({
            'success': True,
            'message': 'CineBrain preferences updated successfully',
            'user': {
                'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
                'preferred_genres': json.loads(current_user.preferred_genres or '[]')
            },
            'recommendation_refresh': 'triggered'
        }), 200
        
    except Exception as e:
        logger.error(f"Update CineBrain preferences error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update CineBrain preferences'}), 500