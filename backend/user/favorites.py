# user/favorites.py
from flask import request, jsonify
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

def get_favorites(current_user):
    """Get user's favorites"""
    try:
        # Import here to ensure we get the initialized versions
        from .utils import db, UserInteraction, Content, format_content_for_response
        
        favorite_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='favorite'
        ).order_by(UserInteraction.timestamp.desc()).all()
        
        content_ids = [interaction.content_id for interaction in favorite_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        content_map = {content.id: content for content in contents}
        
        result = []
        for interaction in favorite_interactions:
            content = content_map.get(interaction.content_id)
            if content:
                formatted_content = format_content_for_response(content, interaction)
                formatted_content['favorited_at'] = interaction.timestamp.isoformat()
                formatted_content['user_rating'] = interaction.rating
                result.append(formatted_content)
        
        return jsonify({
            'favorites': result,
            'total_count': len(result),
            'last_updated': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain favorites error: {e}")
        return jsonify({'error': 'Failed to get CineBrain favorites'}), 500

def add_to_favorites(current_user):
    """Add content to favorites"""
    try:
        # Import here to ensure we get the initialized versions
        from .utils import db, UserInteraction, Content, recommendation_engine
        
        data = request.get_json()
        content_id = data.get('content_id')
        rating = data.get('rating')
        
        if not content_id:
            return jsonify({'error': 'Content ID required'}), 400
        
        # Check if content exists
        content_exists = Content.query.filter_by(id=content_id).first()
        if not content_exists:
            return jsonify({'error': 'Content not found in database'}), 404
        
        # Check if already in favorites
        existing = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='favorite'
        ).first()
        
        if existing:
            # Update rating if provided
            if rating:
                existing.rating = rating
                db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Already in CineBrain favorites'
            }), 200
        
        # Add to favorites
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='favorite',
            rating=rating
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        # Update recommendation engine
        if recommendation_engine:
            try:
                recommendation_engine.update_user_preferences_realtime(
                    current_user.id,
                    {
                        'content_id': content_id,
                        'interaction_type': 'favorite',
                        'rating': rating
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update CineBrain recommendations: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Added to CineBrain favorites'
        }), 201
        
    except Exception as e:
        logger.error(f"Add to CineBrain favorites error: {e}")
        # Import db here for rollback
        from .utils import db
        if db:
            db.session.rollback()
        return jsonify({'error': 'Failed to add to CineBrain favorites'}), 500

def remove_from_favorites(current_user, content_id):
    """Remove content from favorites"""
    try:
        # Import here to ensure we get the initialized versions
        from .utils import db, UserInteraction, recommendation_engine
        
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='favorite'
        ).first()
        
        if interaction:
            db.session.delete(interaction)
            db.session.commit()
            
            if recommendation_engine:
                try:
                    recommendation_engine.update_user_preferences_realtime(
                        current_user.id,
                        {
                            'content_id': content_id,
                            'interaction_type': 'remove_favorite'
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to update CineBrain recommendations: {e}")
            
            return jsonify({
                'success': True,
                'message': 'Removed from CineBrain favorites'
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Content not in CineBrain favorites'
            }), 404
            
    except Exception as e:
        logger.error(f"Remove from CineBrain favorites error: {e}")
        # Import db here for rollback
        from .utils import db
        if db:
            db.session.rollback()
        return jsonify({'error': 'Failed to remove from CineBrain favorites'}), 500

def check_favorite_status(current_user, content_id):
    """Check if content is in favorites"""
    try:
        # Import here to ensure we get the initialized versions
        from .utils import UserInteraction
        
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='favorite'
        ).first()
        
        return jsonify({
            'in_favorites': interaction is not None,
            'favorited_at': interaction.timestamp.isoformat() if interaction else None,
            'user_rating': interaction.rating if interaction else None
        }), 200
        
    except Exception as e:
        logger.error(f"Check CineBrain favorite status error: {e}")
        return jsonify({'error': 'Failed to check CineBrain favorite status'}), 500