# user/favorites.py
from flask import request, jsonify
from datetime import datetime
import json
import logging
from .utils import require_auth, db, UserInteraction, Content, format_content_for_response, recommendation_engine

logger = logging.getLogger(__name__)

def get_favorites(current_user):
    """Get user's favorites"""
    try:
        logger.info(f"CineBrain: Getting favorites for user {current_user.id} ({current_user.username})")
        
        favorite_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='favorite'
        ).order_by(UserInteraction.timestamp.desc()).all()
        
        logger.info(f"CineBrain: Found {len(favorite_interactions)} favorite interactions")
        
        content_ids = [interaction.content_id for interaction in favorite_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all() if content_ids else []
        content_map = {content.id: content for content in contents}
        
        logger.info(f"CineBrain: Retrieved {len(contents)} content items from database")
        
        result = []
        for interaction in favorite_interactions:
            content = content_map.get(interaction.content_id)
            if content:
                try:
                    formatted_content = format_content_for_response(content, interaction)
                    formatted_content['favorited_at'] = interaction.timestamp.isoformat()
                    formatted_content['user_rating'] = interaction.rating
                    result.append(formatted_content)
                except Exception as e:
                    logger.warning(f"CineBrain: Error formatting content {interaction.content_id}: {e}")
            else:
                logger.warning(f"CineBrain: Content {interaction.content_id} not found in database")
        
        logger.info(f"CineBrain: Successfully formatted {len(result)} favorites")
        
        return jsonify({
            'favorites': result,
            'total_count': len(result),
            'last_updated': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain favorites error for user {current_user.id}: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to get CineBrain favorites', 'details': str(e)}), 500

def add_to_favorites(current_user):
    """Add content to favorites"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        content_id = data.get('content_id')
        rating = data.get('rating')
        
        if not content_id:
            return jsonify({'error': 'Content ID required'}), 400
        
        # Ensure content_id is an integer
        try:
            content_id = int(content_id)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid content ID format'}), 400
        
        logger.info(f"CineBrain: Adding content {content_id} to favorites for user {current_user.id}")
        
        # Check if content exists
        content = Content.query.get(content_id)
        if not content:
            logger.warning(f"CineBrain: Content {content_id} not found in database")
            return jsonify({'error': 'Content not found'}), 404
        
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
                existing.timestamp = datetime.utcnow()
                db.session.commit()
                logger.info(f"CineBrain: Updated existing favorite with new rating {rating}")
            
            return jsonify({
                'success': True,
                'message': 'Already in CineBrain favorites',
                'was_updated': bool(rating)
            }), 200
        
        # Add to favorites
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='favorite',
            rating=rating,
            timestamp=datetime.utcnow()
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        logger.info(f"CineBrain: Successfully added content {content_id} to favorites for user {current_user.id}")
        
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
                logger.info(f"CineBrain: Updated recommendation engine for user {current_user.id}")
            except Exception as e:
                logger.warning(f"Failed to update CineBrain recommendations: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Added to CineBrain favorites',
            'interaction_id': interaction.id
        }), 201
        
    except Exception as e:
        logger.error(f"Add to CineBrain favorites error for user {current_user.id}: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to add to CineBrain favorites', 'details': str(e)}), 500

def remove_from_favorites(current_user, content_id):
    """Remove content from favorites"""
    try:
        # Ensure content_id is an integer
        try:
            content_id = int(content_id)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid content ID format'}), 400
        
        logger.info(f"CineBrain: Removing content {content_id} from favorites for user {current_user.id}")
        
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='favorite'
        ).first()
        
        if interaction:
            db.session.delete(interaction)
            db.session.commit()
            
            logger.info(f"CineBrain: Successfully removed content {content_id} from favorites for user {current_user.id}")
            
            if recommendation_engine:
                try:
                    recommendation_engine.update_user_preferences_realtime(
                        current_user.id,
                        {
                            'content_id': content_id,
                            'interaction_type': 'remove_favorite'
                        }
                    )
                    logger.info(f"CineBrain: Updated recommendation engine after removal for user {current_user.id}")
                except Exception as e:
                    logger.warning(f"Failed to update CineBrain recommendations: {e}")
            
            return jsonify({
                'success': True,
                'message': 'Removed from CineBrain favorites'
            }), 200
        else:
            logger.info(f"CineBrain: Content {content_id} was not in favorites for user {current_user.id}")
            return jsonify({
                'success': False,
                'message': 'Content not in CineBrain favorites'
            }), 404
            
    except Exception as e:
        logger.error(f"Remove from CineBrain favorites error for user {current_user.id}: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to remove from CineBrain favorites', 'details': str(e)}), 500

def check_favorite_status(current_user, content_id):
    """Check if content is in favorites"""
    try:
        # Ensure content_id is an integer
        try:
            content_id = int(content_id)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid content ID format'}), 400
        
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
        logger.error(f"Check CineBrain favorite status error for user {current_user.id}: {e}")
        return jsonify({'error': 'Failed to check CineBrain favorite status', 'details': str(e)}), 500