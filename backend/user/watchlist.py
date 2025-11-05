# user/watchlist.py
from flask import request, jsonify
from datetime import datetime
import json
import logging
from .utils import require_auth, db, UserInteraction, Content, format_content_for_response, recommendation_engine

logger = logging.getLogger(__name__)

@require_auth
def get_watchlist(current_user):
    """Get user's watchlist"""
    try:
        watchlist_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='watchlist'
        ).order_by(UserInteraction.timestamp.desc()).all()
        
        content_ids = [interaction.content_id for interaction in watchlist_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        content_map = {content.id: content for content in contents}
        
        result = []
        for interaction in watchlist_interactions:
            content = content_map.get(interaction.content_id)
            if content:
                formatted_content = format_content_for_response(content, interaction)
                result.append(formatted_content)
        
        return jsonify({
            'watchlist': result,
            'total_count': len(result),
            'last_updated': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain watchlist error: {e}")
        return jsonify({'error': 'Failed to get CineBrain watchlist'}), 500

@require_auth
def add_to_watchlist(current_user):
    """Add content to watchlist"""
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        
        if not content_id:
            return jsonify({'error': 'Content ID required'}), 400
        
        # Check if already in watchlist
        existing = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='watchlist'
        ).first()
        
        if existing:
            return jsonify({
                'success': True,
                'message': 'Already in CineBrain watchlist'
            }), 200
        
        # Add to watchlist
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='watchlist'
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
                        'interaction_type': 'watchlist'
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update CineBrain recommendations: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Added to CineBrain watchlist'
        }), 201
        
    except Exception as e:
        logger.error(f"Add to CineBrain watchlist error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to add to CineBrain watchlist'}), 500

@require_auth
def remove_from_watchlist(current_user, content_id):
    """Remove content from watchlist"""
    try:
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='watchlist'
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
                            'interaction_type': 'remove_watchlist'
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to update CineBrain recommendations: {e}")
            
            return jsonify({
                'success': True,
                'message': 'Removed from CineBrain watchlist'
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Content not in CineBrain watchlist'
            }), 404
            
    except Exception as e:
        logger.error(f"Remove from CineBrain watchlist error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to remove from CineBrain watchlist'}), 500

@require_auth
def check_watchlist_status(current_user, content_id):
    """Check if content is in watchlist"""
    try:
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='watchlist'
        ).first()
        
        return jsonify({
            'in_watchlist': interaction is not None,
            'added_at': interaction.timestamp.isoformat() if interaction else None
        }), 200
        
    except Exception as e:
        logger.error(f"Check CineBrain watchlist status error: {e}")
        return jsonify({'error': 'Failed to check CineBrain watchlist status'}), 500