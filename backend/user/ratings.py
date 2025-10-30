# user/ratings.py
from flask import request, jsonify
from datetime import datetime
import json
import logging
import numpy as np
from .utils import require_auth, db, UserInteraction, Content

logger = logging.getLogger(__name__)

@require_auth
def get_user_ratings(current_user):
    """Get user's ratings"""
    try:
        rating_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='rating'
        ).filter(UserInteraction.rating.isnot(None)).order_by(
            UserInteraction.timestamp.desc()
        ).all()
        
        content_ids = [interaction.content_id for interaction in rating_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        content_map = {content.id: content for content in contents}
        
        result = []
        for interaction in rating_interactions:
            content = content_map.get(interaction.content_id)
            if content:
                result.append({
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'user_rating': interaction.rating,
                    'imdb_rating': content.rating,
                    'rated_at': interaction.timestamp.isoformat()
                })
        
        ratings = [interaction.rating for interaction in rating_interactions]
        stats = {
            'total_ratings': len(ratings),
            'average_rating': round(sum(ratings) / len(ratings), 1) if ratings else 0,
            'highest_rating': max(ratings) if ratings else 0,
            'lowest_rating': min(ratings) if ratings else 0
        }
        
        return jsonify({
            'ratings': result,
            'stats': stats,
            'last_updated': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain user ratings error: {e}")
        return jsonify({'error': 'Failed to get CineBrain user ratings'}), 500

@require_auth
def add_rating(current_user):
    """Add or update a rating"""
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        rating = data.get('rating')
        
        if not content_id or rating is None:
            return jsonify({'error': 'Content ID and rating required'}), 400
        
        if not (1 <= rating <= 10):
            return jsonify({'error': 'Rating must be between 1 and 10'}), 400
        
        # Check if rating already exists
        existing = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='rating'
        ).first()
        
        if existing:
            existing.rating = rating
            existing.timestamp = datetime.utcnow()
            message = 'Rating updated successfully'
        else:
            interaction = UserInteraction(
                user_id=current_user.id,
                content_id=content_id,
                interaction_type='rating',
                rating=rating
            )
            db.session.add(interaction)
            message = 'Rating added successfully'
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': message,
            'rating': rating
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain add rating error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to add rating'}), 500

@require_auth
def remove_rating(current_user, content_id):
    """Remove a rating"""
    try:
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='rating'
        ).first()
        
        if interaction:
            db.session.delete(interaction)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Rating removed successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'No rating found for this content'
            }), 404
            
    except Exception as e:
        logger.error(f"Remove rating error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to remove rating'}), 500

@require_auth
def get_rating_for_content(current_user, content_id):
    """Get user's rating for specific content"""
    try:
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='rating'
        ).first()
        
        return jsonify({
            'has_rating': interaction is not None,
            'rating': interaction.rating if interaction else None,
            'rated_at': interaction.timestamp.isoformat() if interaction else None
        }), 200
        
    except Exception as e:
        logger.error(f"Get rating error: {e}")
        return jsonify({'error': 'Failed to get rating'}), 500