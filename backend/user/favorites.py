# user/favorites.py
from flask import request, jsonify
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

def get_actual_content_id(content_id):
    """Get the actual database content ID, handling TMDB ID mapping"""
    try:
        from .utils import Content
        
        # First try the ID as-is
        content = Content.query.filter_by(id=content_id).first()
        if content:
            return content_id
        
        # If not found, try as TMDB ID
        content_by_tmdb = Content.query.filter_by(tmdb_id=content_id).first()
        if content_by_tmdb:
            logger.info(f"CineBrain: Mapped TMDB ID {content_id} to database ID {content_by_tmdb.id}")
            return content_by_tmdb.id
        
        return None
    except Exception as e:
        logger.error(f"Error mapping content ID {content_id}: {e}")
        return None

def create_content_from_tmdb_id(tmdb_id, request_data):
    """Create content from TMDB ID when frontend sends tmdb_movie_xxx format"""
    try:
        from app import CineBrainTMDBService
        from .utils import content_service
        
        # Determine content type from the original ID format
        original_id = request_data.get('content_id', '')
        if 'movie' in original_id:
            content_type = 'movie'
        elif 'tv' in original_id:
            content_type = 'tv'
        else:
            content_type = 'movie'  # Default
            
        # Fetch from TMDB
        tmdb_data = CineBrainTMDBService.get_content_details(int(tmdb_id), content_type)
        
        if tmdb_data and content_service:
            new_content = content_service.save_content_from_tmdb(tmdb_data, content_type)
            if new_content:
                return new_content.id
                
        return None
        
    except Exception as e:
        logger.error(f"Error creating content from TMDB ID {tmdb_id}: {e}")
        return None

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
        from .utils import db, UserInteraction, Content, recommendation_engine, content_service, create_minimal_content_record
        
        data = request.get_json()
        raw_content_id = data.get('content_id')
        rating = data.get('rating')
        
        if not raw_content_id:
            return jsonify({'error': 'Content ID required'}), 400
        
        # FIX: Validate and convert content_id to integer
        try:
            if isinstance(raw_content_id, str):
                if raw_content_id.startswith(('tmdb_', 'mal_', 'imdb_')):
                    if 'tmdb_movie_' in raw_content_id or 'tmdb_tv_' in raw_content_id:
                        tmdb_id = int(raw_content_id.split('_')[-1])
                        # Try to find existing content by TMDB ID
                        existing_content = Content.query.filter_by(tmdb_id=tmdb_id).first()
                        if existing_content:
                            content_id = existing_content.id
                            logger.info(f"Found existing content by TMDB ID: {tmdb_id} -> ID: {content_id}")
                        else:
                            # Will create new content below
                            content_id = None
                            logger.info(f"Will create new content for TMDB ID: {tmdb_id}")
                    else:
                        return jsonify({'error': 'Invalid content ID format'}), 400
                else:
                    content_id = int(raw_content_id)
            else:
                content_id = int(raw_content_id)
                
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid content ID format: {raw_content_id}, error: {e}")
            return jsonify({'error': 'Invalid content ID format'}), 400
        
        # Check if content exists (if we have an ID)
        content_exists = None
        if content_id:
            content_exists = Content.query.filter_by(id=content_id).first()
        
        if not content_exists:
            logger.warning(f"Content {content_id or raw_content_id} not found, attempting to create from request data")
            
            content_metadata = data.get('metadata', {})
            content_info = content_metadata.get('content_info')
            
            if content_info:
                # Try to fetch from TMDB if we have tmdb_id
                if content_service and content_info.get('tmdb_id'):
                    try:
                        from app import CineBrainTMDBService
                        content_type = content_info.get('content_type', 'movie')
                        tmdb_data = CineBrainTMDBService.get_content_details(
                            content_info['tmdb_id'], 
                            content_type
                        )
                        if tmdb_data:
                            content_exists = content_service.save_content_from_tmdb(tmdb_data, content_type)
                            if content_exists:
                                content_id = content_exists.id  # FIX: Use the actual generated ID
                                logger.info(f"Created content from TMDB with new ID {content_id}")
                    except Exception as e:
                        logger.warning(f"Failed to fetch from TMDB: {e}")
                
                # If TMDB fetch failed, create minimal record
                if not content_exists:
                    # FIX: Don't force the original ID, let it auto-generate
                    content_exists = create_minimal_content_record(None, content_info)
                    if content_exists:
                        content_id = content_exists.id  # FIX: Use the actual generated ID
                        logger.info(f"Created minimal content record with new ID {content_id}")
            
            if not content_exists:
                logger.error(f"Failed to create content record")
                return jsonify({
                    'error': 'Content not available',
                    'message': 'Unable to add this content to favorites at this time'
                }), 400
        
        # Check if already in favorites (with actual content ID)
        existing = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,  # FIX: Use actual content_id
            interaction_type='favorite'
        ).first()
        
        if existing:
            if rating:
                existing.rating = rating
                db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Already in CineBrain favorites',
                'content_id': content_id,  # FIX: Return actual content_id
                'actual_content_id': content_id
            }), 200
        
        # Add to favorites
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=content_id,  # FIX: Use actual content_id
            interaction_type='favorite',
            rating=rating
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        # Update recommendation engine with actual content ID
        if recommendation_engine:
            try:
                if hasattr(recommendation_engine, 'update_user_preferences_realtime'):
                    recommendation_engine.update_user_preferences_realtime(
                        current_user.id,
                        {
                            'content_id': content_id,  # FIX: Use actual content_id
                            'interaction_type': 'favorite',
                            'rating': rating
                        }
                    )
                elif hasattr(recommendation_engine, 'update_user_profile'):
                    recommendation_engine.update_user_profile(
                        current_user.id,
                        {
                            'content_id': content_id,  # FIX: Use actual content_id
                            'interaction_type': 'favorite',
                            'rating': rating
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to update CineBrain recommendations: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Added to CineBrain favorites',
            'content_id': content_id,  # FIX: Return actual content_id
            'actual_content_id': content_id
        }), 201
        
    except Exception as e:
        logger.error(f"Add to CineBrain favorites error: {e}")
        from .utils import db
        if db:
            db.session.rollback()
        return jsonify({'error': 'Failed to add to CineBrain favorites'}), 500

def remove_from_favorites(current_user, content_id):
    """Remove content from favorites"""
    try:
        from .utils import db, UserInteraction, Content, recommendation_engine
        
        # FIX: Use helper function for ID mapping
        try:
            if isinstance(content_id, str):
                # Handle string IDs that might contain non-numeric characters
                if content_id.startswith(('tmdb_', 'mal_', 'imdb_')):
                    return jsonify({
                        'success': False,
                        'error': 'Invalid content ID format for removal',
                        'message': 'Cannot remove content with external ID format'
                    }), 400
                content_id = int(content_id)
            else:
                content_id = int(content_id)
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'Invalid content ID format',
                'message': 'Content ID must be a valid integer'
            }), 400
        
        # FIX: Use helper function to get actual content ID
        actual_content_id = get_actual_content_id(content_id)
        
        if not actual_content_id:
            logger.warning(f"CineBrain: No content found for ID {content_id}")
            return jsonify({
                'success': False,
                'message': 'Content not found in CineBrain database',
                'details': f'No content record found for ID {content_id}'
            }), 404
        
        # Try to find the favorite interaction
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=actual_content_id,
            interaction_type='favorite'
        ).first()
        
        if interaction:
            db.session.delete(interaction)
            db.session.commit()
            
            if recommendation_engine:
                try:
                    if hasattr(recommendation_engine, 'update_user_preferences_realtime'):
                        recommendation_engine.update_user_preferences_realtime(
                            current_user.id,
                            {
                                'content_id': actual_content_id,  # Use mapped ID
                                'interaction_type': 'remove_favorite'
                            }
                        )
                    elif hasattr(recommendation_engine, 'update_user_profile'):
                        recommendation_engine.update_user_profile(
                            current_user.id,
                            {
                                'content_id': actual_content_id,  # Use mapped ID
                                'interaction_type': 'remove_favorite'
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to update CineBrain recommendations: {e}")
            
            return jsonify({
                'success': True,
                'message': 'Removed from CineBrain favorites',
                'actual_content_id': actual_content_id  # Return the actual ID used
            }), 200
        else:
            # Better error response for missing content
            logger.warning(f"CineBrain: No favorite found for content ID {content_id} (mapped to {actual_content_id})")
            return jsonify({
                'success': False,
                'message': 'Content not in CineBrain favorites',
                'details': f'No favorite record found for content ID {content_id}',
                'searched_id': actual_content_id
            }), 404
            
    except Exception as e:
        logger.error(f"Remove from CineBrain favorites error: {e}")
        from .utils import db
        if db:
            db.session.rollback()
        return jsonify({'error': 'Failed to remove from CineBrain favorites'}), 500

def check_favorite_status(current_user, content_id):
    """Check if content is in favorites"""
    try:
        from .utils import UserInteraction
        
        # FIX: Use helper function for ID mapping
        actual_content_id = get_actual_content_id(content_id)
        
        if not actual_content_id:
            return jsonify({
                'in_favorites': False,
                'favorited_at': None,
                'user_rating': None,
                'error': 'Content not found'
            }), 200
        
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=actual_content_id,
            interaction_type='favorite'
        ).first()
        
        return jsonify({
            'in_favorites': interaction is not None,
            'favorited_at': interaction.timestamp.isoformat() if interaction else None,
            'user_rating': interaction.rating if interaction else None,
            'actual_content_id': actual_content_id
        }), 200
        
    except Exception as e:
        logger.error(f"Check CineBrain favorite status error: {e}")
        return jsonify({'error': 'Failed to check CineBrain favorite status'}), 500