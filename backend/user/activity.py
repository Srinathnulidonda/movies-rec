# user/activity.py
from flask import request, jsonify
from datetime import datetime
import json
import logging
from .utils import (
    require_auth, db, UserInteraction, Content, create_minimal_content_record, 
    content_service, recommendation_engine, profile_analyzer, personalized_recommendation_engine
)

logger = logging.getLogger(__name__)

@require_auth
def record_interaction(current_user):
    """Record user interaction with content"""
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'interaction_type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields for CineBrain interaction'}), 400
        
        content_id = data['content_id']
        
        content_exists = Content.query.filter_by(id=content_id).first()
        if not content_exists:
            logger.warning(f"CineBrain: Content {content_id} not found in database, attempting to create")
            
            try:
                content_metadata = data.get('metadata', {})
                content_info = content_metadata.get('content_info')
                
                if content_info:
                    if content_service and content_info.get('tmdb_id'):
                        try:
                            from app import CineBrainTMDBService
                            tmdb_data = CineBrainTMDBService.get_content_details(
                                content_info['tmdb_id'], 
                                content_info.get('content_type', 'movie').strip()
                            )
                            if tmdb_data:
                                content_exists = content_service.save_content_from_tmdb(
                                    tmdb_data, 
                                    content_info.get('content_type', 'movie').strip()
                                )
                                logger.info(f"CineBrain: Created content from TMDB for ID {content_id}")
                        except Exception as e:
                            logger.warning(f"Failed to fetch from TMDB: {e}")
                    
                    if not content_exists:
                        content_exists = create_minimal_content_record(content_id, content_info)
                
                if not content_exists:
                    return jsonify({
                        'error': 'Content not found in CineBrain database',
                        'details': 'Unable to create or fetch content record. Please try again.'
                    }), 404
                    
            except Exception as e:
                logger.error(f"Failed to create content record: {e}")
                return jsonify({
                    'error': 'Content not found in CineBrain database',
                    'details': 'Unable to create content record due to data validation error'
                }), 404
        
        # Handle removal interactions
        if data['interaction_type'] in ['remove_watchlist', 'remove_favorite']:
            interaction_type = 'watchlist' if data['interaction_type'] == 'remove_watchlist' else 'favorite'
            interaction = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=data['content_id'],
                interaction_type=interaction_type
            ).first()
            
            if interaction:
                db.session.delete(interaction)
                db.session.commit()
                
                # Update all recommendation systems for removal
                update_results = []
                
                # Update new personalized system
                if profile_analyzer:
                    try:
                        success = profile_analyzer.update_profile_realtime(
                            current_user.id,
                            {
                                'content_id': data['content_id'],
                                'interaction_type': data['interaction_type'],
                                'metadata': data.get('metadata', {})
                            }
                        )
                        if success:
                            update_results.append('advanced_profile_updated')
                    except Exception as e:
                        logger.warning(f"Failed to update CineBrain advanced profile for removal: {e}")
                
                # Update legacy system
                if recommendation_engine and recommendation_engine != personalized_recommendation_engine:
                    try:
                        recommendation_engine.update_user_preferences_realtime(
                            current_user.id,
                            {
                                'content_id': data['content_id'],
                                'interaction_type': data['interaction_type'],
                                'metadata': data.get('metadata', {})
                            }
                        )
                        update_results.append('legacy_profile_updated')
                    except Exception as e:
                        logger.warning(f"Failed to update CineBrain legacy profile for removal: {e}")
                
                message = f'Removed from CineBrain {"watchlist" if interaction_type == "watchlist" else "favorites"}'
                return jsonify({
                    'success': True,
                    'message': message,
                    'real_time_updates': update_results
                }), 200
            else:
                item_type = "watchlist" if interaction_type == "watchlist" else "favorites"
                return jsonify({
                    'success': False,
                    'message': f'Content not in CineBrain {item_type}'
                }), 404
        
        # Check for existing interaction
        if data['interaction_type'] in ['watchlist', 'favorite']:
            existing = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=data['content_id'],
                interaction_type=data['interaction_type']
            ).first()
            
            if existing:
                item_type = "watchlist" if data['interaction_type'] == "watchlist" else "favorites"
                return jsonify({
                    'success': True,
                    'message': f'Already in CineBrain {item_type}'
                }), 200
        
        # Create new interaction
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=data['content_id'],
            interaction_type=data['interaction_type'],
            rating=data.get('rating'),
            interaction_metadata=json.dumps(data.get('metadata', {}))
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        # Update all recommendation systems
        update_results = []
        
        # Update new personalized system
        if profile_analyzer:
            try:
                success = profile_analyzer.update_profile_realtime(
                    current_user.id,
                    {
                        'content_id': data['content_id'],
                        'interaction_type': data['interaction_type'],
                        'rating': data.get('rating'),
                        'metadata': data.get('metadata', {})
                    }
                )
                if success:
                    update_results.append('advanced_profile_updated')
                    logger.info(f"Successfully updated CineBrain advanced profile for user {current_user.id}")
            except Exception as e:
                logger.warning(f"Failed to update CineBrain advanced profile: {e}")
        
        # Update legacy system
        if recommendation_engine and recommendation_engine != personalized_recommendation_engine:
            try:
                recommendation_engine.update_user_preferences_realtime(
                    current_user.id,
                    {
                        'content_id': data['content_id'],
                        'interaction_type': data['interaction_type'],
                        'rating': data.get('rating'),
                        'metadata': data.get('metadata', {})
                    }
                )
                update_results.append('legacy_profile_updated')
                logger.info(f"Successfully updated CineBrain legacy profile for user {current_user.id}")
            except Exception as e:
                logger.warning(f"Failed to update CineBrain legacy profile: {e}")
        
        return jsonify({
            'success': True,
            'message': 'CineBrain interaction recorded successfully',
            'interaction_id': interaction.id,
            'real_time_updates': update_results,
            'advanced_features_active': bool(profile_analyzer)
        }), 201
        
    except Exception as e:
        logger.error(f"CineBrain interaction recording error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record CineBrain interaction'}), 500

def get_public_activity(username):
    """Get public activity for a username"""
    try:
        from .utils import User
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Return limited public activity
        public_interactions = UserInteraction.query.filter_by(
            user_id=user.id,
            interaction_type='rating'  # Only show ratings publicly
        ).order_by(UserInteraction.timestamp.desc()).limit(10).all()
        
        formatted_activity = []
        for interaction in public_interactions:
            content = Content.query.get(interaction.content_id)
            if content:
                formatted_activity.append({
                    'interaction_type': interaction.interaction_type,
                    'rating': interaction.rating,
                    'timestamp': interaction.timestamp.isoformat(),
                    'content': {
                        'id': content.id,
                        'title': content.title,
                        'content_type': content.content_type,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path
                    }
                })
        
        return jsonify({'recent_activity': formatted_activity}), 200
    except Exception as e:
        logger.error(f"Error getting public activity: {e}")
        return jsonify({'error': 'Failed to get activity'}), 500

def get_public_stats(username):
    """Get public stats for a username"""
    try:
        from .utils import User
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Return limited public stats
        interactions = UserInteraction.query.filter_by(user_id=user.id).all()
        
        public_stats = {
            'total_interactions': len(interactions),
            'favorites': len([i for i in interactions if i.interaction_type == 'favorite']),
            'ratings_given': len([i for i in interactions if i.interaction_type == 'rating'])
        }
        
        return jsonify({'stats': public_stats}), 200
    except Exception as e:
        logger.error(f"Error getting public stats: {e}")
        return jsonify({'error': 'Failed to get stats'}), 500