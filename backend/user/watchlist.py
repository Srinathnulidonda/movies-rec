# backend/user/watchlist.py

from flask import request, jsonify
from datetime import datetime
import json
import logging
from .utils import require_auth, db, UserInteraction, Content, format_content_for_response, recommendation_engine, profile_analyzer, cache_get, cache_set, cache_delete, get_cache_key, get_content_by_ids

logger = logging.getLogger(__name__)

@require_auth
def get_watchlist(current_user):
    try:
        cache_key = get_cache_key('user_watchlist', current_user.id)
        cached_watchlist = cache_get(cache_key)
        
        if cached_watchlist:
            return jsonify(cached_watchlist), 200
        
        watchlist_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='watchlist'
        ).order_by(UserInteraction.timestamp.desc()).all()
        
        content_ids = [interaction.content_id for interaction in watchlist_interactions]
        contents = get_content_by_ids(content_ids)
        content_map = {content.id: content for content in contents}
        
        result = []
        smart_suggestions = []
        
        for interaction in watchlist_interactions:
            content = content_map.get(interaction.content_id)
            if content:
                formatted_content = format_content_for_response(content, interaction)
                result.append(formatted_content)
        
        if recommendation_engine and len(result) > 0:
            try:
                rec_data = recommendation_engine.generate_recommendations(
                    user_id=current_user.id,
                    limit=5,
                    context={'source': 'watchlist_suggestions'}
                )
                smart_suggestions = rec_data.get('recommendations', [])[:3]
            except Exception as e:
                logger.warning(f"Failed to get watchlist suggestions: {e}")
        
        watchlist_analytics = {}
        if profile_analyzer:
            try:
                profile = profile_analyzer.build_user_profile(current_user.id)
                watchlist_analytics = {
                    'total_items': len(result),
                    'avg_rating': sum([c.get('rating', 0) for c in result]) / len(result) if result else 0,
                    'genres_distribution': profile.get('implicit_preferences', {}).get('genre_preferences', {}),
                    'completion_rate': profile.get('engagement_metrics', {}).get('engagement_score', 0)
                }
            except Exception as e:
                logger.warning(f"Failed to get watchlist analytics: {e}")
        
        response_data = {
            'watchlist': result,
            'total_count': len(result),
            'smart_suggestions': smart_suggestions,
            'analytics': watchlist_analytics,
            'last_updated': datetime.utcnow().isoformat()
        }
        
        cache_set(cache_key, response_data, timeout=300)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"CineBrain watchlist error: {e}")
        return jsonify({'error': 'Failed to get CineBrain watchlist'}), 500

@require_auth
def add_to_watchlist(current_user):
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        
        if not content_id:
            return jsonify({'error': 'Content ID required'}), 400
        
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
        
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='watchlist',
            interaction_metadata=json.dumps(data.get('metadata', {}))
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        cache_key = get_cache_key('user_watchlist', current_user.id)
        cache_delete(cache_key)
        
        if recommendation_engine:
            try:
                recommendation_engine.update_user_feedback(
                    current_user.id,
                    content_id,
                    'watchlist'
                )
            except Exception as e:
                logger.warning(f"Failed to update CineBrain recommendations: {e}")
        
        if profile_analyzer:
            try:
                profile_analyzer.update_profile_realtime(
                    current_user.id,
                    {
                        'content_id': content_id,
                        'interaction_type': 'watchlist'
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update CineBrain profile: {e}")
        
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
    try:
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='watchlist'
        ).first()
        
        if interaction:
            db.session.delete(interaction)
            db.session.commit()
            
            cache_key = get_cache_key('user_watchlist', current_user.id)
            cache_delete(cache_key)
            
            if recommendation_engine:
                try:
                    recommendation_engine.update_user_feedback(
                        current_user.id,
                        content_id,
                        'remove_watchlist'
                    )
                except Exception as e:
                    logger.warning(f"Failed to update CineBrain recommendations: {e}")
            
            if profile_analyzer:
                try:
                    profile_analyzer.update_profile_realtime(
                        current_user.id,
                        {
                            'content_id': content_id,
                            'interaction_type': 'remove_watchlist'
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to update CineBrain profile: {e}")
            
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

@require_auth
def get_watchlist_recommendations(current_user):
    try:
        if not recommendation_engine:
            return jsonify({'error': 'Recommendation engine not available'}), 503
        
        watchlist_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='watchlist'
        ).all()
        
        if not watchlist_interactions:
            return jsonify({
                'recommendations': [],
                'message': 'Add items to your watchlist to get personalized recommendations'
            }), 200
        
        recommendations = recommendation_engine.generate_recommendations(
            user_id=current_user.id,
            limit=10,
            context={'source': 'watchlist_based', 'strategy': 'similar_to_watchlist'}
        )
        
        return jsonify({
            'success': True,
            'recommendations': recommendations.get('recommendations', []),
            'metadata': recommendations.get('metadata', {}),
            'based_on_watchlist': True
        }), 200
        
    except Exception as e:
        logger.error(f"Watchlist recommendations error: {e}")
        return jsonify({'error': 'Failed to get watchlist recommendations'}), 500

@require_auth
def organize_watchlist(current_user):
    try:
        data = request.get_json()
        organization_type = data.get('type', 'genre')
        
        watchlist_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='watchlist'
        ).order_by(UserInteraction.timestamp.desc()).all()
        
        content_ids = [interaction.content_id for interaction in watchlist_interactions]
        contents = get_content_by_ids(content_ids)
        
        organized = {}
        
        if organization_type == 'genre':
            for content in contents:
                genres = json.loads(content.genres or '[]')
                for genre in genres:
                    if genre not in organized:
                        organized[genre] = []
                    organized[genre].append(format_content_for_response(content))
        
        elif organization_type == 'rating':
            for content in contents:
                rating_range = 'Low (< 6)' if content.rating < 6 else 'Medium (6-7.5)' if content.rating < 7.5 else 'High (7.5+)'
                if rating_range not in organized:
                    organized[rating_range] = []
                organized[rating_range].append(format_content_for_response(content))
        
        elif organization_type == 'content_type':
            for content in contents:
                content_type = content.content_type.title()
                if content_type not in organized:
                    organized[content_type] = []
                organized[content_type].append(format_content_for_response(content))
        
        elif organization_type == 'date_added':
            for interaction in watchlist_interactions:
                content = next((c for c in contents if c.id == interaction.content_id), None)
                if content:
                    date_key = interaction.timestamp.strftime('%Y-%m')
                    if date_key not in organized:
                        organized[date_key] = []
                    organized[date_key].append(format_content_for_response(content, interaction))
        
        return jsonify({
            'success': True,
            'organized_watchlist': organized,
            'organization_type': organization_type,
            'total_categories': len(organized)
        }), 200
        
    except Exception as e:
        logger.error(f"Organize watchlist error: {e}")
        return jsonify({'error': 'Failed to organize watchlist'}), 500
