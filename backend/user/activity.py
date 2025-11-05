# backend/user/activity.py

from flask import request, jsonify
from datetime import datetime, timedelta
import json
import logging
from .utils import require_auth, db, UserInteraction, Content, create_minimal_content_record, content_service, recommendation_engine, profile_analyzer, cache_get, cache_set, get_cache_key, get_content_by_ids
from collections import Counter, defaultdict
import numpy as np

logger = logging.getLogger(__name__)

@require_auth
def record_interaction(current_user):
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
                            # Import from app correctly
                            from flask import current_app
                            
                            # Get TMDB service from services dict if available
                            tmdb_service = None
                            if hasattr(current_app, 'config') and 'TMDBService' in current_app.config:
                                tmdb_service = current_app.config['TMDBService']
                            
                            if not tmdb_service:
                                # Try to import from app module
                                try:
                                    import app
                                    tmdb_service = app.CineBrainTMDBService
                                except ImportError:
                                    logger.warning("Could not import TMDB service")
                            
                            if tmdb_service:
                                tmdb_data = tmdb_service.get_content_details(
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
                
                if recommendation_engine:
                    try:
                        recommendation_engine.update_user_feedback(
                            current_user.id,
                            data['content_id'],
                            data['interaction_type'],
                            data.get('metadata', {})
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update CineBrain real-time recommendations: {e}")
                
                if profile_analyzer:
                    try:
                        profile_analyzer.update_profile_realtime(
                            current_user.id,
                            {
                                'content_id': data['content_id'],
                                'interaction_type': data['interaction_type'],
                                'metadata': data.get('metadata', {})
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update CineBrain real-time profile: {e}")
                
                cache_key = get_cache_key('user_activity', current_user.id)
                cache_set(cache_key, None)
                
                message = f'Removed from CineBrain {"watchlist" if interaction_type == "watchlist" else "favorites"}'
                return jsonify({
                    'success': True,
                    'message': message
                }), 200
            else:
                item_type = "watchlist" if interaction_type == "watchlist" else "favorites"
                return jsonify({
                    'success': False,
                    'message': f'Content not in CineBrain {item_type}'
                }), 404
        
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
        
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=data['content_id'],
            interaction_type=data['interaction_type'],
            rating=data.get('rating'),
            interaction_metadata=json.dumps(data.get('metadata', {}))
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        cache_key = get_cache_key('user_activity', current_user.id)
        cache_set(cache_key, None)
        
        if recommendation_engine:
            try:
                recommendation_engine.update_user_feedback(
                    current_user.id,
                    data['content_id'],
                    data['interaction_type'],
                    data.get('rating')
                )
            except Exception as e:
                logger.warning(f"Failed to update CineBrain real-time recommendations: {e}")
        
        if profile_analyzer:
            try:
                profile_analyzer.update_profile_realtime(
                    current_user.id,
                    {
                        'content_id': data['content_id'],
                        'interaction_type': data['interaction_type'],
                        'rating': data.get('rating'),
                        'metadata': data.get('metadata', {})
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update CineBrain real-time profile: {e}")
        
        return jsonify({
            'success': True,
            'message': 'CineBrain interaction recorded successfully',
            'interaction_id': interaction.id
        }), 201
        
    except Exception as e:
        logger.error(f"CineBrain interaction recording error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record CineBrain interaction'}), 500

@require_auth
def get_user_activity(current_user):
    try:
        cache_key = get_cache_key('user_activity', current_user.id)
        cached_activity = cache_get(cache_key)
        
        if cached_activity:
            return jsonify(cached_activity), 200
        
        limit = int(request.args.get('limit', 50))
        interaction_type = request.args.get('type')
        days = int(request.args.get('days', 30))
        
        query = UserInteraction.query.filter(
            UserInteraction.user_id == current_user.id,
            UserInteraction.timestamp >= datetime.utcnow() - timedelta(days=days)
        )
        
        if interaction_type:
            query = query.filter(UserInteraction.interaction_type == interaction_type)
        
        interactions = query.order_by(UserInteraction.timestamp.desc()).limit(limit).all()
        
        content_ids = [interaction.content_id for interaction in interactions]
        contents = get_content_by_ids(content_ids)
        content_map = {content.id: content for content in contents}
        
        activity_data = []
        for interaction in interactions:
            content = content_map.get(interaction.content_id)
            if content:
                activity_data.append({
                    'id': interaction.id,
                    'interaction_type': interaction.interaction_type,
                    'timestamp': interaction.timestamp.isoformat(),
                    'rating': interaction.rating,
                    'metadata': json.loads(interaction.interaction_metadata or '{}'),
                    'content': {
                        'id': content.id,
                        'title': content.title,
                        'content_type': content.content_type,
                        'genres': json.loads(content.genres or '[]'),
                        'rating': content.rating,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                        'slug': content.slug
                    }
                })
        
        activity_stats = analyze_activity_patterns(interactions, contents)
        
        response_data = {
            'activity': activity_data,
            'stats': activity_stats,
            'total_interactions': len(activity_data),
            'period_days': days,
            'generated_at': datetime.utcnow().isoformat()
        }
        
        cache_set(cache_key, response_data, timeout=300)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"CineBrain activity error: {e}")
        return jsonify({'error': 'Failed to get CineBrain activity'}), 500

def analyze_activity_patterns(interactions, contents):
    try:
        if not interactions:
            return {}
        
        interaction_counts = Counter(i.interaction_type for i in interactions)
        
        hourly_activity = defaultdict(int)
        daily_activity = defaultdict(int)
        
        for interaction in interactions:
            hour = interaction.timestamp.hour
            day = interaction.timestamp.strftime('%A')
            hourly_activity[hour] += 1
            daily_activity[day] += 1
        
        peak_hour = max(hourly_activity.items(), key=lambda x: x[1])[0] if hourly_activity else 0
        peak_day = max(daily_activity.items(), key=lambda x: x[1])[0] if daily_activity else 'Monday'
        
        content_map = {c.id: c for c in contents}
        genre_preferences = defaultdict(int)
        language_preferences = defaultdict(int)
        
        for interaction in interactions:
            content = content_map.get(interaction.content_id)
            if content:
                if content.genres:
                    for genre in json.loads(content.genres or '[]'):
                        genre_preferences[genre] += 1
                
                if content.languages:
                    for language in json.loads(content.languages or '[]'):
                        language_preferences[language] += 1
        
        recent_streak = calculate_activity_streak(interactions)
        
        return {
            'interaction_breakdown': dict(interaction_counts),
            'peak_activity_hour': peak_hour,
            'peak_activity_day': peak_day,
            'top_genres': dict(sorted(genre_preferences.items(), key=lambda x: x[1], reverse=True)[:5]),
            'top_languages': dict(sorted(language_preferences.items(), key=lambda x: x[1], reverse=True)[:3]),
            'activity_streak_days': recent_streak,
            'hourly_distribution': dict(hourly_activity),
            'daily_distribution': dict(daily_activity)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing activity patterns: {e}")
        return {}

def calculate_activity_streak(interactions):
    try:
        if not interactions:
            return 0
        
        dates = sorted(set(i.timestamp.date() for i in interactions), reverse=True)
        
        if not dates:
            return 0
        
        streak = 0
        today = datetime.utcnow().date()
        
        for i, date in enumerate(dates):
            expected_date = today - timedelta(days=i)
            if date == expected_date:
                streak += 1
            else:
                break
        
        return streak
        
    except Exception as e:
        logger.error(f"Error calculating activity streak: {e}")
        return 0

def get_public_activity(username):
    try:
        from .utils import User
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        public_interactions = UserInteraction.query.filter_by(
            user_id=user.id,
            interaction_type='rating'
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
    try:
        from .utils import User
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        interactions = UserInteraction.query.filter_by(user_id=user.id).all()
        
        public_stats = {
            'total_interactions': len(interactions),
            'favorites': len([i for i in interactions if i.interaction_type == 'favorite']),
            'ratings_given': len([i for i in interactions if i.interaction_type == 'rating']),
            'member_since': user.created_at.isoformat(),
            'last_active': user.last_active.isoformat() if user.last_active else None
        }
        
        return jsonify({'stats': public_stats}), 200
    except Exception as e:
        logger.error(f"Error getting public stats: {e}")
        return jsonify({'error': 'Failed to get stats'}), 500

@require_auth
def get_activity_insights(current_user):
    try:
        if not profile_analyzer:
            return jsonify({'error': 'Profile analyzer not available'}), 503
        
        profile = profile_analyzer.build_user_profile(current_user.id)
        
        insights = {
            'viewing_patterns': profile.get('behavioral_patterns', {}),
            'engagement_level': profile.get('engagement_metrics', {}),
            'content_diversity': profile.get('diversity_score', 0),
            'taste_evolution': profile.get('temporal_patterns', {}),
            'cinematic_dna': profile.get('cinematic_dna', {}),
            'recommendation_accuracy': profile.get('confidence_score', 0),
            'user_segment': profile.get('user_segment', 'new_user')
        }
        
        return jsonify({
            'success': True,
            'insights': insights,
            'generated_at': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Activity insights error: {e}")
        return jsonify({'error': 'Failed to get activity insights'}), 500

@require_auth
def get_weekly_recap(current_user):
    try:
        week_start = datetime.utcnow() - timedelta(days=7)
        
        interactions = UserInteraction.query.filter(
            UserInteraction.user_id == current_user.id,
            UserInteraction.timestamp >= week_start
        ).all()
        
        if not interactions:
            return jsonify({
                'recap': {
                    'message': 'No activity this week',
                    'suggestion': 'Explore CineBrain recommendations to discover great content!'
                }
            }), 200
        
        content_ids = [i.content_id for i in interactions]
        contents = get_content_by_ids(content_ids)
        content_map = {c.id: c for c in contents}
        
        interaction_counts = Counter(i.interaction_type for i in interactions)
        
        genres_watched = []
        total_runtime = 0
        
        for interaction in interactions:
            content = content_map.get(interaction.content_id)
            if content:
                if content.genres:
                    genres_watched.extend(json.loads(content.genres or '[]'))
                if content.runtime and interaction.interaction_type in ['view', 'favorite']:
                    total_runtime += content.runtime
        
        top_genres = [genre for genre, _ in Counter(genres_watched).most_common(3)]
        
        recap = {
            'period': f"{week_start.strftime('%B %d')} - {datetime.utcnow().strftime('%B %d, %Y')}",
            'total_interactions': len(interactions),
            'breakdown': dict(interaction_counts),
            'content_discovered': len(set(content_ids)),
            'estimated_watch_time_minutes': total_runtime,
            'estimated_watch_time_hours': round(total_runtime / 60, 1),
            'top_genres_explored': top_genres,
            'most_active_day': max(
                Counter(i.timestamp.strftime('%A') for i in interactions).items(),
                key=lambda x: x[1]
            )[0] if interactions else None
        }
        
        return jsonify({
            'success': True,
            'recap': recap,
            'generated_at': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Weekly recap error: {e}")
        return jsonify({'error': 'Failed to generate weekly recap'}), 500