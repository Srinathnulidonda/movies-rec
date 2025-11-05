# backend/user/favorites.py

from flask import request, jsonify
from datetime import datetime
import json
import logging
from .utils import require_auth, db, UserInteraction, Content, format_content_for_response, recommendation_engine, profile_analyzer, cache_get, cache_set, cache_delete, get_cache_key, get_content_by_ids

logger = logging.getLogger(__name__)

@require_auth
def get_favorites(current_user):
    try:
        cache_key = get_cache_key('user_favorites', current_user.id)
        cached_favorites = cache_get(cache_key)
        
        if cached_favorites:
            return jsonify(cached_favorites), 200
        
        favorite_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='favorite'
        ).order_by(UserInteraction.timestamp.desc()).all()
        
        content_ids = [interaction.content_id for interaction in favorite_interactions]
        contents = get_content_by_ids(content_ids)
        content_map = {content.id: content for content in contents}
        
        result = []
        for interaction in favorite_interactions:
            content = content_map.get(interaction.content_id)
            if content:
                formatted_content = format_content_for_response(content, interaction)
                formatted_content['favorited_at'] = interaction.timestamp.isoformat()
                formatted_content['user_rating'] = interaction.rating
                result.append(formatted_content)
        
        favorites_analytics = {}
        if profile_analyzer:
            try:
                profile = profile_analyzer.build_user_profile(current_user.id)
                
                genre_distribution = {}
                total_rating = 0
                rated_count = 0
                
                for fav in result:
                    for genre in fav.get('genres', []):
                        genre_distribution[genre] = genre_distribution.get(genre, 0) + 1
                    
                    if fav.get('rating'):
                        total_rating += fav['rating']
                        rated_count += 1
                
                favorites_analytics = {
                    'total_favorites': len(result),
                    'avg_rating': round(total_rating / rated_count, 1) if rated_count > 0 else 0,
                    'genre_distribution': genre_distribution,
                    'taste_profile': profile.get('cinematic_dna', {}),
                    'quality_preference': total_rating / rated_count if rated_count > 0 else 7.0
                }
            except Exception as e:
                logger.warning(f"Failed to get favorites analytics: {e}")
        
        smart_collections = {}
        if len(result) >= 3:
            try:
                high_rated = [f for f in result if f.get('rating', 0) >= 8.0]
                recent_favorites = sorted(result, key=lambda x: x.get('favorited_at', ''), reverse=True)[:5]
                
                genre_groups = {}
                for fav in result:
                    for genre in fav.get('genres', []):
                        if genre not in genre_groups:
                            genre_groups[genre] = []
                        genre_groups[genre].append(fav)
                
                top_genre = max(genre_groups.keys(), key=lambda g: len(genre_groups[g])) if genre_groups else None
                
                smart_collections = {
                    'high_rated': high_rated[:5],
                    'recent_favorites': recent_favorites,
                    'top_genre_collection': {
                        'genre': top_genre,
                        'items': genre_groups.get(top_genre, [])[:5]
                    } if top_genre else None
                }
            except Exception as e:
                logger.warning(f"Failed to create smart collections: {e}")
        
        similar_recommendations = []
        if recommendation_engine and len(result) > 0:
            try:
                rec_data = recommendation_engine.generate_recommendations(
                    user_id=current_user.id,
                    limit=5,
                    context={'source': 'favorites_based', 'strategy': 'similar_to_favorites'}
                )
                similar_recommendations = rec_data.get('recommendations', [])[:3]
            except Exception as e:
                logger.warning(f"Failed to get similar recommendations: {e}")
        
        response_data = {
            'favorites': result,
            'total_count': len(result),
            'analytics': favorites_analytics,
            'smart_collections': smart_collections,
            'similar_recommendations': similar_recommendations,
            'last_updated': datetime.utcnow().isoformat()
        }
        
        cache_set(cache_key, response_data, timeout=300)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"CineBrain favorites error: {e}")
        return jsonify({'error': 'Failed to get CineBrain favorites'}), 500

@require_auth
def add_to_favorites(current_user):
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        rating = data.get('rating')
        
        if not content_id:
            return jsonify({'error': 'Content ID required'}), 400
        
        existing = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='favorite'
        ).first()
        
        if existing:
            if rating:
                existing.rating = rating
                db.session.commit()
                
                cache_key = get_cache_key('user_favorites', current_user.id)
                cache_delete(cache_key)
            
            return jsonify({
                'success': True,
                'message': 'Already in CineBrain favorites'
            }), 200
        
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='favorite',
            rating=rating,
            interaction_metadata=json.dumps(data.get('metadata', {}))
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        cache_key = get_cache_key('user_favorites', current_user.id)
        cache_delete(cache_key)
        
        if recommendation_engine:
            try:
                recommendation_engine.update_user_feedback(
                    current_user.id,
                    content_id,
                    'favorite',
                    rating
                )
            except Exception as e:
                logger.warning(f"Failed to update CineBrain recommendations: {e}")
        
        if profile_analyzer:
            try:
                profile_analyzer.update_profile_realtime(
                    current_user.id,
                    {
                        'content_id': content_id,
                        'interaction_type': 'favorite',
                        'rating': rating
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update CineBrain profile: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Added to CineBrain favorites'
        }), 201
        
    except Exception as e:
        logger.error(f"Add to CineBrain favorites error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to add to CineBrain favorites'}), 500

@require_auth
def remove_from_favorites(current_user, content_id):
    try:
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='favorite'
        ).first()
        
        if interaction:
            db.session.delete(interaction)
            db.session.commit()
            
            cache_key = get_cache_key('user_favorites', current_user.id)
            cache_delete(cache_key)
            
            if recommendation_engine:
                try:
                    recommendation_engine.update_user_feedback(
                        current_user.id,
                        content_id,
                        'remove_favorite'
                    )
                except Exception as e:
                    logger.warning(f"Failed to update CineBrain recommendations: {e}")
            
            if profile_analyzer:
                try:
                    profile_analyzer.update_profile_realtime(
                        current_user.id,
                        {
                            'content_id': content_id,
                            'interaction_type': 'remove_favorite'
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to update CineBrain profile: {e}")
            
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
        db.session.rollback()
        return jsonify({'error': 'Failed to remove from CineBrain favorites'}), 500

@require_auth
def check_favorite_status(current_user, content_id):
    try:
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

@require_auth
def get_favorites_insights(current_user):
    try:
        if not profile_analyzer:
            return jsonify({'error': 'Profile analyzer not available'}), 503
        
        favorite_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='favorite'
        ).all()
        
        if not favorite_interactions:
            return jsonify({
                'insights': {
                    'message': 'Add favorites to see personalized insights',
                    'recommendations': 'Start by favoriting content you love'
                }
            }), 200
        
        profile = profile_analyzer.build_user_profile(current_user.id)
        
        content_ids = [i.content_id for i in favorite_interactions]
        contents = get_content_by_ids(content_ids)
        
        genres_count = {}
        decades_count = {}
        languages_count = {}
        avg_ratings = []
        
        for content in contents:
            if content.genres:
                for genre in json.loads(content.genres or '[]'):
                    genres_count[genre] = genres_count.get(genre, 0) + 1
            
            if content.release_date:
                decade = (content.release_date.year // 10) * 10
                decades_count[f"{decade}s"] = decades_count.get(f"{decade}s", 0) + 1
            
            if content.languages:
                for lang in json.loads(content.languages or '[]'):
                    languages_count[lang] = languages_count.get(lang, 0) + 1
            
            if content.rating:
                avg_ratings.append(content.rating)
        
        insights = {
            'total_favorites': len(favorite_interactions),
            'top_genres': sorted(genres_count.items(), key=lambda x: x[1], reverse=True)[:5],
            'favorite_decades': sorted(decades_count.items(), key=lambda x: x[1], reverse=True)[:3],
            'preferred_languages': sorted(languages_count.items(), key=lambda x: x[1], reverse=True)[:3],
            'average_rating_preference': round(sum(avg_ratings) / len(avg_ratings), 1) if avg_ratings else 0,
            'quality_score': len([r for r in avg_ratings if r >= 8]) / len(avg_ratings) if avg_ratings else 0,
            'cinematic_dna': profile.get('cinematic_dna', {}),
            'taste_evolution': profile.get('temporal_patterns', {})
        }
        
        return jsonify({
            'success': True,
            'insights': insights,
            'generated_at': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Favorites insights error: {e}")
        return jsonify({'error': 'Failed to get favorites insights'}), 500

@require_auth
def export_favorites(current_user):
    try:
        favorite_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='favorite'
        ).order_by(UserInteraction.timestamp.desc()).all()
        
        content_ids = [interaction.content_id for interaction in favorite_interactions]
        contents = get_content_by_ids(content_ids)
        content_map = {content.id: content for content in contents}
        
        export_data = []
        for interaction in favorite_interactions:
            content = content_map.get(interaction.content_id)
            if content:
                export_data.append({
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating,
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'user_rating': interaction.rating,
                    'favorited_at': interaction.timestamp.isoformat(),
                    'tmdb_id': content.tmdb_id,
                    'imdb_id': content.imdb_id
                })
        
        return jsonify({
            'success': True,
            'export_data': export_data,
            'total_items': len(export_data),
            'exported_at': datetime.utcnow().isoformat(),
            'user': current_user.username
        }), 200
        
    except Exception as e:
        logger.error(f"Export favorites error: {e}")
        return jsonify({'error': 'Failed to export favorites'}), 500