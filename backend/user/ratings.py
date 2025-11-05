# backend/user/ratings.py

from flask import request, jsonify
from datetime import datetime, timedelta
import json
import logging
import numpy as np
from .utils import require_auth, db, UserInteraction, Content, recommendation_engine, profile_analyzer, cache_get, cache_set, cache_delete, get_cache_key, get_content_by_ids
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

@require_auth
def get_user_ratings(current_user):
    try:
        cache_key = get_cache_key('user_ratings', current_user.id)
        cached_ratings = cache_get(cache_key)
        
        if cached_ratings:
            return jsonify(cached_ratings), 200
        
        rating_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='rating'
        ).filter(UserInteraction.rating.isnot(None)).order_by(
            UserInteraction.timestamp.desc()
        ).all()
        
        content_ids = [interaction.content_id for interaction in rating_interactions]
        contents = get_content_by_ids(content_ids)
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
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'user_rating': interaction.rating,
                    'imdb_rating': content.rating,
                    'rated_at': interaction.timestamp.isoformat(),
                    'rating_difference': round(interaction.rating - (content.rating or 0), 1)
                })
        
        ratings = [interaction.rating for interaction in rating_interactions]
        
        basic_stats = {
            'total_ratings': len(ratings),
            'average_rating': round(sum(ratings) / len(ratings), 1) if ratings else 0,
            'highest_rating': max(ratings) if ratings else 0,
            'lowest_rating': min(ratings) if ratings else 0,
            'rating_distribution': dict(Counter(ratings)),
            'median_rating': round(np.median(ratings), 1) if ratings else 0,
            'std_deviation': round(np.std(ratings), 2) if ratings else 0
        }
        
        advanced_stats = calculate_advanced_rating_stats(rating_interactions, contents, content_map)
        
        rating_trends = analyze_rating_trends(rating_interactions)
        
        personalized_insights = {}
        if profile_analyzer:
            try:
                profile = profile_analyzer.build_user_profile(current_user.id)
                personalized_insights = {
                    'rating_patterns': profile.get('implicit_preferences', {}).get('rating_patterns', {}),
                    'genre_rating_preferences': analyze_genre_rating_patterns(rating_interactions, content_map),
                    'quality_preference_score': profile.get('recommendation_context', {}).get('quality_preference', 7.0),
                    'rating_consistency': calculate_rating_consistency(ratings),
                    'critical_vs_popular': analyze_critical_vs_popular_taste(rating_interactions, content_map)
                }
            except Exception as e:
                logger.warning(f"Failed to get personalized rating insights: {e}")
        
        response_data = {
            'ratings': result,
            'stats': {**basic_stats, **advanced_stats},
            'trends': rating_trends,
            'insights': personalized_insights,
            'last_updated': datetime.utcnow().isoformat()
        }
        
        cache_set(cache_key, response_data, timeout=300)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"CineBrain user ratings error: {e}")
        return jsonify({'error': 'Failed to get CineBrain user ratings'}), 500

def calculate_advanced_rating_stats(rating_interactions, contents, content_map):
    try:
        if not rating_interactions:
            return {}
        
        content_type_ratings = defaultdict(list)
        genre_ratings = defaultdict(list)
        decade_ratings = defaultdict(list)
        language_ratings = defaultdict(list)
        
        for interaction in rating_interactions:
            content = content_map.get(interaction.content_id)
            if not content:
                continue
            
            content_type_ratings[content.content_type].append(interaction.rating)
            
            if content.genres:
                for genre in json.loads(content.genres or '[]'):
                    genre_ratings[genre].append(interaction.rating)
            
            if content.release_date:
                decade = (content.release_date.year // 10) * 10
                decade_ratings[f"{decade}s"].append(interaction.rating)
            
            if content.languages:
                for language in json.loads(content.languages or '[]'):
                    language_ratings[language].append(interaction.rating)
        
        return {
            'content_type_averages': {
                ct: round(np.mean(ratings), 1) 
                for ct, ratings in content_type_ratings.items()
            },
            'genre_averages': {
                genre: round(np.mean(ratings), 1) 
                for genre, ratings in sorted(genre_ratings.items(), key=lambda x: np.mean(x[1]), reverse=True)[:10]
            },
            'decade_averages': {
                decade: round(np.mean(ratings), 1) 
                for decade, ratings in decade_ratings.items()
            },
            'language_averages': {
                lang: round(np.mean(ratings), 1) 
                for lang, ratings in language_ratings.items()
            },
            'rating_range': max([i.rating for i in rating_interactions]) - min([i.rating for i in rating_interactions]) if rating_interactions else 0
        }
        
    except Exception as e:
        logger.error(f"Error calculating advanced rating stats: {e}")
        return {}

def analyze_rating_trends(rating_interactions):
    try:
        if len(rating_interactions) < 5:
            return {'trend': 'insufficient_data'}
        
        recent_ratings = sorted(rating_interactions, key=lambda x: x.timestamp)[-10:]
        older_ratings = sorted(rating_interactions, key=lambda x: x.timestamp)[:-10] if len(rating_interactions) > 10 else []
        
        recent_avg = np.mean([r.rating for r in recent_ratings])
        older_avg = np.mean([r.rating for r in older_ratings]) if older_ratings else recent_avg
        
        trend = 'stable'
        if recent_avg - older_avg > 0.5:
            trend = 'improving'
        elif older_avg - recent_avg > 0.5:
            trend = 'declining'
        
        monthly_averages = defaultdict(list)
        for interaction in rating_interactions:
            month_key = interaction.timestamp.strftime('%Y-%m')
            monthly_averages[month_key].append(interaction.rating)
        
        monthly_trends = {
            month: round(np.mean(ratings), 1) 
            for month, ratings in sorted(monthly_averages.items())[-6:]
        }
        
        return {
            'overall_trend': trend,
            'recent_average': round(recent_avg, 1),
            'historical_average': round(older_avg, 1),
            'trend_change': round(recent_avg - older_avg, 1),
            'monthly_averages': monthly_trends
        }
        
    except Exception as e:
        logger.error(f"Error analyzing rating trends: {e}")
        return {'trend': 'error'}

def analyze_genre_rating_patterns(rating_interactions, content_map):
    try:
        genre_ratings = defaultdict(list)
        genre_counts = defaultdict(int)
        
        for interaction in rating_interactions:
            content = content_map.get(interaction.content_id)
            if content and content.genres:
                for genre in json.loads(content.genres or '[]'):
                    genre_ratings[genre].append(interaction.rating)
                    genre_counts[genre] += 1
        
        genre_analysis = {}
        for genre, ratings in genre_ratings.items():
            if len(ratings) >= 3:
                genre_analysis[genre] = {
                    'average_rating': round(np.mean(ratings), 1),
                    'total_rated': len(ratings),
                    'highest_rating': max(ratings),
                    'consistency': round(1 - (np.std(ratings) / np.mean(ratings)), 2) if np.mean(ratings) > 0 else 0
                }
        
        return dict(sorted(genre_analysis.items(), key=lambda x: x[1]['average_rating'], reverse=True)[:8])
        
    except Exception as e:
        logger.error(f"Error analyzing genre rating patterns: {e}")
        return {}

def calculate_rating_consistency(ratings):
    try:
        if len(ratings) < 5:
            return 0
        
        std_dev = np.std(ratings)
        mean_rating = np.mean(ratings)
        
        consistency_score = max(0, 1 - (std_dev / 5))
        
        return round(consistency_score, 2)
        
    except Exception as e:
        logger.error(f"Error calculating rating consistency: {e}")
        return 0

def analyze_critical_vs_popular_taste(rating_interactions, content_map):
    try:
        user_ratings = []
        imdb_ratings = []
        popularity_scores = []
        
        for interaction in rating_interactions:
            content = content_map.get(interaction.content_id)
            if content and content.rating:
                user_ratings.append(interaction.rating)
                imdb_ratings.append(content.rating)
                if content.popularity:
                    popularity_scores.append(content.popularity)
        
        if len(user_ratings) < 5:
            return {'analysis': 'insufficient_data'}
        
        correlation_with_critics = np.corrcoef(user_ratings, imdb_ratings)[0, 1] if len(user_ratings) > 1 else 0
        
        high_rated_unpopular = sum(1 for i, (ur, ir, pop) in enumerate(zip(user_ratings, imdb_ratings, popularity_scores)) 
                                  if ur >= 8 and ir >= 7.5 and pop < 20) if popularity_scores else 0
        
        mainstream_preference = sum(1 for pop in popularity_scores if pop > 50) / len(popularity_scores) if popularity_scores else 0
        
        taste_profile = 'balanced'
        if correlation_with_critics > 0.7:
            taste_profile = 'critical'
        elif mainstream_preference > 0.7:
            taste_profile = 'mainstream'
        elif high_rated_unpopular > len(user_ratings) * 0.3:
            taste_profile = 'indie'
        
        return {
            'taste_profile': taste_profile,
            'correlation_with_critics': round(correlation_with_critics, 2),
            'mainstream_preference': round(mainstream_preference, 2),
            'hidden_gems_discovered': high_rated_unpopular
        }
        
    except Exception as e:
        logger.error(f"Error analyzing critical vs popular taste: {e}")
        return {'analysis': 'error'}

@require_auth
def add_rating(current_user):
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        rating = data.get('rating')
        
        if not content_id or rating is None:
            return jsonify({'error': 'Content ID and rating required'}), 400
        
        if not (1 <= rating <= 10):
            return jsonify({'error': 'Rating must be between 1 and 10'}), 400
        
        existing = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='rating'
        ).first()
        
        if existing:
            old_rating = existing.rating
            existing.rating = rating
            existing.timestamp = datetime.utcnow()
            message = f'Rating updated from {old_rating} to {rating}'
        else:
            interaction = UserInteraction(
                user_id=current_user.id,
                content_id=content_id,
                interaction_type='rating',
                rating=rating,
                interaction_metadata=json.dumps(data.get('metadata', {}))
            )
            db.session.add(interaction)
            message = 'Rating added successfully'
        
        db.session.commit()
        
        cache_key = get_cache_key('user_ratings', current_user.id)
        cache_delete(cache_key)
        
        if recommendation_engine:
            try:
                recommendation_engine.update_user_feedback(
                    current_user.id,
                    content_id,
                    'rating',
                    rating
                )
            except Exception as e:
                logger.warning(f"Failed to update recommendations: {e}")
        
        if profile_analyzer:
            try:
                profile_analyzer.update_profile_realtime(
                    current_user.id,
                    {
                        'content_id': content_id,
                        'interaction_type': 'rating',
                        'rating': rating
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update profile: {e}")
        
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
    try:
        interaction = UserInteraction.query.filter_by(
            user_id=current_user.id,
            content_id=content_id,
            interaction_type='rating'
        ).first()
        
        if interaction:
            db.session.delete(interaction)
            db.session.commit()
            
            cache_key = get_cache_key('user_ratings', current_user.id)
            cache_delete(cache_key)
            
            if recommendation_engine:
                try:
                    recommendation_engine.update_user_feedback(
                        current_user.id,
                        content_id,
                        'remove_rating'
                    )
                except Exception as e:
                    logger.warning(f"Failed to update recommendations: {e}")
            
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

@require_auth
def get_rating_recommendations(current_user):
    try:
        if not recommendation_engine:
            return jsonify({'error': 'Recommendation engine not available'}), 503
        
        rating_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='rating'
        ).filter(UserInteraction.rating >= 8).all()
        
        if len(rating_interactions) < 3:
            return jsonify({
                'recommendations': [],
                'message': 'Rate more content (8+ stars) to get personalized recommendations'
            }), 200
        
        recommendations = recommendation_engine.generate_recommendations(
            user_id=current_user.id,
            limit=15,
            context={'source': 'high_ratings_based', 'min_user_rating': 8}
        )
        
        return jsonify({
            'success': True,
            'recommendations': recommendations.get('recommendations', []),
            'metadata': recommendations.get('metadata', {}),
            'based_on_high_ratings': True,
            'high_rated_count': len(rating_interactions)
        }), 200
        
    except Exception as e:
        logger.error(f"Rating recommendations error: {e}")
        return jsonify({'error': 'Failed to get rating-based recommendations'}), 500

@require_auth
def export_ratings(current_user):
    try:
        rating_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='rating'
        ).filter(UserInteraction.rating.isnot(None)).order_by(
            UserInteraction.timestamp.desc()
        ).all()
        
        content_ids = [interaction.content_id for interaction in rating_interactions]
        contents = get_content_by_ids(content_ids)
        content_map = {content.id: content for content in contents}
        
        export_data = []
        for interaction in rating_interactions:
            content = content_map.get(interaction.content_id)
            if content:
                export_data.append({
                    'title': content.title,
                    'content_type': content.content_type,
                    'user_rating': interaction.rating,
                    'imdb_rating': content.rating,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'rated_at': interaction.timestamp.isoformat(),
                    'tmdb_id': content.tmdb_id,
                    'imdb_id': content.imdb_id
                })
        
        return jsonify({
            'success': True,
            'export_data': export_data,
            'total_ratings': len(export_data),
            'exported_at': datetime.utcnow().isoformat(),
            'user': current_user.username
        }), 200
        
    except Exception as e:
        logger.error(f"Export ratings error: {e}")
        return jsonify({'error': 'Failed to export ratings'}), 500