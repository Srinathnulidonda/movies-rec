#backend/services/critics_choice.py
import math
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
from flask import Blueprint, request, jsonify, current_app
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

logger = logging.getLogger(__name__)

critics_choice_bp = Blueprint('critics_choice', __name__)

class CineBrainCriticsChoiceEngine:
    def __init__(self, app=None, db=None, models=None, services=None, cache=None):
        self.app = app
        self.db = db
        self.models = models
        self.services = services
        self.cache = cache
        
        self.tmdb_api_key = app.config.get('TMDB_API_KEY') if app else None
        self.omdb_api_key = app.config.get('OMDB_API_KEY') if app else None
        self.http_session = services.get('http_session') if services else None
        self.content_service = services.get('ContentService') if services else None
        
        self.language_priorities = ['telugu', 'english', 'hindi', 'malayalam', 'kannada', 'tamil', 'japanese']
        
        self.award_keywords = [
            'oscar', 'academy award', 'golden globe', 'cannes', 'venice', 'berlin',
            'bafta', 'critics choice', 'sundance', 'toronto', 'filmfare', 'national film award',
            'palm d\'or', 'golden bear', 'golden lion', 'saturn award', 'annie award'
        ]
        
        self.critics_thresholds = {
            'movie': {'min_rating': 7.0, 'min_votes': 1000, 'metacritic_min': 70},
            'tv': {'min_rating': 7.5, 'min_votes': 500, 'metacritic_min': 75},
            'anime': {'min_rating': 7.8, 'min_votes': 10000, 'mal_min': 8.0}
        }
        
        self.omdb_failures = 0
        self.omdb_circuit_open = False
        self.last_failure_time = None

    def get_enhanced_critics_choice(self, content_type='all', limit=20, genre=None, 
                                  language=None, time_period='all', region='global'):
        try:
            cache_key = f"cinebrain:critics:{content_type}:{limit}:{genre}:{language}:{time_period}:{region}"
            
            if self.cache:
                try:
                    cached_result = self.cache.get(cache_key)
                    if cached_result:
                        logger.info("CineBrain: Returning cached critics choice results")
                        return cached_result
                except Exception as e:
                    logger.warning(f"CineBrain cache get error: {e}")

            all_recommendations = []
            max_items = min(limit * 2, 30)
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                
                if content_type in ['all', 'movie']:
                    futures.append(executor.submit(
                        self._get_movie_critics_choice, max_items//2, genre, language, time_period, region
                    ))
                    
                if content_type in ['all', 'tv']:
                    futures.append(executor.submit(
                        self._get_tv_critics_choice, max_items//3, genre, language, time_period, region
                    ))
                    
                if content_type in ['all', 'anime']:
                    futures.append(executor.submit(
                        self._get_anime_critics_choice, max_items//3, genre, language, time_period
                    ))
                
                for future in as_completed(futures, timeout=15):
                    try:
                        batch_results = future.result()
                        if batch_results:
                            all_recommendations.extend(batch_results[:15])
                            logger.info(f"CineBrain: Collected {len(batch_results)} critics choice items from batch")
                    except Exception as e:
                        logger.error(f"CineBrain critics choice batch error: {e}")
                        continue

            if not all_recommendations:
                logger.warning("CineBrain: No critics choice recommendations found")
                return {'items': [], 'metadata': {'error': 'No recommendations found'}}

            if len(all_recommendations) > 15 or self._should_skip_omdb():
                logger.info("CineBrain: Skipping OMDb enhancement to prevent timeout")
                enhanced_recommendations = all_recommendations
            else:
                enhanced_recommendations = self._enhance_with_critics_data(all_recommendations)
                
            scored_recommendations = self._calculate_critics_score(enhanced_recommendations)
            final_recommendations = self._apply_diversity_and_ranking(
                scored_recommendations, limit, content_type, language, region
            )
            
            metadata = self._generate_metadata(final_recommendations, content_type, genre, language)
            
            result = {
                'items': final_recommendations,
                'metadata': metadata
            }
            
            if self.cache:
                try:
                    self.cache.set(cache_key, result, timeout=900)
                except Exception as e:
                    logger.warning(f"CineBrain cache set error: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"CineBrain critics choice engine error: {e}")
            return {'items': [], 'metadata': {'error': str(e)}}

    def _should_skip_omdb(self):
        if self.omdb_circuit_open:
            if self.last_failure_time and (time.time() - self.last_failure_time) > 300:
                self.omdb_circuit_open = False
                self.omdb_failures = 0
                return False
            return True
        return False
        
    def _record_omdb_failure(self):
        self.omdb_failures += 1
        self.last_failure_time = time.time()
        
        if self.omdb_failures >= 3:
            self.omdb_circuit_open = True
            logger.warning("CineBrain OMDb circuit breaker opened")

    def _get_movie_critics_choice(self, limit, genre, language, time_period, region):
        recommendations = []
        
        try:
            discovery_configs = [
                {
                    'vote_average.gte': 8.0,
                    'vote_count.gte': 5000,
                    'sort_by': 'vote_average.desc'
                },
                {
                    'vote_average.gte': 7.5,
                    'vote_count.gte': 2000,
                    'sort_by': 'popularity.desc'
                }
            ]
            
            for config in discovery_configs:
                if len(recommendations) >= limit:
                    break
                    
                params = {
                    'api_key': self.tmdb_api_key,
                    **config,
                    'include_adult': False
                }
                
                if genre:
                    genre_id = self._get_genre_id(genre, 'movie')
                    if genre_id:
                        params['with_genres'] = genre_id
                        
                if language and language != 'all':
                    lang_code = self._get_language_code(language)
                    if lang_code:
                        params['with_original_language'] = lang_code
                        
                if time_period != 'all':
                    date_range = self._get_date_range(time_period)
                    if date_range:
                        params.update(date_range)
                        
                if region != 'global':
                    params['region'] = region
                    
                for page in range(1, 3):
                    if len(recommendations) >= limit:
                        break
                        
                    params['page'] = page
                    
                    response = self._make_api_request(
                        'https://api.themoviedb.org/3/discover/movie', 
                        params
                    )
                    
                    if response and response.get('results'):
                        for item in response['results']:
                            if len(recommendations) >= limit:
                                break
                                
                            enhanced_item = self._enhance_movie_data(item)
                            if enhanced_item and self._meets_critics_criteria(enhanced_item, 'movie'):
                                recommendations.append(enhanced_item)
                                
                    time.sleep(0.2)
            
        except Exception as e:
            logger.error(f"CineBrain movie critics choice error: {e}")
            
        return recommendations

    def _get_tv_critics_choice(self, limit, genre, language, time_period, region):
        recommendations = []
        
        try:
            discovery_configs = [
                {
                    'vote_average.gte': 8.2,
                    'vote_count.gte': 1000,
                    'sort_by': 'vote_average.desc'
                },
                {
                    'vote_average.gte': 7.8,
                    'vote_count.gte': 500,
                    'sort_by': 'popularity.desc'
                }
            ]
            
            for config in discovery_configs:
                if len(recommendations) >= limit:
                    break
                    
                params = {
                    'api_key': self.tmdb_api_key,
                    **config
                }
                
                if genre:
                    genre_id = self._get_genre_id(genre, 'tv')
                    if genre_id:
                        params['with_genres'] = genre_id
                        
                if language and language != 'all':
                    lang_code = self._get_language_code(language)
                    if lang_code:
                        params['with_original_language'] = lang_code
                        
                if time_period != 'all':
                    date_range = self._get_date_range(time_period, 'tv')
                    if date_range:
                        params.update(date_range)
                        
                for page in range(1, 2):
                    if len(recommendations) >= limit:
                        break
                        
                    params['page'] = page
                    
                    response = self._make_api_request(
                        'https://api.themoviedb.org/3/discover/tv', 
                        params
                    )
                    
                    if response and response.get('results'):
                        for item in response['results']:
                            if len(recommendations) >= limit:
                                break
                                
                            enhanced_item = self._enhance_tv_data(item)
                            if enhanced_item and self._meets_critics_criteria(enhanced_item, 'tv'):
                                recommendations.append(enhanced_item)
                                
                    time.sleep(0.2)
                    
        except Exception as e:
            logger.error(f"CineBrain TV critics choice error: {e}")
            
        return recommendations

    def _get_anime_critics_choice(self, limit, genre, language, time_period):
        recommendations = []
        
        try:
            anime_configs = [
                {
                    'order_by': 'score',
                    'sort': 'desc',
                    'min_score': 8.5,
                    'status': 'complete'
                }
            ]
            
            for config in anime_configs:
                if len(recommendations) >= limit:
                    break
                    
                params = config.copy()
                
                if genre:
                    anime_genre = self._get_anime_genre(genre)
                    if anime_genre:
                        params['genres'] = anime_genre
                        
                if time_period != 'all':
                    year_range = self._get_anime_year_range(time_period)
                    if year_range:
                        params.update(year_range)
                        
                for page in range(1, 2):
                    if len(recommendations) >= limit:
                        break
                        
                    params['page'] = page
                    params['limit'] = 25
                    
                    response = self._make_api_request(
                        'https://api.jikan.moe/v4/anime', 
                        params
                    )
                    
                    if response and response.get('data'):
                        for item in response['data']:
                            if len(recommendations) >= limit:
                                break
                                
                            enhanced_item = self._enhance_anime_data(item)
                            if enhanced_item and self._meets_critics_criteria(enhanced_item, 'anime'):
                                recommendations.append(enhanced_item)
                                
                    time.sleep(1.2)
                    
        except Exception as e:
            logger.error(f"CineBrain anime critics choice error: {e}")
            
        return recommendations

    def _enhance_movie_data(self, item):
        try:
            enhanced = {
                'tmdb_id': item.get('id'),
                'title': item.get('title', ''),
                'original_title': item.get('original_title', ''),
                'content_type': 'movie',
                'release_date': item.get('release_date', ''),
                'rating': float(item.get('vote_average', 0)),
                'vote_count': item.get('vote_count', 0),
                'popularity': float(item.get('popularity', 0)),
                'overview': item.get('overview', ''),
                'poster_path': item.get('poster_path'),
                'backdrop_path': item.get('backdrop_path'),
                'genre_ids': item.get('genre_ids', []),
                'original_language': item.get('original_language', 'en'),
                'adult': item.get('adult', False)
            }
            
            if self.tmdb_api_key:
                detailed_data = self._get_movie_details(item.get('id'))
                if detailed_data:
                    enhanced.update({
                        'runtime': detailed_data.get('runtime'),
                        'budget': detailed_data.get('budget'),
                        'revenue': detailed_data.get('revenue'),
                        'production_countries': detailed_data.get('production_countries', []),
                        'genres': [g.get('name') for g in detailed_data.get('genres', [])],
                        'imdb_id': detailed_data.get('imdb_id')
                    })
                            
            return enhanced
            
        except Exception as e:
            logger.error(f"CineBrain movie enhancement error: {e}")
            return None

    def _enhance_tv_data(self, item):
        try:
            enhanced = {
                'tmdb_id': item.get('id'),
                'title': item.get('name', ''),
                'original_title': item.get('original_name', ''),
                'content_type': 'tv',
                'first_air_date': item.get('first_air_date', ''),
                'rating': float(item.get('vote_average', 0)),
                'vote_count': item.get('vote_count', 0),
                'popularity': float(item.get('popularity', 0)),
                'overview': item.get('overview', ''),
                'poster_path': item.get('poster_path'),
                'backdrop_path': item.get('backdrop_path'),
                'genre_ids': item.get('genre_ids', []),
                'original_language': item.get('original_language', 'en'),
                'origin_country': item.get('origin_country', [])
            }
            
            if self.tmdb_api_key:
                detailed_data = self._get_tv_details(item.get('id'))
                if detailed_data:
                    enhanced.update({
                        'number_of_seasons': detailed_data.get('number_of_seasons'),
                        'number_of_episodes': detailed_data.get('number_of_episodes'),
                        'episode_run_time': detailed_data.get('episode_run_time', []),
                        'in_production': detailed_data.get('in_production'),
                        'genres': [g.get('name') for g in detailed_data.get('genres', [])],
                        'networks': [n.get('name') for n in detailed_data.get('networks', [])],
                        'production_companies': detailed_data.get('production_companies', [])
                    })
                    
            return enhanced
            
        except Exception as e:
            logger.error(f"CineBrain TV enhancement error: {e}")
            return None

    def _enhance_anime_data(self, item):
        try:
            enhanced = {
                'mal_id': item.get('mal_id'),
                'title': item.get('title', ''),
                'original_title': item.get('title_japanese', ''),
                'content_type': 'anime',
                'aired_from': item.get('aired', {}).get('from', ''),
                'aired_to': item.get('aired', {}).get('to', ''),
                'rating': float(item.get('score', 0) or 0),
                'vote_count': item.get('scored_by', 0),
                'popularity': float(item.get('popularity', 0) or 0),
                'overview': item.get('synopsis', ''),
                'poster_path': item.get('images', {}).get('jpg', {}).get('image_url'),
                'backdrop_path': item.get('images', {}).get('jpg', {}).get('large_image_url'),
                'genres': [g.get('name') for g in item.get('genres', [])],
                'themes': [t.get('name') for t in item.get('themes', [])],
                'demographics': [d.get('name') for d in item.get('demographics', [])],
                'type': item.get('type'),
                'episodes': item.get('episodes'),
                'status': item.get('status'),
                'duration': item.get('duration'),
                'rating_age': item.get('rating'),
                'studios': [s.get('name') for s in item.get('studios', [])],
                'rank': item.get('rank'),
                'members': item.get('members'),
                'favorites': item.get('favorites'),
                'original_language': 'ja'
            }
            
            return enhanced
            
        except Exception as e:
            logger.error(f"CineBrain anime enhancement error: {e}")
            return None

    def _enhance_with_critics_data(self, recommendations):
        enhanced = []
        batch_size = 5
        max_enhancements = 8
        
        for i, rec in enumerate(recommendations[:max_enhancements]):
            try:
                if i >= batch_size:
                    break
                    
                if rec.get('imdb_id') and self.omdb_api_key and not self._should_skip_omdb():
                    try:
                        omdb_data = self._get_omdb_data_with_fallback(rec['imdb_id'])
                        if omdb_data:
                            rec.update({
                                'metacritic_score': self._parse_score(omdb_data.get('Metascore')),
                                'imdb_rating': self._parse_score(omdb_data.get('imdbRating')),
                                'rotten_tomatoes': self._extract_rt_score(omdb_data.get('Ratings', [])),
                                'awards': omdb_data.get('Awards', '')
                            })
                    except Exception as e:
                        logger.warning(f"CineBrain OMDb enhancement failed for {rec.get('title')}: {e}")
                        self._record_omdb_failure()
                        
                enhanced.append(rec)
                
            except Exception as e:
                logger.warning(f"CineBrain enhancement error for {rec.get('title', 'Unknown')}: {e}")
                enhanced.append(rec)
                
        enhanced.extend(recommendations[len(enhanced):])
        return enhanced

    def _get_omdb_data_with_fallback(self, imdb_id):
        try:
            cache_key = f"cinebrain:omdb:{imdb_id}"
            if self.cache:
                cached = self.cache.get(cache_key)
                if cached:
                    return cached
            
            params = {'apikey': self.omdb_api_key, 'i': imdb_id}
            
            if self.http_session:
                response = self.http_session.get(
                    'http://www.omdbapi.com/', 
                    params=params, 
                    timeout=3
                )
            else:
                response = requests.get(
                    'http://www.omdbapi.com/', 
                    params=params, 
                    timeout=3
                )
                
            if response.status_code == 200:
                data = response.json()
                if self.cache:
                    self.cache.set(cache_key, data, timeout=86400)
                return data
                
        except Exception as e:
            logger.warning(f"CineBrain OMDb API error for {imdb_id}: {e}")
            
        return None

    def _meets_critics_criteria(self, item, content_type):
        try:
            thresholds = self.critics_thresholds.get(content_type, {})
            
            rating = item.get('rating', 0)
            vote_count = item.get('vote_count', 0)
            
            if rating < thresholds.get('min_rating', 7.0):
                return False
                
            if vote_count < thresholds.get('min_votes', 100):
                return False
                
            if content_type == 'anime':
                if rating < thresholds.get('mal_min', 8.0):
                    return False
                if vote_count < 5000:
                    return False
                    
            if item.get('metacritic_score'):
                if item['metacritic_score'] < thresholds.get('metacritic_min', 70):
                    return False
                    
            if item.get('adult', False) and content_type == 'movie':
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"CineBrain criteria check error: {e}")
            return False

    def _calculate_critics_score(self, recommendations):
        try:
            for item in recommendations:
                score_components = []
                
                tmdb_score = (item.get('rating', 0) / 10) * 30
                score_components.append(tmdb_score)
                
                vote_weight = min(math.log10(max(item.get('vote_count', 1), 1)) / 6, 1) * 20
                score_components.append(vote_weight)
                
                if item.get('metacritic_score'):
                    metacritic_weight = (item['metacritic_score'] / 100) * 25
                    score_components.append(metacritic_weight)
                else:
                    score_components.append(0)
                    
                if item.get('imdb_rating'):
                    imdb_weight = (item['imdb_rating'] / 10) * 15
                    score_components.append(imdb_weight)
                else:
                    score_components.append(0)
                    
                if item.get('rotten_tomatoes'):
                    rt_weight = (item['rotten_tomatoes'] / 100) * 10
                    score_components.append(rt_weight)
                else:
                    score_components.append(0)
                    
                language_bonus = self._get_language_bonus(item.get('original_language', 'en'))
                score_components.append(language_bonus)
                
                award_bonus = self._get_award_bonus(item.get('awards', ''))
                score_components.append(award_bonus)
                
                recency_bonus = self._get_recency_bonus(item)
                score_components.append(recency_bonus)
                
                diversity_bonus = self._get_diversity_bonus(item)
                score_components.append(diversity_bonus)
                
                final_score = sum(score_components)
                item['critics_score'] = min(final_score, 100)
                item['score_breakdown'] = {
                    'tmdb_rating': round(tmdb_score, 2),
                    'vote_weight': round(vote_weight, 2),
                    'metacritic': round(score_components[2], 2),
                    'imdb': round(score_components[3], 2),
                    'rotten_tomatoes': round(score_components[4], 2),
                    'language_bonus': round(language_bonus, 2),
                    'award_bonus': round(award_bonus, 2),
                    'recency_bonus': round(recency_bonus, 2),
                    'diversity_bonus': round(diversity_bonus, 2)
                }
                
            return recommendations
            
        except Exception as e:
            logger.error(f"CineBrain scoring error: {e}")
            return recommendations

    def _apply_diversity_and_ranking(self, recommendations, limit, content_type, language, region):
        try:
            sorted_recs = sorted(recommendations, key=lambda x: x.get('critics_score', 0), reverse=True)
            
            final_recs = []
            seen_titles = set()
            genre_counts = defaultdict(int)
            language_counts = defaultdict(int)
            year_counts = defaultdict(int)
            
            max_per_genre = max(2, limit // 8)
            max_per_language = max(3, limit // 6)
            max_per_year = max(2, limit // 10)
            
            for rec in sorted_recs:
                if len(final_recs) >= limit:
                    break
                    
                title_key = rec.get('title', '').lower().strip()
                if title_key in seen_titles:
                    continue
                    
                rec_genres = rec.get('genres', [])
                rec_language = rec.get('original_language', 'en')
                rec_year = self._extract_year(rec)
                
                should_add = True
                
                if rec_genres:
                    main_genre = rec_genres[0] if rec_genres else 'unknown'
                    if genre_counts[main_genre] >= max_per_genre:
                        should_add = False
                        
                if language_counts[rec_language] >= max_per_language:
                    should_add = False
                    
                if rec_year and year_counts[rec_year] >= max_per_year:
                    should_add = False
                    
                if should_add:
                    final_rec = self._format_recommendation(rec)
                    if final_rec:
                        final_recs.append(final_rec)
                        seen_titles.add(title_key)
                        
                        if rec_genres:
                            genre_counts[rec_genres[0]] += 1
                        language_counts[rec_language] += 1
                        if rec_year:
                            year_counts[rec_year] += 1
                            
            if len(final_recs) < limit and language == 'telugu':
                telugu_boost = [r for r in sorted_recs if r.get('original_language') == 'te'][:5]
                for rec in telugu_boost:
                    if len(final_recs) >= limit:
                        break
                    title_key = rec.get('title', '').lower().strip()
                    if title_key not in seen_titles:
                        final_rec = self._format_recommendation(rec)
                        if final_rec:
                            final_recs.append(final_rec)
                            seen_titles.add(title_key)
                            
            return final_recs
            
        except Exception as e:
            logger.error(f"CineBrain diversity ranking error: {e}")
            return recommendations[:limit]

    def _format_recommendation(self, rec):
        try:
            content_id = rec.get('tmdb_id') or rec.get('mal_id')
            
            # Generate slug without content service if it fails
            slug = self._generate_slug(rec.get('title', ''))
            if content_id:
                slug = f"{slug}-{content_id}"
                
            youtube_url = None
            if rec.get('youtube_trailer_id'):
                youtube_url = f"https://www.youtube.com/watch?v={rec['youtube_trailer_id']}"
                
            poster_path = rec.get('poster_path')
            if poster_path and not poster_path.startswith('http'):
                poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
                
            backdrop_path = rec.get('backdrop_path')
            if backdrop_path and not backdrop_path.startswith('http'):
                backdrop_path = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
                
            return {
                'id': content_id,
                'slug': slug,
                'title': rec.get('title', ''),
                'original_title': rec.get('original_title'),
                'content_type': rec.get('content_type'),
                'genres': rec.get('genres', []),
                'languages': [self._get_language_name(rec.get('original_language', 'en'))],
                'rating': rec.get('rating', 0),
                'vote_count': rec.get('vote_count', 0),
                'critics_score': round(rec.get('critics_score', 0), 1),
                'metacritic_score': rec.get('metacritic_score'),
                'imdb_rating': rec.get('imdb_rating'),
                'rotten_tomatoes': rec.get('rotten_tomatoes'),
                'release_date': self._get_release_date(rec),
                'runtime': rec.get('runtime') or rec.get('duration'),
                'poster_path': poster_path,
                'backdrop_path': backdrop_path,
                'overview': (rec.get('overview', '') or '')[:200] + '...' if rec.get('overview') else '',
                'youtube_trailer': youtube_url,
                'awards': rec.get('awards', ''),
                'is_critics_choice': True,
                'score_breakdown': rec.get('score_breakdown', {}),
                'cinebrain_enhanced': True
            }
            
        except Exception as e:
            logger.error(f"CineBrain format recommendation error: {e}")
            return None
    def _get_language_bonus(self, language):
        try:
            if language in ['te']:
                return 8
            elif language in ['hi', 'ta', 'ml', 'kn']:
                return 5
            elif language in ['en']:
                return 3
            elif language in ['ja', 'ko']:
                return 2
            else:
                return 0
        except:
            return 0

    def _get_award_bonus(self, awards_text):
        try:
            if not awards_text:
                return 0
                
            awards_lower = awards_text.lower()
            bonus = 0
            
            for keyword in self.award_keywords:
                if keyword in awards_lower:
                    if keyword in ['oscar', 'academy award']:
                        bonus += 10
                    elif keyword in ['golden globe', 'bafta']:
                        bonus += 7
                    elif keyword in ['cannes', 'venice', 'berlin']:
                        bonus += 6
                    elif keyword in ['filmfare', 'national film award']:
                        bonus += 8
                    else:
                        bonus += 3
                        
            return min(bonus, 15)
        except:
            return 0

    def _get_recency_bonus(self, item):
        try:
            current_year = datetime.now().year
            release_year = self._extract_year(item)
            
            if not release_year:
                return 0
                
            years_ago = current_year - release_year
            
            if years_ago <= 1:
                return 5
            elif years_ago <= 3:
                return 3
            elif years_ago <= 5:
                return 1
            else:
                return 0
        except:
            return 0

    def _get_diversity_bonus(self, item):
        try:
            bonus = 0
            
            if item.get('original_language') != 'en':
                bonus += 2
                
            genres = item.get('genres', [])
            if any(g in ['Animation', 'Documentary', 'Foreign'] for g in genres):
                bonus += 2
                
            if item.get('content_type') == 'anime':
                bonus += 3
                
            return bonus
        except:
            return 0

    def _generate_metadata(self, recommendations, content_type, genre, language):
        try:
            total_items = len(recommendations)
            
            if total_items == 0:
                return {
                    'total_recommendations': 0,
                    'error': 'No recommendations found',
                    'cinebrain_service': 'enhanced_critics_choice'
                }
            
            genre_dist = defaultdict(int)
            language_dist = defaultdict(int)
            year_dist = defaultdict(int)
            score_ranges = {'90-100': 0, '80-89': 0, '70-79': 0, '60-69': 0}
            
            avg_score = 0
            avg_rating = 0
            
            for rec in recommendations:
                for genre_item in rec.get('genres', []):
                    genre_dist[genre_item] += 1
                    
                for lang in rec.get('languages', []):
                    language_dist[lang] += 1
                    
                year = self._extract_year_from_date(rec.get('release_date', ''))
                if year:
                    year_dist[str(year)] += 1
                    
                score = rec.get('critics_score', 0)
                if score >= 90:
                    score_ranges['90-100'] += 1
                elif score >= 80:
                    score_ranges['80-89'] += 1
                elif score >= 70:
                    score_ranges['70-79'] += 1
                else:
                    score_ranges['60-69'] += 1
                    
                avg_score += score
                avg_rating += rec.get('rating', 0)
                
            return {
                'total_recommendations': total_items,
                'content_type_filter': content_type,
                'genre_filter': genre,
                'language_filter': language,
                'average_critics_score': round(avg_score / max(total_items, 1), 2),
                'average_rating': round(avg_rating / max(total_items, 1), 2),
                'genre_distribution': dict(genre_dist),
                'language_distribution': dict(language_dist),
                'year_distribution': dict(year_dist),
                'score_distribution': score_ranges,
                'algorithm': 'cinebrain_enhanced_critics_choice_v2',
                'sources': ['tmdb', 'omdb', 'jikan', 'metacritic', 'rotten_tomatoes'],
                'telugu_priority': True,
                'diversity_applied': True,
                'timestamp': datetime.utcnow().isoformat(),
                'cinebrain_service': 'enhanced_critics_choice'
            }
            
        except Exception as e:
            logger.error(f"CineBrain metadata generation error: {e}")
            return {'error': str(e)}

    def _make_api_request(self, url, params):
        try:
            if self.http_session:
                response = self.http_session.get(url, params=params, timeout=5)
            else:
                response = requests.get(url, params=params, timeout=5)
                
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"CineBrain API request error for {url}: {e}")
        return None

    def _get_movie_details(self, tmdb_id):
        if not tmdb_id or not self.tmdb_api_key:
            return None
        return self._make_api_request(
            f'https://api.themoviedb.org/3/movie/{tmdb_id}',
            {'api_key': self.tmdb_api_key}
        )

    def _get_tv_details(self, tmdb_id):
        if not tmdb_id or not self.tmdb_api_key:
            return None
        return self._make_api_request(
            f'https://api.themoviedb.org/3/tv/{tmdb_id}',
            {'api_key': self.tmdb_api_key}
        )

    def _parse_score(self, score_str):
        try:
            if score_str and score_str != 'N/A':
                return float(score_str)
        except:
            pass
        return None

    def _extract_rt_score(self, ratings_list):
        try:
            for rating in ratings_list:
                if rating.get('Source') == 'Rotten Tomatoes':
                    score_str = rating.get('Value', '').replace('%', '')
                    return float(score_str)
        except:
            pass
        return None

    def _get_genre_id(self, genre_name, content_type):
        genre_maps = {
            'movie': {
                'action': 28, 'adventure': 12, 'animation': 16, 'comedy': 35,
                'crime': 80, 'documentary': 99, 'drama': 18, 'family': 10751,
                'fantasy': 14, 'history': 36, 'horror': 27, 'music': 10402,
                'mystery': 9648, 'romance': 10749, 'science fiction': 878,
                'thriller': 53, 'war': 10752, 'western': 37
            },
            'tv': {
                'action & adventure': 10759, 'animation': 16, 'comedy': 35,
                'crime': 80, 'documentary': 99, 'drama': 18, 'family': 10751,
                'kids': 10762, 'mystery': 9648, 'news': 10763, 'reality': 10764,
                'sci-fi & fantasy': 10765, 'soap': 10766, 'talk': 10767,
                'war & politics': 10768, 'western': 37
            }
        }
        return genre_maps.get(content_type, {}).get(genre_name.lower())

    def _get_language_code(self, language):
        lang_codes = {
            'telugu': 'te', 'hindi': 'hi', 'tamil': 'ta', 'kannada': 'kn',
            'malayalam': 'ml', 'english': 'en', 'japanese': 'ja', 'korean': 'ko'
        }
        return lang_codes.get(language.lower())

    def _get_language_name(self, code):
        code_to_name = {
            'te': 'Telugu', 'hi': 'Hindi', 'ta': 'Tamil', 'kn': 'Kannada',
            'ml': 'Malayalam', 'en': 'English', 'ja': 'Japanese', 'ko': 'Korean'
        }
        return code_to_name.get(code, code.upper())

    def _get_date_range(self, time_period, content_type='movie'):
        current_year = datetime.now().year
        current_date = datetime.now().date()
        
        if time_period == 'recent':
            start_date = f"{current_year - 2}-01-01"
            end_date = current_date.isoformat()
        elif time_period == 'classic':
            start_date = "1950-01-01"
            end_date = f"{current_year - 10}-12-31"
        elif time_period == 'modern':
            start_date = f"{current_year - 10}-01-01"
            end_date = current_date.isoformat()
        else:
            return None
            
        if content_type == 'movie':
            return {
                'primary_release_date.gte': start_date,
                'primary_release_date.lte': end_date
            }
        else:
            return {
                'first_air_date.gte': start_date,
                'first_air_date.lte': end_date
            }

    def _extract_year(self, item):
        try:
            date_str = (item.get('release_date') or 
                       item.get('first_air_date') or 
                       item.get('aired_from', ''))
            if date_str:
                return int(date_str[:4])
        except:
            pass
        return None

    def _extract_year_from_date(self, date_str):
        try:
            if date_str:
                return int(date_str[:4])
        except:
            pass
        return None

    def _get_release_date(self, rec):
        return (rec.get('release_date') or 
                rec.get('first_air_date') or 
                rec.get('aired_from', ''))

    def _get_anime_genre(self, genre):
        anime_genre_map = {
            'action': 'Action',
            'adventure': 'Adventure', 
            'comedy': 'Comedy',
            'drama': 'Drama',
            'fantasy': 'Fantasy',
            'romance': 'Romance',
            'thriller': 'Thriller',
            'horror': 'Horror'
        }
        return anime_genre_map.get(genre.lower())

    def _get_anime_year_range(self, time_period):
        current_year = datetime.now().year
        
        if time_period == 'recent':
            return {'start_date': f"{current_year - 2}-01-01"}
        elif time_period == 'classic':
            return {'end_date': f"{current_year - 10}-12-31"}
        elif time_period == 'modern':
            return {
                'start_date': f"{current_year - 10}-01-01",
                'end_date': f"{current_year}-12-31"
            }
        return None

    def _generate_slug(self, title):
        try:
            slug = title.lower()
            slug = ''.join(c if c.isalnum() or c == ' ' else '' for c in slug)
            slug = '-'.join(slug.split())
            return slug[:50] if len(slug) > 50 else slug
        except:
            return f"content-{int(time.time())}"

cinebrain_critics_engine = None

@critics_choice_bp.route('/api/recommendations/critics-choice', methods=['GET'])
def get_enhanced_critics_choice():
    try:
        content_type = request.args.get('type', 'all')
        limit = min(int(request.args.get('limit', 8)), 12)
        genre = request.args.get('genre')
        language = request.args.get('language')
        time_period = request.args.get('time_period', 'all')
        region = request.args.get('region', 'global')
        
        if not cinebrain_critics_engine:
            return jsonify({
                'error': 'CineBrain Critics Choice service not available',
                'recommendations': [],
                'cinebrain_service': 'enhanced_critics_choice'
            }), 503
        
        recommendations = cinebrain_critics_engine.get_enhanced_critics_choice(
            content_type=content_type,
            limit=limit,
            genre=genre,
            language=language,
            time_period=time_period,
            region=region
        )
        
        return jsonify({
            'recommendations': recommendations['items'],
            'metadata': recommendations['metadata'],
            'cinebrain_service': 'enhanced_critics_choice'
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain enhanced critics choice endpoint error: {e}")
        return jsonify({
            'error': 'Failed to get CineBrain critics choice',
            'recommendations': [],
            'cinebrain_service': 'enhanced_critics_choice'
        }), 500

def init_critics_choice_service(app, db, models, services, cache):
    try:
        global cinebrain_critics_engine
        cinebrain_critics_engine = CineBrainCriticsChoiceEngine(app, db, models, services, cache)
        
        logger.info("CineBrain Critics Choice service initialized successfully")
        return cinebrain_critics_engine
        
    except Exception as e:
        logger.error(f"CineBrain Critics Choice service initialization error: {e}")
        return None