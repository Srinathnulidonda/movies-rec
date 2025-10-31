"""
CineBrain Metadata Utilities
Common helpers for merging and processing metadata from different sources
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class MetadataUtils:
    """Utilities for processing and merging metadata"""
    
    def __init__(self):
        # Genre mappings for consistency
        self.genre_mappings = {
            'sci-fi': 'Science Fiction',
            'rom-com': 'Romantic Comedy',
            'thriller': 'Thriller',
            'horror': 'Horror',
            'action': 'Action',
            'drama': 'Drama',
            'comedy': 'Comedy',
            'documentary': 'Documentary',
            'animation': 'Animation',
            'family': 'Family',
            'fantasy': 'Fantasy',
            'crime': 'Crime',
            'mystery': 'Mystery',
            'romance': 'Romance',
            'adventure': 'Adventure',
            'western': 'Western',
            'war': 'War',
            'music': 'Music',
            'history': 'History',
            'biography': 'Biography',
            'sport': 'Sport'
        }
        
        # Language mappings
        self.language_mappings = {
            'en': 'English',
            'hi': 'Hindi',
            'te': 'Telugu',
            'ta': 'Tamil',
            'ml': 'Malayalam',
            'kn': 'Kannada',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'fr': 'French',
            'de': 'German',
            'es': 'Spanish',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian'
        }
        
    def build_synopsis(self, content: Any, tmdb_data: Dict, omdb_data: Dict) -> Dict:
        """Build comprehensive synopsis from multiple sources"""
        try:
            synopsis = {
                'overview': '',
                'plot': '',
                'tagline': '',
                'content_warnings': [],
                'themes': [],
                'keywords': [],
                'synopsis_sources': []
            }
            
            # Overview (primary source: content DB, fallback: TMDB)
            if content.overview:
                synopsis['overview'] = content.overview
                synopsis['synopsis_sources'].append('database')
            elif tmdb_data.get('overview'):
                synopsis['overview'] = tmdb_data['overview']
                synopsis['synopsis_sources'].append('tmdb')
            
            # Plot (detailed - from OMDb if available)
            if omdb_data.get('Plot') and omdb_data['Plot'] != 'N/A':
                synopsis['plot'] = omdb_data['Plot']
                synopsis['synopsis_sources'].append('omdb')
            elif synopsis['overview']:
                synopsis['plot'] = synopsis['overview']
            
            # Tagline
            if tmdb_data.get('tagline'):
                synopsis['tagline'] = tmdb_data['tagline']
            
            # Content warnings and ratings
            content_warnings = self._extract_content_warnings(tmdb_data, omdb_data)
            synopsis['content_warnings'] = content_warnings
            
            # Keywords and themes
            keywords = self._extract_keywords(tmdb_data)
            synopsis['keywords'] = keywords
            
            themes = self._extract_themes(synopsis['overview'], synopsis['plot'], keywords)
            synopsis['themes'] = themes
            
            return synopsis
            
        except Exception as e:
            logger.error(f"Error building synopsis: {e}")
            return {
                'overview': getattr(content, 'overview', ''),
                'plot': '',
                'tagline': '',
                'content_warnings': [],
                'themes': [],
                'keywords': [],
                'synopsis_sources': []
            }
    
    def build_ratings(self, content: Any, tmdb_data: Dict, omdb_data: Dict) -> Dict:
        """Build comprehensive ratings from multiple sources"""
        try:
            ratings = {
                'tmdb': {'score': 0, 'count': 0, 'scale': '0-10'},
                'imdb': {'score': 0, 'votes': 'N/A', 'scale': '0-10'},
                'omdb_sources': [],
                'composite_score': 0,
                'confidence_level': 'low',
                'critics_consensus': '',
                'audience_score': None
            }
            
            # TMDB ratings
            tmdb_rating = content.rating or tmdb_data.get('vote_average', 0)
            tmdb_count = content.vote_count or tmdb_data.get('vote_count', 0)
            
            if tmdb_rating > 0:
                ratings['tmdb']['score'] = round(float(tmdb_rating), 1)
                ratings['tmdb']['count'] = int(tmdb_count)
            
            # IMDb ratings from OMDb
            if omdb_data.get('imdbRating') and omdb_data['imdbRating'] != 'N/A':
                try:
                    ratings['imdb']['score'] = round(float(omdb_data['imdbRating']), 1)
                    ratings['imdb']['votes'] = omdb_data.get('imdbVotes', 'N/A')
                except (ValueError, TypeError):
                    pass
            
            # Process OMDb ratings from various sources
            if omdb_data.get('Ratings'):
                for rating in omdb_data['Ratings']:
                    processed_rating = self._process_omdb_rating(rating)
                    if processed_rating:
                        ratings['omdb_sources'].append(processed_rating)
            
            # Calculate composite score
            composite_info = self._calculate_composite_score(ratings)
            ratings.update(composite_info)
            
            return ratings
            
        except Exception as e:
            logger.error(f"Error building ratings: {e}")
            return {
                'tmdb': {'score': 0, 'count': 0, 'scale': '0-10'},
                'imdb': {'score': 0, 'votes': 'N/A', 'scale': '0-10'},
                'omdb_sources': [],
                'composite_score': 0,
                'confidence_level': 'low',
                'critics_consensus': '',
                'audience_score': None
            }
    
    def build_metadata(self, content: Any, tmdb_data: Dict, omdb_data: Dict) -> Dict:
        """Build comprehensive metadata from multiple sources"""
        try:
            # Parse genres
            try:
                genres = json.loads(content.genres) if content.genres else []
            except (json.JSONDecodeError, TypeError):
                genres = []
            
            # Add TMDB genres if missing
            if not genres and tmdb_data.get('genres'):
                genres = [genre['name'] for genre in tmdb_data['genres']]
            
            metadata = {
                'genres': self._normalize_genres(genres),
                'release_info': self._build_release_info(content, tmdb_data, omdb_data),
                'technical_specs': self._build_technical_specs(content, tmdb_data, omdb_data),
                'production_info': self._build_production_info(tmdb_data, omdb_data),
                'languages': self._build_language_info(tmdb_data, omdb_data),
                'certifications': self._build_certification_info(tmdb_data, omdb_data),
                'business_info': self._build_business_info(tmdb_data, omdb_data),
                'external_ids': self._build_external_ids(tmdb_data),
                'status_flags': self._build_status_flags(content, tmdb_data),
                'metadata_sources': self._get_metadata_sources(tmdb_data, omdb_data)
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error building metadata: {e}")
            return self._get_minimal_metadata(content)
    
    def _extract_content_warnings(self, tmdb_data: Dict, omdb_data: Dict) -> List[Dict]:
        """Extract content warnings and ratings"""
        warnings = []
        
        try:
            # TMDB content ratings
            if tmdb_data.get('content_ratings'):
                ratings = tmdb_data['content_ratings'].get('results', [])
                us_rating = next((r for r in ratings if r['iso_3166_1'] == 'US'), None)
                if us_rating:
                    warnings.append({
                        'type': 'content_rating',
                        'country': 'US',
                        'rating': us_rating.get('rating'),
                        'descriptors': us_rating.get('descriptors', []),
                        'source': 'tmdb'
                    })
            
            # OMDb rated information
            if omdb_data.get('Rated') and omdb_data['Rated'] != 'N/A':
                warnings.append({
                    'type': 'mpaa_rating',
                    'country': 'US',
                    'rating': omdb_data['Rated'],
                    'source': 'omdb'
                })
            
            return warnings
            
        except Exception as e:
            logger.error(f"Error extracting content warnings: {e}")
            return []
    
    def _extract_keywords(self, tmdb_data: Dict) -> List[str]:
        """Extract keywords from TMDB data"""
        try:
            keywords = []
            
            if tmdb_data.get('keywords'):
                keyword_data = tmdb_data['keywords'].get('keywords', []) or tmdb_data['keywords'].get('results', [])
                keywords = [kw['name'] for kw in keyword_data if kw.get('name')]
            
            return keywords[:20]  # Limit keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _extract_themes(self, overview: str, plot: str, keywords: List[str]) -> List[str]:
        """Extract themes from text and keywords"""
        try:
            themes = set()
            
            # Theme keywords mapping
            theme_mappings = {
                'love': ['love', 'romance', 'relationship', 'marriage'],
                'friendship': ['friend', 'friendship', 'buddy', 'companion'],
                'family': ['family', 'father', 'mother', 'parent', 'child'],
                'revenge': ['revenge', 'vengeance', 'payback'],
                'survival': ['survival', 'survive', 'death', 'life-threatening'],
                'justice': ['justice', 'law', 'crime', 'police', 'court'],
                'power': ['power', 'control', 'domination', 'authority'],
                'freedom': ['freedom', 'liberty', 'escape', 'prison'],
                'war': ['war', 'battle', 'conflict', 'soldier', 'military'],
                'technology': ['technology', 'computer', 'robot', 'artificial intelligence']
            }
            
            # Analyze text
            text_to_analyze = f"{overview} {plot}".lower()
            
            for theme, keywords_list in theme_mappings.items():
                if any(keyword in text_to_analyze for keyword in keywords_list):
                    themes.add(theme)
            
            # Analyze provided keywords
            for keyword in keywords:
                keyword_lower = keyword.lower()
                for theme, keywords_list in theme_mappings.items():
                    if any(kw in keyword_lower for kw in keywords_list):
                        themes.add(theme)
            
            return list(themes)
            
        except Exception as e:
            logger.error(f"Error extracting themes: {e}")
            return []
    
    def _process_omdb_rating(self, rating: Dict) -> Optional[Dict]:
        """Process individual OMDb rating"""
        try:
            source = rating.get('Source', '')
            value = rating.get('Value', '')
            
            if not source or not value or value == 'N/A':
                return None
            
            processed = {
                'source': source,
                'value': value,
                'normalized_score': None,
                'scale': 'unknown'
            }
            
            # Normalize different rating scales
            if 'rotten tomatoes' in source.lower():
                match = re.search(r'(\d+)%', value)
                if match:
                    score = int(match.group(1))
                    processed['normalized_score'] = round(score / 10.0, 1)
                    processed['scale'] = '0-100%'
            
            elif 'metacritic' in source.lower():
                match = re.search(r'(\d+)', value)
                if match:
                    score = int(match.group(1))
                    processed['normalized_score'] = round(score / 10.0, 1)
                    processed['scale'] = '0-100'
            
            elif 'imdb' in source.lower():
                match = re.search(r'(\d+\.?\d*)', value)
                if match:
                    score = float(match.group(1))
                    processed['normalized_score'] = round(score, 1)
                    processed['scale'] = '0-10'
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing OMDb rating: {e}")
            return None
    
    def _calculate_composite_score(self, ratings: Dict) -> Dict:
        """Calculate composite rating score"""
        try:
            scores = []
            weights = []
            
            # TMDB score
            if ratings['tmdb']['score'] > 0:
                scores.append(ratings['tmdb']['score'])
                # Weight based on vote count
                vote_weight = min(ratings['tmdb']['count'] / 1000.0, 1.0)
                weights.append(0.4 + (vote_weight * 0.2))
            
            # IMDb score
            if ratings['imdb']['score'] > 0:
                scores.append(ratings['imdb']['score'])
                weights.append(0.5)
            
            # Other sources
            for source in ratings['omdb_sources']:
                if source.get('normalized_score') and source['normalized_score'] > 0:
                    scores.append(source['normalized_score'])
                    # Weight based on source reliability
                    if 'metacritic' in source['source'].lower():
                        weights.append(0.3)
                    elif 'rotten tomatoes' in source['source'].lower():
                        weights.append(0.25)
                    else:
                        weights.append(0.1)
            
            if not scores:
                return {
                    'composite_score': 0,
                    'confidence_level': 'none'
                }
            
            # Calculate weighted average
            if len(scores) == len(weights):
                composite = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)
            else:
                composite = sum(scores) / len(scores)
            
            # Determine confidence level
            confidence = 'high' if len(scores) >= 3 else 'medium' if len(scores) >= 2 else 'low'
            
            return {
                'composite_score': round(composite, 1),
                'confidence_level': confidence
            }
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {e}")
            return {
                'composite_score': 0,
                'confidence_level': 'error'
            }
    
    def _normalize_genres(self, genres: List[str]) -> List[str]:
        """Normalize genre names for consistency"""
        try:
            normalized = []
            for genre in genres:
                if isinstance(genre, str):
                    normalized_genre = self.genre_mappings.get(genre.lower(), genre)
                    if normalized_genre not in normalized:
                        normalized.append(normalized_genre)
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing genres: {e}")
            return genres
    
    def _build_release_info(self, content: Any, tmdb_data: Dict, omdb_data: Dict) -> Dict:
        """Build release information"""
        try:
            return {
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'year': content.release_date.year if content.release_date else None,
                'status': tmdb_data.get('status', 'Released'),
                'release_type': tmdb_data.get('release_type'),
                'dvd_release': omdb_data.get('DVD') if omdb_data.get('DVD') != 'N/A' else None
            }
        except Exception as e:
            logger.error(f"Error building release info: {e}")
            return {}
    
    def _build_technical_specs(self, content: Any, tmdb_data: Dict, omdb_data: Dict) -> Dict:
        """Build technical specifications"""
        try:
            runtime = content.runtime or tmdb_data.get('runtime')
            
            # Parse OMDb runtime if needed
            if not runtime and omdb_data.get('Runtime'):
                runtime_str = omdb_data['Runtime']
                runtime_match = re.search(r'(\d+)', runtime_str)
                if runtime_match:
                    runtime = int(runtime_match.group(1))
            
            return {
                'runtime': runtime,
                'runtime_formatted': f"{runtime} min" if runtime else None,
                'original_language': tmdb_data.get('original_language'),
                'aspect_ratio': None,  # Not available from current sources
                'sound_mix': None,     # Not available from current sources
                'color': None          # Not available from current sources
            }
        except Exception as e:
            logger.error(f"Error building technical specs: {e}")
            return {}
    
    def _build_production_info(self, tmdb_data: Dict, omdb_data: Dict) -> Dict:
        """Build production information"""
        try:
            production_info = {
                'companies': [],
                'countries': [],
                'director': '',
                'writer': '',
                'actors': '',
                'awards': ''
            }
            
            # Production companies from TMDB
            if tmdb_data.get('production_companies'):
                production_info['companies'] = [
                    {
                        'id': company['id'],
                        'name': company['name'],
                        'logo_path': company.get('logo_path')
                    }
                    for company in tmdb_data['production_companies'][:5]
                ]
            
            # Production countries from TMDB
            if tmdb_data.get('production_countries'):
                production_info['countries'] = [
                    country['name'] for country in tmdb_data['production_countries']
                ]
            
            # OMDb information
            if omdb_data:
                production_info.update({
                    'director': omdb_data.get('Director', ''),
                    'writer': omdb_data.get('Writer', ''),
                    'actors': omdb_data.get('Actors', ''),
                    'awards': omdb_data.get('Awards', '')
                })
            
            return production_info
            
        except Exception as e:
            logger.error(f"Error building production info: {e}")
            return {}
    
    def _build_language_info(self, tmdb_data: Dict, omdb_data: Dict) -> Dict:
        """Build language information"""
        try:
            languages = {
                'original': None,
                'spoken': [],
                'subtitles': []
            }
            
            # Original language
            if tmdb_data.get('original_language'):
                lang_code = tmdb_data['original_language']
                languages['original'] = {
                    'code': lang_code,
                    'name': self.language_mappings.get(lang_code, lang_code.upper())
                }
            
            # Spoken languages
            if tmdb_data.get('spoken_languages'):
                languages['spoken'] = [
                    {
                        'code': lang['iso_639_1'],
                        'name': lang['english_name'] or lang['name']
                    }
                    for lang in tmdb_data['spoken_languages']
                ]
            
            return languages
            
        except Exception as e:
            logger.error(f"Error building language info: {e}")
            return {}
    
    def _build_certification_info(self, tmdb_data: Dict, omdb_data: Dict) -> Dict:
        """Build certification/rating information"""
        try:
            certifications = {}
            
            # TMDB certifications
            if tmdb_data.get('release_dates'):
                for country_data in tmdb_data['release_dates'].get('results', []):
                    country_code = country_data['iso_3166_1']
                    for release in country_data['release_dates']:
                        if release.get('certification'):
                            certifications[country_code] = release['certification']
                            break
            
            # OMDb rating
            if omdb_data.get('Rated') and omdb_data['Rated'] != 'N/A':
                certifications['US_MPAA'] = omdb_data['Rated']
            
            return certifications
            
        except Exception as e:
            logger.error(f"Error building certification info: {e}")
            return {}
    
    def _build_business_info(self, tmdb_data: Dict, omdb_data: Dict) -> Dict:
        """Build business/box office information"""
        try:
            business = {
                'budget': tmdb_data.get('budget', 0),
                'revenue': tmdb_data.get('revenue', 0),
                'box_office': omdb_data.get('BoxOffice') if omdb_data.get('BoxOffice') != 'N/A' else None,
                'profit': None,
                'roi': None
            }
            
            # Calculate profit and ROI if we have budget and revenue
            if business['budget'] and business['revenue']:
                business['profit'] = business['revenue'] - business['budget']
                if business['budget'] > 0:
                    business['roi'] = round((business['profit'] / business['budget']) * 100, 1)
            
            return business
            
        except Exception as e:
            logger.error(f"Error building business info: {e}")
            return {}
    
    def _build_external_ids(self, tmdb_data: Dict) -> Dict:
        """Build external IDs information"""
        try:
            external_ids = {}
            
            if tmdb_data.get('external_ids'):
                ids = tmdb_data['external_ids']
                
                # Map common external IDs
                id_mappings = {
                    'imdb_id': 'imdb',
                    'wikidata_id': 'wikidata',
                    'facebook_id': 'facebook',
                    'instagram_id': 'instagram',
                    'twitter_id': 'twitter'
                }
                
                for tmdb_key, our_key in id_mappings.items():
                    if ids.get(tmdb_key):
                        external_ids[our_key] = ids[tmdb_key]
            
            return external_ids
            
        except Exception as e:
            logger.error(f"Error building external IDs: {e}")
            return {}
    
    def _build_status_flags(self, content: Any, tmdb_data: Dict) -> Dict:
        """Build status flags"""
        try:
            return {
                'is_trending': getattr(content, 'is_trending', False),
                'is_new_release': getattr(content, 'is_new_release', False),
                'is_critics_choice': getattr(content, 'is_critics_choice', False),
                'is_adult': tmdb_data.get('adult', False),
                'is_video': tmdb_data.get('video', False),
                'popularity': content.popularity or tmdb_data.get('popularity', 0)
            }
        except Exception as e:
            logger.error(f"Error building status flags: {e}")
            return {}
    
    def _get_metadata_sources(self, tmdb_data: Dict, omdb_data: Dict) -> List[str]:
        """Get list of metadata sources used"""
        sources = ['database']
        
        if tmdb_data:
            sources.append('tmdb')
        
        if omdb_data and omdb_data.get('Response') == 'True':
            sources.append('omdb')
        
        return sources
    
    def _get_minimal_metadata(self, content: Any) -> Dict:
        """Get minimal metadata as fallback"""
        try:
            return {
                'genres': [],
                'release_info': {
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'year': content.release_date.year if content.release_date else None,
                    'status': 'Released'
                },
                'technical_specs': {
                    'runtime': content.runtime,
                    'runtime_formatted': f"{content.runtime} min" if content.runtime else None
                },
                'production_info': {},
                'languages': {},
                'certifications': {},
                'business_info': {},
                'external_ids': {},
                'status_flags': {
                    'is_trending': getattr(content, 'is_trending', False),
                    'is_new_release': getattr(content, 'is_new_release', False),
                    'is_critics_choice': getattr(content, 'is_critics_choice', False),
                    'popularity': content.popularity or 0
                },
                'metadata_sources': ['database']
            }
        except Exception as e:
            logger.error(f"Error creating minimal metadata: {e}")
            return {}