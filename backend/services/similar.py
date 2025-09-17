import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import and_, or_, func, desc
from sqlalchemy.orm import sessionmaker

# Try to import sentence-transformers, fallback if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SimilarContentService:
    """
    Advanced content similarity service focused on story content and language matching.
    
    Features:
    - Semantic similarity using sentence embeddings (BERT-based)
    - TF-IDF cosine similarity as fallback
    - Language-exact matching with priority scoring
    - Genre and theme alignment
    - Weighted multi-factor scoring
    - Configurable thresholds and ranking
    """
    
    def __init__(self, db, content_model, cache=None, config=None):
        """
        Initialize the similarity service.
        
        Args:
            db: SQLAlchemy database session
            content_model: Content SQLAlchemy model
            cache: Cache instance (optional)
            config: Configuration dictionary (optional)
        """
        self.db = db
        self.Content = content_model
        self.cache = cache
        
        # Default configuration
        self.config = {
            'embedding_model': 'all-MiniLM-L6-v2',  # Lightweight but effective
            'similarity_threshold': 0.3,
            'max_candidates': 1000,  # Limit initial candidate pool
            'weights': {
                'story_similarity': 0.4,      # Primary focus on story
                'language_match': 0.3,        # Strong language priority
                'genre_alignment': 0.2,       # Genre compatibility
                'title_similarity': 0.1       # Minor title consideration
            },
            'language_boost': 0.2,            # Extra boost for exact language match
            'genre_boost': 0.1,               # Extra boost for genre match
            'cache_timeout': 1800,            # 30 minutes
            'min_overview_length': 10,        # Minimum overview length for processing
            'tfidf_max_features': 5000,       # TF-IDF feature limit
            'embedding_batch_size': 32        # Batch size for embeddings
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Initialize models
        self._sentence_model = None
        self._tfidf_vectorizer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models with error handling."""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self._sentence_model = SentenceTransformer(self.config['embedding_model'])
                logger.info(f"Initialized sentence transformer: {self.config['embedding_model']}")
            else:
                logger.warning("Sentence transformers not available, using TF-IDF only")
            
            # Initialize TF-IDF as backup
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config['tfidf_max_features'],
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
            self._sentence_model = None
    
    def get_similar(self, content_id: int, limit: int = 10, 
                   strict_language_match: bool = True,
                   min_similarity: float = None,
                   include_metadata: bool = True) -> Dict[str, Any]:
        """
        Get similar content for a given content ID.
        
        Args:
            content_id: ID of the base content
            limit: Maximum number of similar items to return
            strict_language_match: Whether to enforce exact language matching
            min_similarity: Minimum similarity threshold (overrides config)
            include_metadata: Whether to include detailed metadata
        
        Returns:
            Dictionary with similar content and metadata
        """
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._generate_cache_key(
                content_id, limit, strict_language_match, min_similarity
            )
            
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for similar content {content_id}")
                    return cached_result
            
            # Get base content
            base_content = self._get_content_by_id(content_id)
            if not base_content:
                return self._create_error_response("Content not found", content_id)
            
            # Validate base content has required fields
            if not self._validate_content_for_similarity(base_content):
                return self._create_error_response(
                    "Content lacks required fields for similarity analysis", 
                    content_id
                )
            
            # Get candidate pool
            candidates = self._get_candidate_pool(base_content, strict_language_match)
            if not candidates:
                return self._create_empty_response(base_content, "No suitable candidates found")
            
            # Calculate similarities
            similarities = self._calculate_similarities(base_content, candidates)
            
            # Apply threshold
            threshold = min_similarity or self.config['similarity_threshold']
            filtered_similarities = [
                sim for sim in similarities 
                if sim['final_score'] >= threshold
            ]
            
            # Sort and limit results
            sorted_similarities = sorted(
                filtered_similarities, 
                key=lambda x: x['final_score'], 
                reverse=True
            )[:limit]
            
            # Format results
            similar_content = self._format_similar_content(sorted_similarities)
            
            # Create response
            response = {
                'base_content': self._format_base_content(base_content),
                'similar_content': similar_content,
                'metadata': {
                    'algorithm': 'advanced_semantic_similarity',
                    'total_candidates_analyzed': len(candidates),
                    'results_returned': len(similar_content),
                    'similarity_threshold': threshold,
                    'strict_language_match': strict_language_match,
                    'processing_time_ms': round((time.time() - start_time) * 1000, 2),
                    'model_used': self._get_model_info(),
                    'weights_applied': self.config['weights'],
                    'timestamp': datetime.utcnow().isoformat()
                } if include_metadata else {}
            }
            
            # Cache the result
            if self.cache:
                try:
                    self.cache.set(cache_key, response, timeout=self.config['cache_timeout'])
                except Exception as e:
                    logger.warning(f"Failed to cache result: {e}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in get_similar for content {content_id}: {e}")
            return self._create_error_response(f"Similarity calculation failed: {str(e)}", content_id)
    
    def _get_content_by_id(self, content_id: int):
        """Get content by ID with error handling."""
        try:
            return self.Content.query.get(content_id)
        except Exception as e:
            logger.error(f"Database error getting content {content_id}: {e}")
            return None
    
    def _validate_content_for_similarity(self, content) -> bool:
        """Validate that content has required fields for similarity analysis."""
        if not content:
            return False
        
        # Check for overview (primary field for story similarity)
        if not content.overview or len(content.overview.strip()) < self.config['min_overview_length']:
            logger.warning(f"Content {content.id} has insufficient overview for similarity")
            return False
        
        # Check for title
        if not content.title:
            logger.warning(f"Content {content.id} missing title")
            return False
        
        return True
    
    def _get_candidate_pool(self, base_content, strict_language_match: bool) -> List:
        """
        Get pool of candidate content for similarity comparison.
        
        Args:
            base_content: Base content object
            strict_language_match: Whether to enforce language matching
        
        Returns:
            List of candidate content objects
        """
        try:
            query = self.Content.query.filter(
                self.Content.id != base_content.id,
                self.Content.overview.isnot(None),
                func.length(self.Content.overview) >= self.config['min_overview_length']
            )
            
            # Language filtering
            if strict_language_match and base_content.languages:
                try:
                    base_languages = json.loads(base_content.languages) if isinstance(base_content.languages, str) else base_content.languages
                    if base_languages:
                        # Create language filter conditions
                        language_conditions = []
                        for lang in base_languages:
                            if isinstance(lang, str) and lang.strip():
                                language_conditions.append(
                                    self.Content.languages.contains(lang.strip().lower())
                                )
                        
                        if language_conditions:
                            query = query.filter(or_(*language_conditions))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Error parsing languages for content {base_content.id}: {e}")
            
            # Content type matching (optional enhancement)
            if hasattr(base_content, 'content_type') and base_content.content_type:
                query = query.filter(self.Content.content_type == base_content.content_type)
            
            # Limit candidates for performance
            candidates = query.limit(self.config['max_candidates']).all()
            
            logger.info(f"Found {len(candidates)} candidates for content {base_content.id}")
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting candidate pool: {e}")
            return []
    
    def _calculate_similarities(self, base_content, candidates: List) -> List[Dict]:
        """
        Calculate comprehensive similarity scores for all candidates.
        
        Args:
            base_content: Base content object
            candidates: List of candidate content objects
        
        Returns:
            List of similarity dictionaries with scores and metadata
        """
        similarities = []
        
        try:
            # Prepare text data
            base_text = self._prepare_text_for_similarity(base_content)
            candidate_texts = [self._prepare_text_for_similarity(candidate) for candidate in candidates]
            
            # Calculate story similarity using best available method
            story_similarities = self._calculate_story_similarities(base_text, candidate_texts)
            
            # Calculate other similarity factors
            for i, candidate in enumerate(candidates):
                try:
                    # Get individual similarity scores
                    story_score = story_similarities[i] if i < len(story_similarities) else 0.0
                    language_score = self._calculate_language_similarity(base_content, candidate)
                    genre_score = self._calculate_genre_similarity(base_content, candidate)
                    title_score = self._calculate_title_similarity(base_content, candidate)
                    
                    # Calculate weighted final score
                    final_score = (
                        story_score * self.config['weights']['story_similarity'] +
                        language_score * self.config['weights']['language_match'] +
                        genre_score * self.config['weights']['genre_alignment'] +
                        title_score * self.config['weights']['title_similarity']
                    )
                    
                    # Apply boosts for exact matches
                    if language_score > 0.8:  # High language match
                        final_score += self.config['language_boost']
                    
                    if genre_score > 0.6:  # Good genre match
                        final_score += self.config['genre_boost']
                    
                    # Cap final score at 1.0
                    final_score = min(final_score, 1.0)
                    
                    # Determine match reason
                    match_reason = self._determine_match_reason(
                        story_score, language_score, genre_score, title_score
                    )
                    
                    similarities.append({
                        'content': candidate,
                        'story_similarity': round(story_score, 4),
                        'language_similarity': round(language_score, 4),
                        'genre_similarity': round(genre_score, 4),
                        'title_similarity': round(title_score, 4),
                        'final_score': round(final_score, 4),
                        'match_reason': match_reason,
                        'match_type': self._classify_match_type(final_score, match_reason)
                    })
                    
                except Exception as e:
                    logger.warning(f"Error calculating similarity for candidate {candidate.id}: {e}")
                    continue
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error in similarity calculation: {e}")
            return []
    
    def _prepare_text_for_similarity(self, content) -> str:
        """
        Prepare text content for similarity analysis.
        
        Args:
            content: Content object
        
        Returns:
            Cleaned and prepared text string
        """
        try:
            # Combine overview and title for richer context
            text_parts = []
            
            if content.overview:
                text_parts.append(content.overview.strip())
            
            if content.title:
                text_parts.append(content.title.strip())
            
            # Add genre information as context
            if hasattr(content, 'genres') and content.genres:
                try:
                    genres = json.loads(content.genres) if isinstance(content.genres, str) else content.genres
                    if genres and isinstance(genres, list):
                        text_parts.append(' '.join(genres))
                except (json.JSONDecodeError, TypeError):
                    pass
            
            combined_text = ' '.join(text_parts)
            
            # Clean the text
            cleaned_text = self._clean_text(combined_text)
            
            return cleaned_text
            
        except Exception as e:
            logger.warning(f"Error preparing text for content {content.id}: {e}")
            return content.overview or content.title or ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better similarity matching."""
        if not text:
            return ""
        
        try:
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep basic punctuation
            text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
            
            # Convert to lowercase
            text = text.lower().strip()
            
            return text
            
        except Exception as e:
            logger.warning(f"Error cleaning text: {e}")
            return text
    
    def _calculate_story_similarities(self, base_text: str, candidate_texts: List[str]) -> List[float]:
        """
        Calculate story similarity using the best available method.
        
        Args:
            base_text: Base content text
            candidate_texts: List of candidate texts
        
        Returns:
            List of similarity scores
        """
        try:
            # Try sentence embeddings first (most accurate)
            if self._sentence_model and SENTENCE_TRANSFORMERS_AVAILABLE:
                return self._calculate_embedding_similarities(base_text, candidate_texts)
            
            # Fallback to TF-IDF
            return self._calculate_tfidf_similarities(base_text, candidate_texts)
            
        except Exception as e:
            logger.error(f"Error calculating story similarities: {e}")
            return [0.0] * len(candidate_texts)
    
    def _calculate_embedding_similarities(self, base_text: str, candidate_texts: List[str]) -> List[float]:
        """Calculate similarities using sentence embeddings."""
        try:
            if not base_text.strip():
                return [0.0] * len(candidate_texts)
            
            # Filter out empty texts
            valid_texts = [text for text in candidate_texts if text.strip()]
            if not valid_texts:
                return [0.0] * len(candidate_texts)
            
            # Generate embeddings
            all_texts = [base_text] + valid_texts
            embeddings = self._sentence_model.encode(
                all_texts, 
                batch_size=self.config['embedding_batch_size'],
                show_progress_bar=False
            )
            
            # Calculate cosine similarities
            base_embedding = embeddings[0].reshape(1, -1)
            candidate_embeddings = embeddings[1:]
            
            similarities = cosine_similarity(base_embedding, candidate_embeddings)[0]
            
            # Handle the case where some texts were filtered out
            result_similarities = []
            valid_idx = 0
            
            for text in candidate_texts:
                if text.strip():
                    result_similarities.append(float(similarities[valid_idx]))
                    valid_idx += 1
                else:
                    result_similarities.append(0.0)
            
            return result_similarities
            
        except Exception as e:
            logger.error(f"Error in embedding similarity calculation: {e}")
            return [0.0] * len(candidate_texts)
    
    def _calculate_tfidf_similarities(self, base_text: str, candidate_texts: List[str]) -> List[float]:
        """Calculate similarities using TF-IDF."""
        try:
            if not base_text.strip():
                return [0.0] * len(candidate_texts)
            
            # Filter out empty texts
            valid_texts = [text for text in candidate_texts if text.strip()]
            if not valid_texts:
                return [0.0] * len(candidate_texts)
            
            # Prepare all texts
            all_texts = [base_text] + valid_texts
            
            # Fit TF-IDF and transform
            tfidf_matrix = self._tfidf_vectorizer.fit_transform(all_texts)
            
            # Calculate similarities
            base_vector = tfidf_matrix[0]
            candidate_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(base_vector, candidate_vectors)[0]
            
            # Handle filtered texts
            result_similarities = []
            valid_idx = 0
            
            for text in candidate_texts:
                if text.strip():
                    result_similarities.append(float(similarities[valid_idx]))
                    valid_idx += 1
                else:
                    result_similarities.append(0.0)
            
            return result_similarities
            
        except Exception as e:
            logger.error(f"Error in TF-IDF similarity calculation: {e}")
            return [0.0] * len(candidate_texts)
    
    def _calculate_language_similarity(self, base_content, candidate) -> float:
        """Calculate language similarity score."""
        try:
            # Parse languages safely
            base_languages = self._parse_languages(base_content.languages)
            candidate_languages = self._parse_languages(candidate.languages)
            
            if not base_languages or not candidate_languages:
                return 0.0
            
            # Convert to sets for intersection
            base_set = set(lang.lower().strip() for lang in base_languages if lang)
            candidate_set = set(lang.lower().strip() for lang in candidate_languages if lang)
            
            if not base_set or not candidate_set:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = base_set.intersection(candidate_set)
            union = base_set.union(candidate_set)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating language similarity: {e}")
            return 0.0
    
    def _calculate_genre_similarity(self, base_content, candidate) -> float:
        """Calculate genre similarity score."""
        try:
            # Parse genres safely
            base_genres = self._parse_genres(base_content.genres)
            candidate_genres = self._parse_genres(candidate.genres)
            
            if not base_genres or not candidate_genres:
                return 0.0
            
            # Convert to sets for intersection
            base_set = set(genre.lower().strip() for genre in base_genres if genre)
            candidate_set = set(genre.lower().strip() for genre in candidate_genres if genre)
            
            if not base_set or not candidate_set:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = base_set.intersection(candidate_set)
            union = base_set.union(candidate_set)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating genre similarity: {e}")
            return 0.0
    
    def _calculate_title_similarity(self, base_content, candidate) -> float:
        """Calculate title similarity using simple string matching."""
        try:
            if not base_content.title or not candidate.title:
                return 0.0
            
            base_title = self._clean_text(base_content.title)
            candidate_title = self._clean_text(candidate.title)
            
            if not base_title or not candidate_title:
                return 0.0
            
            # Simple word overlap calculation
            base_words = set(base_title.split())
            candidate_words = set(candidate_title.split())
            
            if not base_words or not candidate_words:
                return 0.0
            
            intersection = base_words.intersection(candidate_words)
            union = base_words.union(candidate_words)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating title similarity: {e}")
            return 0.0
    
    def _parse_languages(self, languages_field) -> List[str]:
        """Safely parse languages field."""
        try:
            if not languages_field:
                return []
            
            if isinstance(languages_field, str):
                return json.loads(languages_field)
            elif isinstance(languages_field, list):
                return languages_field
            else:
                return []
                
        except (json.JSONDecodeError, TypeError):
            return []
    
    def _parse_genres(self, genres_field) -> List[str]:
        """Safely parse genres field."""
        try:
            if not genres_field:
                return []
            
            if isinstance(genres_field, str):
                return json.loads(genres_field)
            elif isinstance(genres_field, list):
                return genres_field
            else:
                return []
                
        except (json.JSONDecodeError, TypeError):
            return []
    
    def _determine_match_reason(self, story_score: float, language_score: float, 
                              genre_score: float, title_score: float) -> str:
        """Determine the primary reason for the match."""
        scores = {
            'story_content': story_score,
            'language_match': language_score,
            'genre_alignment': genre_score,
            'title_similarity': title_score
        }
        
        # Find the highest scoring factor
        max_factor = max(scores.keys(), key=lambda k: scores[k])
        max_score = scores[max_factor]
        
        # Create detailed reason based on score combinations
        reasons = []
        
        if story_score >= 0.6:
            reasons.append("similar story content")
        if language_score >= 0.8:
            reasons.append("exact language match")
        if genre_score >= 0.5:
            reasons.append("shared genres")
        if title_score >= 0.3:
            reasons.append("title similarity")
        
        if not reasons:
            return f"low similarity match ({max_factor.replace('_', ' ')})"
        
        return ", ".join(reasons)
    
    def _classify_match_type(self, final_score: float, match_reason: str) -> str:
        """Classify the type of match based on score and reason."""
        if final_score >= 0.8:
            return "excellent_match"
        elif final_score >= 0.6:
            return "good_match"
        elif final_score >= 0.4:
            return "moderate_match"
        else:
            return "weak_match"
    
    def _format_similar_content(self, similarities: List[Dict]) -> List[Dict]:
        """Format similarity results for API response."""
        formatted_results = []
        
        for similarity in similarities:
            content = similarity['content']
            
            try:
                # Ensure content has slug (try to generate if missing)
                slug = getattr(content, 'slug', None)
                if not slug:
                    slug = f"content-{content.id}"
                
                # Get YouTube trailer URL if available
                youtube_url = None
                if hasattr(content, 'youtube_trailer_id') and content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                # Format poster path
                poster_path = None
                if hasattr(content, 'poster_path') and content.poster_path:
                    if content.poster_path.startswith('http'):
                        poster_path = content.poster_path
                    else:
                        poster_path = f"https://image.tmdb.org/t/p/w300{content.poster_path}"
                
                formatted_content = {
                    'id': content.id,
                    'slug': slug,
                    'title': content.title,
                    'overview': content.overview[:200] + '...' if content.overview and len(content.overview) > 200 else content.overview,
                    'poster_path': poster_path,
                    'rating': getattr(content, 'rating', None),
                    'content_type': getattr(content, 'content_type', 'unknown'),
                    'release_date': content.release_date.isoformat() if hasattr(content, 'release_date') and content.release_date else None,
                    'genres': self._parse_genres(getattr(content, 'genres', None)),
                    'languages': self._parse_languages(getattr(content, 'languages', None)),
                    'youtube_trailer': youtube_url,
                    'similarity_score': similarity['final_score'],
                    'similarity_breakdown': {
                        'story_similarity': similarity['story_similarity'],
                        'language_similarity': similarity['language_similarity'],
                        'genre_similarity': similarity['genre_similarity'],
                        'title_similarity': similarity['title_similarity']
                    },
                    'match_reason': similarity['match_reason'],
                    'match_type': similarity['match_type']
                }
                
                formatted_results.append(formatted_content)
                
            except Exception as e:
                logger.warning(f"Error formatting content {content.id}: {e}")
                continue
        
        return formatted_results
    
    def _format_base_content(self, base_content) -> Dict:
        """Format base content for API response."""
        try:
            slug = getattr(base_content, 'slug', None)
            if not slug:
                slug = f"content-{base_content.id}"
            
            return {
                'id': base_content.id,
                'slug': slug,
                'title': base_content.title,
                'content_type': getattr(base_content, 'content_type', 'unknown'),
                'rating': getattr(base_content, 'rating', None),
                'genres': self._parse_genres(getattr(base_content, 'genres', None)),
                'languages': self._parse_languages(getattr(base_content, 'languages', None))
            }
        except Exception as e:
            logger.error(f"Error formatting base content: {e}")
            return {'id': base_content.id, 'title': getattr(base_content, 'title', 'Unknown')}
    
    def _get_model_info(self) -> Dict:
        """Get information about the models being used."""
        return {
            'primary_method': 'sentence_embeddings' if self._sentence_model else 'tfidf',
            'embedding_model': self.config['embedding_model'] if self._sentence_model else None,
            'fallback_method': 'tfidf_cosine_similarity',
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE
        }
    
    def _generate_cache_key(self, content_id: int, limit: int, 
                           strict_language: bool, min_similarity: Optional[float]) -> str:
        """Generate cache key for similarity results."""
        return f"similar:{content_id}:{limit}:{strict_language}:{min_similarity or 'default'}"
    
    def _create_error_response(self, error_message: str, content_id: int) -> Dict:
        """Create standardized error response."""
        return {
            'base_content': {'id': content_id},
            'similar_content': [],
            'metadata': {
                'error': error_message,
                'algorithm': 'advanced_semantic_similarity',
                'timestamp': datetime.utcnow().isoformat()
            }
        }
    
    def _create_empty_response(self, base_content, reason: str) -> Dict:
        """Create response for when no similar content is found."""
        return {
            'base_content': self._format_base_content(base_content),
            'similar_content': [],
            'metadata': {
                'algorithm': 'advanced_semantic_similarity',
                'reason': reason,
                'total_candidates_analyzed': 0,
                'results_returned': 0,
                'timestamp': datetime.utcnow().isoformat()
            }
        }


# Factory function for easy initialization
def create_similar_content_service(db, content_model, cache=None, config=None):
    """
    Factory function to create a SimilarContentService instance.
    
    Args:
        db: SQLAlchemy database session
        content_model: Content SQLAlchemy model
        cache: Cache instance (optional)
        config: Configuration dictionary (optional)
    
    Returns:
        SimilarContentService instance
    """
    return SimilarContentService(db, content_model, cache, config)


# Configuration presets for different use cases
SIMILARITY_CONFIGS = {
    'high_accuracy': {
        'similarity_threshold': 0.5,
        'weights': {
            'story_similarity': 0.5,
            'language_match': 0.3,
            'genre_alignment': 0.15,
            'title_similarity': 0.05
        },
        'language_boost': 0.25,
        'max_candidates': 500
    },
    'fast_performance': {
        'similarity_threshold': 0.3,
        'weights': {
            'story_similarity': 0.4,
            'language_match': 0.3,
            'genre_alignment': 0.2,
            'title_similarity': 0.1
        },
        'max_candidates': 200,
        'embedding_batch_size': 16
    },
    'language_priority': {
        'similarity_threshold': 0.4,
        'weights': {
            'story_similarity': 0.3,
            'language_match': 0.4,
            'genre_alignment': 0.2,
            'title_similarity': 0.1
        },
        'language_boost': 0.3,
        'genre_boost': 0.05
    }
}