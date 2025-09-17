# backend/services/similar.py

import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re
import hashlib
from datetime import datetime, timedelta

# Core libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Advanced NLP libraries (with fallbacks)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class SimilarityResult:
    """Data class for similarity results"""
    content_id: int
    content: Any  # Content object
    similarity_score: float
    match_type: str
    confidence: float
    details: Dict[str, Any]

class TextPreprocessor:
    """Advanced text preprocessing for similarity analysis"""
    
    def __init__(self):
        self.lemmatizer = None
        self.stop_words = set()
        self._initialize_nltk()
    
    def _initialize_nltk(self):
        """Initialize NLTK components with error handling"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
            # Add domain-specific stop words
            self.stop_words.update({
                'movie', 'film', 'story', 'tells', 'follows', 'shows',
                'series', 'episode', 'season', 'anime', 'manga'
            })
            
        except Exception as e:
            logger.warning(f"NLTK initialization failed: {e}")
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text for similarity analysis"""
        if not text:
            return ""
        
        # Remove special characters and normalize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stop words
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            if (len(token) > 2 and 
                token not in self.stop_words and 
                token.isalpha()):
                
                # Lemmatize if available
                if self.lemmatizer:
                    try:
                        token = self.lemmatizer.lemmatize(token)
                    except:
                        pass
                
                filtered_tokens.append(token)
        
        return ' '.join(filtered_tokens)
    
    def extract_plot_keywords(self, text: str) -> List[str]:
        """Extract key plot elements from text"""
        if not text:
            return []
        
        # Define plot-relevant patterns
        plot_patterns = [
            r'\b(murder|kill|death|die|dead)\b',
            r'\b(love|romance|relationship|marry|wedding)\b',
            r'\b(war|fight|battle|conflict|enemy)\b',
            r'\b(family|father|mother|son|daughter|brother|sister)\b',
            r'\b(secret|mystery|hidden|discover|reveal)\b',
            r'\b(journey|travel|adventure|quest|search)\b',
            r'\b(power|magic|supernatural|fantasy|sci-fi)\b',
            r'\b(school|student|teacher|education|college)\b',
            r'\b(crime|police|detective|investigation|law)\b',
            r'\b(revenge|betrayal|conspiracy|plot)\b'
        ]
        
        keywords = []
        text_lower = text.lower()
        
        for pattern in plot_patterns:
            matches = re.findall(pattern, text_lower)
            keywords.extend(matches)
        
        return list(set(keywords))

class LanguageMatcher:
    """Precise language matching for content"""
    
    LANGUAGE_CODES = {
        'english': ['en', 'english', 'eng'],
        'hindi': ['hi', 'hindi', 'hin'],
        'telugu': ['te', 'telugu', 'tel'],
        'tamil': ['ta', 'tamil', 'tam'],
        'kannada': ['kn', 'kannada', 'kan'],
        'malayalam': ['ml', 'malayalam', 'mal'],
        'japanese': ['ja', 'japanese', 'jpn'],
        'korean': ['ko', 'korean', 'kor'],
        'chinese': ['zh', 'chinese', 'chi', 'mandarin'],
        'spanish': ['es', 'spanish', 'spa'],
        'french': ['fr', 'french', 'fra'],
        'german': ['de', 'german', 'deu']
    }
    
    @classmethod
    def normalize_languages(cls, languages: List[str]) -> List[str]:
        """Normalize language list to standard codes"""
        if not languages:
            return ['en']  # Default to English
        
        normalized = []
        for lang in languages:
            lang_lower = lang.lower().strip()
            
            # Find matching language group
            for standard_name, variants in cls.LANGUAGE_CODES.items():
                if lang_lower in variants:
                    normalized.append(standard_name)
                    break
            else:
                # If not found, keep original
                normalized.append(lang_lower)
        
        return list(set(normalized))
    
    @classmethod
    def languages_match(cls, lang1: List[str], lang2: List[str]) -> bool:
        """Check if two language lists have any overlap"""
        norm1 = set(cls.normalize_languages(lang1))
        norm2 = set(cls.normalize_languages(lang2))
        return bool(norm1.intersection(norm2))

class StoryEmbeddingEngine:
    """Advanced story similarity using embeddings"""
    
    def __init__(self, cache=None):
        self.cache = cache
        self.model = None
        self.spacy_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding models with fallbacks"""
        # Try to load Sentence Transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence Transformers model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Sentence Transformers: {e}")
        
        # Try to load spaCy model
        if SPACY_AVAILABLE:
            try:
                self.spacy_model = spacy.load('en_core_web_sm')
                logger.info("spaCy model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get text embedding using best available method"""
        if not text:
            return None
        
        cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
        
        # Check cache first
        if self.cache:
            try:
                cached_embedding = self.cache.get(cache_key)
                if cached_embedding is not None:
                    return np.array(cached_embedding)
            except Exception as e:
                logger.warning(f"Cache get error: {e}")
        
        embedding = None
        
        # Method 1: Sentence Transformers (best)
        if self.model is not None:
            try:
                embedding = self.model.encode([text])[0]
                logger.debug("Used Sentence Transformers for embedding")
            except Exception as e:
                logger.warning(f"Sentence Transformers encoding failed: {e}")
        
        # Method 2: spaCy (good fallback)
        elif self.spacy_model is not None:
            try:
                doc = self.spacy_model(text[:1000000])  # Limit text length
                if doc.vector.any():
                    embedding = doc.vector
                    logger.debug("Used spaCy for embedding")
            except Exception as e:
                logger.warning(f"spaCy encoding failed: {e}")
        
        # Cache the result
        if embedding is not None and self.cache:
            try:
                self.cache.set(cache_key, embedding.tolist(), timeout=86400)  # 24 hours
            except Exception as e:
                logger.warning(f"Cache set error: {e}")
        
        return embedding
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            emb1 = self.get_embedding(text1)
            emb2 = self.get_embedding(text2)
            
            if emb1 is not None and emb2 is not None:
                # Cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                return float(similarity)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Embedding similarity calculation failed: {e}")
            return 0.0

class TitleAccuracyValidator:
    """Validate title accuracy and relevance"""
    
    @staticmethod
    def title_similarity(title1: str, title2: str) -> float:
        """Calculate title similarity using fuzzy matching"""
        if not title1 or not title2:
            return 0.0
        
        # Normalize titles
        norm1 = re.sub(r'[^\w\s]', '', title1.lower()).strip()
        norm2 = re.sub(r'[^\w\s]', '', title2.lower()).strip()
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Sequence matcher
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    @staticmethod
    def is_title_relevant(content_title: str, candidate_title: str, min_relevance: float = 0.3) -> bool:
        """Check if candidate title is relevant to content"""
        similarity = TitleAccuracyValidator.title_similarity(content_title, candidate_title)
        
        # Allow different titles but flag if too similar (possible duplicates)
        if similarity > 0.9:
            logger.warning(f"Possible duplicate titles: '{content_title}' vs '{candidate_title}'")
            return False
        
        # Titles should be different enough to be meaningful recommendations
        return True

class SimilarTitlesEngine:
    """Main engine for generating similar titles recommendations"""
    
    def __init__(self, db, content_model, cache=None):
        self.db = db
        self.Content = content_model
        self.cache = cache
        
        # Initialize components
        self.text_processor = TextPreprocessor()
        self.embedding_engine = StoryEmbeddingEngine(cache)
        self.language_matcher = LanguageMatcher()
        self.title_validator = TitleAccuracyValidator()
        
        # TF-IDF fallback
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        logger.info("SimilarTitlesEngine initialized successfully")
    
    def _get_cache_key(self, content_id: int, limit: int, min_similarity: float) -> str:
        """Generate cache key for similarity results"""
        return f"similar_titles:{content_id}:{limit}:{min_similarity}"
    
    def _get_candidate_content(self, base_content, limit_candidates: int = 500) -> List[Any]:
        """Get candidate content for similarity comparison"""
        try:
            # Parse base content languages
            base_languages = []
            if base_content.languages:
                try:
                    base_languages = json.loads(base_content.languages)
                except (json.JSONDecodeError, TypeError):
                    base_languages = [base_content.languages] if base_content.languages else []
            
            # Parse base content genres for pre-filtering
            base_genres = []
            if base_content.genres:
                try:
                    base_genres = json.loads(base_content.genres)
                except (json.JSONDecodeError, TypeError):
                    base_genres = []
            
            # Build query for candidates
            query = self.Content.query.filter(
                self.Content.id != base_content.id,
                self.Content.content_type == base_content.content_type,
                self.Content.overview.isnot(None),
                self.Content.overview != ''
            )
            
            # Language filtering
            if base_languages:
                language_filters = []
                for lang in base_languages:
                    language_filters.append(self.Content.languages.contains(lang))
                
                if language_filters:
                    query = query.filter(self.db.or_(*language_filters))
            
            # Pre-filter by genre overlap (performance optimization)
            if base_genres:
                genre_filters = []
                for genre in base_genres[:3]:  # Use top 3 genres
                    genre_filters.append(self.Content.genres.contains(genre))
                
                if genre_filters:
                    # Get genre matches first, then others
                    genre_matches = query.filter(self.db.or_(*genre_filters)).order_by(
                        self.Content.rating.desc()
                    ).limit(limit_candidates // 2).all()
                    
                    other_matches = query.filter(
                        ~self.db.or_(*genre_filters)
                    ).order_by(
                        self.Content.popularity.desc()
                    ).limit(limit_candidates // 2).all()
                    
                    return genre_matches + other_matches
            
            # Default: order by rating and popularity
            return query.order_by(
                self.Content.rating.desc(),
                self.Content.popularity.desc()
            ).limit(limit_candidates).all()
            
        except Exception as e:
            logger.error(f"Error getting candidate content: {e}")
            return []
    
    def _calculate_story_similarity(self, base_overview: str, candidate_overview: str) -> Tuple[float, str]:
        """Calculate story similarity using multiple methods"""
        if not base_overview or not candidate_overview:
            return 0.0, "no_overview"
        
        similarities = []
        method_used = "fallback"
        
        # Method 1: Advanced embeddings (primary)
        embedding_sim = self.embedding_engine.calculate_similarity(base_overview, candidate_overview)
        if embedding_sim > 0:
            similarities.append(embedding_sim)
            method_used = "embeddings"
        
        # Method 2: TF-IDF fallback
        try:
            # Clean texts
            clean_base = self.text_processor.clean_text(base_overview)
            clean_candidate = self.text_processor.clean_text(candidate_overview)
            
            if clean_base and clean_candidate:
                # Fit TF-IDF on both texts
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([clean_base, clean_candidate])
                tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                similarities.append(tfidf_sim)
                
                if method_used == "fallback":
                    method_used = "tfidf"
                    
        except Exception as e:
            logger.warning(f"TF-IDF similarity calculation failed: {e}")
        
        # Method 3: Plot keywords similarity
        try:
            base_keywords = set(self.text_processor.extract_plot_keywords(base_overview))
            candidate_keywords = set(self.text_processor.extract_plot_keywords(candidate_overview))
            
            if base_keywords and candidate_keywords:
                keyword_sim = len(base_keywords.intersection(candidate_keywords)) / len(base_keywords.union(candidate_keywords))
                similarities.append(keyword_sim * 0.7)  # Weight down keyword similarity
                
        except Exception as e:
            logger.warning(f"Keyword similarity calculation failed: {e}")
        
        # Combine similarities with weights
        if similarities:
            if len(similarities) == 1:
                final_similarity = similarities[0]
            else:
                # Weighted average (embeddings get highest weight)
                weights = [0.7, 0.2, 0.1][:len(similarities)]
                final_similarity = sum(sim * weight for sim, weight in zip(similarities, weights))
        else:
            final_similarity = 0.0
            method_used = "failed"
        
        return min(final_similarity, 1.0), method_used
    
    def _calculate_additional_factors(self, base_content, candidate_content) -> Dict[str, float]:
        """Calculate additional similarity factors"""
        factors = {}
        
        try:
            # Genre similarity
            base_genres = set(json.loads(base_content.genres or '[]'))
            candidate_genres = set(json.loads(candidate_content.genres or '[]'))
            
            if base_genres and candidate_genres:
                factors['genre_similarity'] = len(base_genres.intersection(candidate_genres)) / len(base_genres.union(candidate_genres))
            else:
                factors['genre_similarity'] = 0.0
            
            # Rating similarity (normalized)
            if base_content.rating and candidate_content.rating:
                rating_diff = abs(base_content.rating - candidate_content.rating)
                factors['rating_similarity'] = max(0, 1 - (rating_diff / 10))
            else:
                factors['rating_similarity'] = 0.5
            
            # Release year proximity
            if base_content.release_date and candidate_content.release_date:
                year_diff = abs(base_content.release_date.year - candidate_content.release_date.year)
                factors['year_proximity'] = max(0, 1 - (year_diff / 20))  # 20-year window
            else:
                factors['year_proximity'] = 0.5
            
            # Language match (exact)
            base_languages = json.loads(base_content.languages or '[]')
            candidate_languages = json.loads(candidate_content.languages or '[]')
            factors['language_match'] = 1.0 if self.language_matcher.languages_match(base_languages, candidate_languages) else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating additional factors: {e}")
            factors = {
                'genre_similarity': 0.0,
                'rating_similarity': 0.5,
                'year_proximity': 0.5,
                'language_match': 0.0
            }
        
        return factors
    
    def get_similar_titles(self, content, limit: int = 8, min_similarity: float = 0.4, use_cache: bool = True) -> List[SimilarityResult]:
        """
        Get similar titles for given content
        
        Args:
            content: Content object to find similarities for
            limit: Maximum number of similar titles to return
            min_similarity: Minimum similarity score threshold
            use_cache: Whether to use caching
            
        Returns:
            List of SimilarityResult objects sorted by similarity score
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(content.id, limit, min_similarity)
            if use_cache and self.cache:
                try:
                    cached_results = self.cache.get(cache_key)
                    if cached_results:
                        logger.info(f"Cache hit for similar titles: {content.id}")
                        return [SimilarityResult(**result) for result in cached_results]
                except Exception as e:
                    logger.warning(f"Cache retrieval error: {e}")
            
            start_time = time.time()
            
            # Validate input content
            if not content or not content.overview:
                logger.warning(f"Content {content.id if content else 'None'} has no overview for similarity")
                return []
            
            # Get candidate content
            candidates = self._get_candidate_content(content, limit_candidates=min(500, limit * 25))
            
            if not candidates:
                logger.warning(f"No candidate content found for {content.id}")
                return []
            
            logger.info(f"Processing {len(candidates)} candidates for content {content.id}")
            
            # Calculate similarities
            similarity_results = []
            
            for candidate in candidates:
                try:
                    # Language filter (strict)
                    base_languages = json.loads(content.languages or '[]')
                    candidate_languages = json.loads(candidate.languages or '[]')
                    
                    if not self.language_matcher.languages_match(base_languages, candidate_languages):
                        continue
                    
                    # Title relevance check
                    if not self.title_validator.is_title_relevant(content.title, candidate.title):
                        continue
                    
                    # Calculate story similarity (primary factor)
                    story_similarity, method = self._calculate_story_similarity(content.overview, candidate.overview)
                    
                    if story_similarity < min_similarity:
                        continue
                    
                    # Calculate additional factors
                    additional_factors = self._calculate_additional_factors(content, candidate)
                    
                    # Calculate final similarity score (story-weighted)
                    final_score = (
                        story_similarity * 0.7 +  # Story is 70% of the score
                        additional_factors['genre_similarity'] * 0.15 +
                        additional_factors['rating_similarity'] * 0.05 +
                        additional_factors['year_proximity'] * 0.05 +
                        additional_factors['language_match'] * 0.05
                    )
                    
                    # Calculate confidence based on multiple factors
                    confidence = min(1.0, (
                        (1.0 if method in ['embeddings', 'tfidf'] else 0.7) +
                        (0.2 if additional_factors['genre_similarity'] > 0.3 else 0.0) +
                        (0.1 if additional_factors['language_match'] > 0.5 else 0.0)
                    ))
                    
                    # Create similarity result
                    result = SimilarityResult(
                        content_id=candidate.id,
                        content=candidate,
                        similarity_score=final_score,
                        match_type=f"story_{method}",
                        confidence=confidence,
                        details={
                            'story_similarity': story_similarity,
                            'similarity_method': method,
                            **additional_factors,
                            'final_weighted_score': final_score
                        }
                    )
                    
                    similarity_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error processing candidate {candidate.id}: {e}")
                    continue
            
            # Sort by similarity score and confidence
            similarity_results.sort(key=lambda x: (x.similarity_score, x.confidence), reverse=True)
            
            # Limit results
            final_results = similarity_results[:limit]
            
            # Cache results
            if use_cache and self.cache and final_results:
                try:
                    cache_data = [
                        {
                            'content_id': result.content_id,
                            'content': result.content,  # Note: This might not serialize well
                            'similarity_score': result.similarity_score,
                            'match_type': result.match_type,
                            'confidence': result.confidence,
                            'details': result.details
                        }
                        for result in final_results
                    ]
                    self.cache.set(cache_key, cache_data, timeout=1800)  # 30 minutes
                except Exception as e:
                    logger.warning(f"Cache storage error: {e}")
            
            processing_time = time.time() - start_time
            logger.info(f"Similar titles calculation completed for {content.id}: "
                       f"{len(final_results)} results in {processing_time:.2f}s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in get_similar_titles for content {content.id if content else 'None'}: {e}")
            return []
    
    def get_similar_titles_formatted(self, content, limit: int = 8, min_similarity: float = 0.4) -> List[Dict[str, Any]]:
        """
        Get similar titles formatted for API response
        
        Returns:
            List of dictionaries with formatted content data
        """
        try:
            results = self.get_similar_titles(content, limit, min_similarity)
            
            formatted_results = []
            for result in results:
                candidate = result.content
                
                # Ensure candidate has slug
                if not candidate.slug:
                    candidate.slug = f"content-{candidate.id}"
                
                # Get YouTube trailer URL
                youtube_url = None
                if candidate.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={candidate.youtube_trailer_id}"
                
                # Format poster path
                poster_path = None
                if candidate.poster_path:
                    if candidate.poster_path.startswith('http'):
                        poster_path = candidate.poster_path
                    else:
                        poster_path = f"https://image.tmdb.org/t/p/w300{candidate.poster_path}"
                
                formatted_result = {
                    'id': candidate.id,
                    'slug': candidate.slug,
                    'title': candidate.title,
                    'poster_path': poster_path,
                    'rating': candidate.rating,
                    'content_type': candidate.content_type,
                    'youtube_trailer': youtube_url,
                    'similarity_score': round(result.similarity_score, 3),
                    'match_type': result.match_type,
                    'confidence': round(result.confidence, 3),
                    'similarity_details': {
                        'story_similarity': round(result.details.get('story_similarity', 0), 3),
                        'genre_similarity': round(result.details.get('genre_similarity', 0), 3),
                        'method': result.details.get('similarity_method', 'unknown')
                    }
                }
                
                # Add genres if available
                try:
                    formatted_result['genres'] = json.loads(candidate.genres or '[]')
                except:
                    formatted_result['genres'] = []
                
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error formatting similar titles: {e}")
            return []

# Factory function for easy integration
def create_similar_titles_engine(db, content_model, cache=None) -> SimilarTitlesEngine:
    """
    Factory function to create SimilarTitlesEngine instance
    
    Args:
        db: SQLAlchemy database instance
        content_model: Content model class
        cache: Cache instance (optional)
        
    Returns:
        SimilarTitlesEngine instance
    """
    return SimilarTitlesEngine(db, content_model, cache)

# Test function
def test_similar_titles_engine():
    """
    Test function for SimilarTitlesEngine
    Usage: python -c "from backend.services.similar import test_similar_titles_engine; test_similar_titles_engine()"
    """
    print("Testing SimilarTitlesEngine...")
    
    # Test text preprocessing
    preprocessor = TextPreprocessor()
    test_text = "A young wizard goes to a magical school and fights against dark forces."
    cleaned = preprocessor.clean_text(test_text)
    keywords = preprocessor.extract_plot_keywords(test_text)
    
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
    print(f"Keywords: {keywords}")
    
    # Test language matching
    lang1 = ['english', 'en']
    lang2 = ['en', 'english']
    lang3 = ['hindi', 'hi']
    
    print(f"English variants match: {LanguageMatcher.languages_match(lang1, lang2)}")
    print(f"English-Hindi match: {LanguageMatcher.languages_match(lang1, lang3)}")
    
    # Test embedding engine (if available)
    embedding_engine = StoryEmbeddingEngine()
    if embedding_engine.model or embedding_engine.spacy_model:
        sim = embedding_engine.calculate_similarity(
            "A young wizard learns magic at school",
            "A teenage sorcerer studies witchcraft at academy"
        )
        print(f"Story similarity: {sim:.3f}")
    else:
        print("No embedding models available for testing")
    
    print("Test completed!")

if __name__ == "__main__":
    test_similar_titles_engine()