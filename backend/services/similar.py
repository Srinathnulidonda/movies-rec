# backend/services/similar.py
import json
import logging
import re
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
from functools import lru_cache, wraps
import difflib
import hashlib
import time

from sqlalchemy import and_, or_, func, text
from sqlalchemy.orm import joinedload
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Configure logging
logger = logging.getLogger(__name__)

class SimilarityType(Enum):
    """Types of similarity calculations"""
    LANGUAGE_EXACT = "language_exact"
    STORY_CONTENT = "story_content"
    GENRE_BASED = "genre_based"
    CAST_CREW = "cast_crew"
    HYBRID = "hybrid"
    PRODUCTION = "production"

class LanguageFamily(Enum):
    """Language family classifications"""
    DRAVIDIAN = "dravidian"  # Telugu, Tamil, Kannada, Malayalam
    INDO_ARYAN = "indo_aryan"  # Hindi, Bengali, etc.
    GERMANIC = "germanic"  # English
    JAPANESE = "japanese"  # Anime
    OTHER = "other"

@dataclass
class SimilarityScore:
    """Comprehensive similarity scoring"""
    overall_score: float
    language_score: float
    story_score: float
    genre_score: float
    cast_score: float
    production_score: float
    quality_score: float
    confidence: float
    match_reasons: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure scores are within valid range"""
        for attr in ['overall_score', 'language_score', 'story_score', 
                    'genre_score', 'cast_score', 'production_score', 
                    'quality_score', 'confidence']:
            value = getattr(self, attr)
            if value < 0 or value > 1:
                setattr(self, attr, max(0, min(1, value)))

@dataclass
class ContentMetadata:
    """Extracted content metadata for similarity analysis"""
    id: int
    title: str
    original_title: str
    languages: List[str]
    genres: List[str]
    anime_genres: List[str]
    overview: str
    directors: List[str]
    writers: List[str]
    main_cast: List[str]
    production_companies: List[str]
    keywords: List[str]
    themes: List[str]
    narrative_elements: List[str]
    content_type: str
    release_year: int
    rating: float
    popularity: float
    language_family: LanguageFamily
    
class AdvancedLanguageDetector:
    """Advanced language detection and classification"""
    
    LANGUAGE_MAPPINGS = {
        # Telugu variations
        'te': 'telugu', 'tel': 'telugu', 'telugu': 'telugu',
        'tollywood': 'telugu', 'andhra': 'telugu', 'telangana': 'telugu',
        
        # Tamil variations
        'ta': 'tamil', 'tam': 'tamil', 'tamil': 'tamil',
        'kollywood': 'tamil', 'tamilnadu': 'tamil',
        
        # Hindi variations
        'hi': 'hindi', 'hin': 'hindi', 'hindi': 'hindi',
        'bollywood': 'hindi', 'hindustani': 'hindi',
        
        # Malayalam variations
        'ml': 'malayalam', 'mal': 'malayalam', 'malayalam': 'malayalam',
        'mollywood': 'malayalam', 'kerala': 'malayalam',
        
        # Kannada variations
        'kn': 'kannada', 'kan': 'kannada', 'kannada': 'kannada',
        'sandalwood': 'kannada', 'karnataka': 'kannada',
        
        # English variations
        'en': 'english', 'eng': 'english', 'english': 'english',
        'hollywood': 'english',
        
        # Japanese
        'ja': 'japanese', 'jpn': 'japanese', 'japanese': 'japanese'
    }
    
    LANGUAGE_FAMILIES = {
        'telugu': LanguageFamily.DRAVIDIAN,
        'tamil': LanguageFamily.DRAVIDIAN,
        'malayalam': LanguageFamily.DRAVIDIAN,
        'kannada': LanguageFamily.DRAVIDIAN,
        'hindi': LanguageFamily.INDO_ARYAN,
        'english': LanguageFamily.GERMANIC,
        'japanese': LanguageFamily.JAPANESE
    }
    
    @classmethod
    def normalize_language(cls, lang: str) -> str:
        """Normalize language string to standard form"""
        if not lang:
            return 'unknown'
        
        lang_clean = re.sub(r'[^a-zA-Z]', '', lang.lower().strip())
        return cls.LANGUAGE_MAPPINGS.get(lang_clean, lang_clean)
    
    @classmethod
    def get_language_family(cls, lang: str) -> LanguageFamily:
        """Get language family for a language"""
        normalized = cls.normalize_language(lang)
        return cls.LANGUAGE_FAMILIES.get(normalized, LanguageFamily.OTHER)
    
    @classmethod
    def extract_languages(cls, content_languages: List[str]) -> Tuple[List[str], LanguageFamily]:
        """Extract and normalize languages from content"""
        if not content_languages:
            return ['unknown'], LanguageFamily.OTHER
        
        normalized_languages = []
        families = []
        
        for lang in content_languages:
            normalized = cls.normalize_language(lang)
            if normalized not in normalized_languages:
                normalized_languages.append(normalized)
                families.append(cls.get_language_family(normalized))
        
        # Determine primary language family
        if families:
            family_counts = Counter(families)
            primary_family = family_counts.most_common(1)[0][0]
        else:
            primary_family = LanguageFamily.OTHER
        
        return normalized_languages, primary_family

class StoryContentAnalyzer:
    """Advanced story and content analysis"""
    
    def __init__(self):
        self.vectorizer = None
        self.svd = None
        self.is_initialized = False
        self._init_nlp_tools()
    
    def _init_nlp_tools(self):
        """Initialize NLP tools"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),
                max_df=0.95,
                min_df=2,
                lowercase=True,
                strip_accents='unicode'
            )
            
            # Initialize SVD for dimensionality reduction
            self.svd = TruncatedSVD(n_components=100, random_state=42)
            
        except Exception as e:
            logger.warning(f"NLP tools initialization failed: {e}")
            self.lemmatizer = None
            self.stop_words = set()
    
    def extract_themes(self, overview: str, genres: List[str]) -> List[str]:
        """Extract themes from content overview and genres"""
        themes = []
        
        if not overview:
            return themes
        
        # Theme keywords mapping
        theme_keywords = {
            'revenge': ['revenge', 'vengeance', 'payback', 'retribution'],
            'love': ['love', 'romance', 'relationship', 'wedding', 'marriage'],
            'family': ['family', 'father', 'mother', 'brother', 'sister', 'son', 'daughter'],
            'friendship': ['friend', 'friendship', 'buddy', 'companion'],
            'betrayal': ['betrayal', 'betray', 'deception', 'backstab'],
            'sacrifice': ['sacrifice', 'sacrifice', 'selfless', 'giving up'],
            'redemption': ['redemption', 'redeem', 'second chance', 'forgiveness'],
            'power': ['power', 'authority', 'control', 'dominance', 'rule'],
            'justice': ['justice', 'law', 'court', 'judge', 'crime'],
            'survival': ['survival', 'survive', 'escape', 'danger', 'threat'],
            'coming_of_age': ['growing up', 'teenager', 'adolescent', 'maturity'],
            'social_issues': ['society', 'social', 'inequality', 'discrimination'],
            'supernatural': ['supernatural', 'magic', 'mystical', 'paranormal'],
            'technology': ['technology', 'ai', 'robot', 'cyber', 'digital'],
            'war': ['war', 'battle', 'conflict', 'military', 'soldier'],
            'identity': ['identity', 'who am i', 'self-discovery', 'purpose']
        }
        
        overview_lower = overview.lower()
        
        # Extract themes based on keywords
        for theme, keywords in theme_keywords.items():
            if any(keyword in overview_lower for keyword in keywords):
                themes.append(theme)
        
        # Add genre-based themes
        genre_themes = {
            'action': ['action', 'adventure'],
            'drama': ['emotional', 'character_driven'],
            'comedy': ['humor', 'lighthearted'],
            'thriller': ['suspense', 'tension'],
            'horror': ['fear', 'supernatural'],
            'romance': ['love', 'relationship'],
            'sci-fi': ['technology', 'future'],
            'fantasy': ['supernatural', 'magic']
        }
        
        for genre in genres:
            genre_lower = genre.lower()
            if genre_lower in genre_themes:
                themes.extend(genre_themes[genre_lower])
        
        return list(set(themes))
    
    def extract_narrative_elements(self, overview: str) -> List[str]:
        """Extract narrative elements from overview"""
        elements = []
        
        if not overview:
            return elements
        
        narrative_patterns = {
            'flashback': r'\b(flashback|past|memory|remembers?|recalls?)\b',
            'twist': r'\b(twist|unexpected|surprise|shocking|reveal)\b',
            'multiple_timeline': r'\b(timeline|parallel|alternate|different time)\b',
            'narrator': r'\b(narrator|tells story|narrates|voice.?over)\b',
            'ensemble': r'\b(group|team|ensemble|multiple characters)\b',
            'journey': r'\b(journey|travel|quest|adventure|road)\b',
            'mystery': r'\b(mystery|secret|hidden|unknown|investigate)\b',
            'transformation': r'\b(change|transform|becomes?|turns? into)\b'
        }
        
        overview_lower = overview.lower()
        
        for element, pattern in narrative_patterns.items():
            if re.search(pattern, overview_lower):
                elements.append(element)
        
        return elements
    
    def calculate_story_similarity(self, content1: ContentMetadata, content2: ContentMetadata) -> float:
        """Calculate story similarity between two content items"""
        try:
            # Overview similarity using TF-IDF
            overview_sim = self._calculate_text_similarity(
                content1.overview or "", 
                content2.overview or ""
            )
            
            # Theme similarity
            theme_sim = self._calculate_set_similarity(
                content1.themes, 
                content2.themes
            )
            
            # Narrative elements similarity
            narrative_sim = self._calculate_set_similarity(
                content1.narrative_elements,
                content2.narrative_elements
            )
            
            # Keyword similarity
            keyword_sim = self._calculate_set_similarity(
                content1.keywords,
                content2.keywords
            )
            
            # Weighted combination
            story_score = (
                overview_sim * 0.4 +
                theme_sim * 0.3 +
                narrative_sim * 0.2 +
                keyword_sim * 0.1
            )
            
            return min(1.0, max(0.0, story_score))
            
        except Exception as e:
            logger.error(f"Story similarity calculation error: {e}")
            return 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        if not text1 or not text2:
            return 0.0
        
        try:
            # Use sequence matcher for basic similarity
            similarity = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
            
            # If vectorizer is available, use TF-IDF
            if self.vectorizer and len(text1) > 20 and len(text2) > 20:
                try:
                    tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
                    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                    # Combine both measures
                    similarity = (similarity + cosine_sim) / 2
                except:
                    pass
            
            return similarity
            
        except Exception as e:
            logger.warning(f"Text similarity calculation error: {e}")
            return 0.0
    
    def _calculate_set_similarity(self, set1: List[str], set2: List[str]) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0
        
        s1 = set(item.lower() for item in set1)
        s2 = set(item.lower() for item in set2)
        
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        
        return intersection / union if union > 0 else 0.0

class PerfectTitleMatcher:
    """Perfect title matching with fuzzy matching capabilities"""
    
    @staticmethod
    def normalize_title(title: str) -> str:
        """Normalize title for comparison"""
        if not title:
            return ""
        
        # Remove common words and punctuation
        title = re.sub(r'[^\w\s]', ' ', title.lower())
        title = re.sub(r'\b(the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title
    
    @classmethod
    def calculate_title_similarity(cls, title1: str, title2: str, original_title1: str = "", original_title2: str = "") -> float:
        """Calculate comprehensive title similarity"""
        if not title1 or not title2:
            return 0.0
        
        similarities = []
        
        # Direct title comparison
        norm_title1 = cls.normalize_title(title1)
        norm_title2 = cls.normalize_title(title2)
        
        if norm_title1 and norm_title2:
            similarities.append(difflib.SequenceMatcher(None, norm_title1, norm_title2).ratio())
        
        # Original title comparison
        if original_title1 and original_title2:
            norm_orig1 = cls.normalize_title(original_title1)
            norm_orig2 = cls.normalize_title(original_title2)
            similarities.append(difflib.SequenceMatcher(None, norm_orig1, norm_orig2).ratio())
        
        # Cross comparisons (title vs original title)
        if original_title1:
            similarities.append(difflib.SequenceMatcher(None, norm_title1, cls.normalize_title(original_title1)).ratio())
        
        if original_title2:
            similarities.append(difflib.SequenceMatcher(None, norm_title2, cls.normalize_title(original_title2)).ratio())
        
        return max(similarities) if similarities else 0.0

class ProductionAnalyzer:
    """Analyze production-related similarities"""
    
    @staticmethod
    def calculate_cast_similarity(cast1: List[str], cast2: List[str]) -> float:
        """Calculate cast similarity"""
        if not cast1 or not cast2:
            return 0.0
        
        set1 = set(name.lower().strip() for name in cast1)
        set2 = set(name.lower().strip() for name in cast2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_crew_similarity(directors1: List[str], directors2: List[str], 
                                writers1: List[str], writers2: List[str]) -> float:
        """Calculate crew similarity"""
        director_sim = ProductionAnalyzer.calculate_cast_similarity(directors1, directors2)
        writer_sim = ProductionAnalyzer.calculate_cast_similarity(writers1, writers2)
        
        # Weight directors more heavily
        return director_sim * 0.7 + writer_sim * 0.3
    
    @staticmethod
    def calculate_production_similarity(companies1: List[str], companies2: List[str]) -> float:
        """Calculate production company similarity"""
        return ProductionAnalyzer.calculate_cast_similarity(companies1, companies2)

class SimilarityCache:
    """Advanced caching for similarity calculations"""
    
    def __init__(self, cache_backend=None):
        self.cache = cache_backend
        self.local_cache = {}
        self.cache_timeout = 3600  # 1 hour
        
    def _generate_cache_key(self, content_id1: int, content_id2: int, similarity_type: str) -> str:
        """Generate cache key for similarity pair"""
        # Ensure consistent ordering
        id1, id2 = sorted([content_id1, content_id2])
        return f"similarity:{similarity_type}:{id1}:{id2}"
    
    def get_similarity(self, content_id1: int, content_id2: int, similarity_type: str) -> Optional[SimilarityScore]:
        """Get cached similarity score"""
        cache_key = self._generate_cache_key(content_id1, content_id2, similarity_type)
        
        # Try Redis cache first
        if self.cache:
            try:
                cached_data = self.cache.get(cache_key)
                if cached_data:
                    return SimilarityScore(**cached_data)
            except Exception as e:
                logger.warning(f"Cache get error: {e}")
        
        # Try local cache
        return self.local_cache.get(cache_key)
    
    def set_similarity(self, content_id1: int, content_id2: int, similarity_type: str, score: SimilarityScore):
        """Cache similarity score"""
        cache_key = self._generate_cache_key(content_id1, content_id2, similarity_type)
        
        # Store in Redis cache
        if self.cache:
            try:
                score_dict = {
                    'overall_score': score.overall_score,
                    'language_score': score.language_score,
                    'story_score': score.story_score,
                    'genre_score': score.genre_score,
                    'cast_score': score.cast_score,
                    'production_score': score.production_score,
                    'quality_score': score.quality_score,
                    'confidence': score.confidence,
                    'match_reasons': score.match_reasons
                }
                self.cache.set(cache_key, score_dict, timeout=self.cache_timeout)
            except Exception as e:
                logger.warning(f"Cache set error: {e}")
        
        # Store in local cache
        self.local_cache[cache_key] = score
        
        # Prevent local cache from growing too large
        if len(self.local_cache) > 1000:
            # Remove oldest 100 entries
            keys_to_remove = list(self.local_cache.keys())[:100]
            for key in keys_to_remove:
                del self.local_cache[key]

class ContentMetadataExtractor:
    """Extract comprehensive metadata from content"""
    
    def __init__(self, db, models):
        self.db = db
        self.Content = models['Content']
        self.ContentPerson = models['ContentPerson']
        self.Person = models['Person']
        self.story_analyzer = StoryContentAnalyzer()
    
    def extract_metadata(self, content) -> ContentMetadata:
        """Extract comprehensive metadata from content object"""
        try:
            # Parse languages
            languages_raw = json.loads(content.languages or '[]')
            languages, language_family = AdvancedLanguageDetector.extract_languages(languages_raw)
            
            # Parse genres
            genres = json.loads(content.genres or '[]')
            anime_genres = json.loads(content.anime_genres or '[]')
            
            # Extract cast and crew
            directors, writers, main_cast, production_companies = self._extract_cast_crew(content.id)
            
            # Extract themes and narrative elements
            themes = self.story_analyzer.extract_themes(content.overview or "", genres)
            narrative_elements = self.story_analyzer.extract_narrative_elements(content.overview or "")
            
            # Extract keywords from title and overview
            keywords = self._extract_keywords(content.title, content.overview)
            
            return ContentMetadata(
                id=content.id,
                title=content.title or "",
                original_title=content.original_title or "",
                languages=languages,
                genres=genres,
                anime_genres=anime_genres,
                overview=content.overview or "",
                directors=directors,
                writers=writers,
                main_cast=main_cast,
                production_companies=production_companies,
                keywords=keywords,
                themes=themes,
                narrative_elements=narrative_elements,
                content_type=content.content_type,
                release_year=content.release_date.year if content.release_date else 0,
                rating=content.rating or 0.0,
                popularity=content.popularity or 0.0,
                language_family=language_family
            )
            
        except Exception as e:
            logger.error(f"Metadata extraction error for content {content.id}: {e}")
            # Return minimal metadata
            return ContentMetadata(
                id=content.id,
                title=content.title or "",
                original_title=content.original_title or "",
                languages=['unknown'],
                genres=[],
                anime_genres=[],
                overview="",
                directors=[],
                writers=[],
                main_cast=[],
                production_companies=[],
                keywords=[],
                themes=[],
                narrative_elements=[],
                content_type=content.content_type,
                release_year=0,
                rating=0.0,
                popularity=0.0,
                language_family=LanguageFamily.OTHER
            )
    
    def _extract_cast_crew(self, content_id: int) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Extract cast and crew information"""
        try:
            # Get content persons with person details
            content_persons = self.db.session.query(self.ContentPerson, self.Person).join(
                self.Person, self.ContentPerson.person_id == self.Person.id
            ).filter(self.ContentPerson.content_id == content_id).all()
            
            directors = []
            writers = []
            main_cast = []
            
            for cp, person in content_persons:
                if cp.role_type == 'crew':
                    if cp.job and 'director' in cp.job.lower():
                        directors.append(person.name)
                    elif cp.job and any(job in cp.job.lower() for job in ['writer', 'screenplay', 'story']):
                        writers.append(person.name)
                elif cp.role_type == 'cast' and cp.order is not None and cp.order < 10:
                    main_cast.append(person.name)
            
            # For now, production companies would need to be extracted from TMDB data
            # This would require additional API calls or database schema changes
            production_companies = []
            
            return directors, writers, main_cast, production_companies
            
        except Exception as e:
            logger.warning(f"Cast/crew extraction error: {e}")
            return [], [], [], []
    
    def _extract_keywords(self, title: str, overview: str) -> List[str]:
        """Extract keywords from title and overview"""
        keywords = []
        
        # Extract from title
        if title:
            title_words = re.findall(r'\b\w{3,}\b', title.lower())
            keywords.extend(title_words)
        
        # Extract from overview
        if overview and self.story_analyzer.lemmatizer:
            try:
                # Tokenize and lemmatize
                tokens = word_tokenize(overview.lower())
                important_words = [
                    self.story_analyzer.lemmatizer.lemmatize(word)
                    for word in tokens
                    if (word.isalpha() and 
                        len(word) > 3 and 
                        word not in self.story_analyzer.stop_words)
                ]
                
                # Get most frequent words
                word_freq = Counter(important_words)
                top_words = [word for word, count in word_freq.most_common(10)]
                keywords.extend(top_words)
                
            except Exception as e:
                logger.warning(f"Keyword extraction error: {e}")
        
        return list(set(keywords))

class AdvancedSimilarityEngine:
    """Advanced similarity engine with multiple algorithms"""
    
    def __init__(self, db, models, cache_backend=None):
        self.db = db
        self.models = models
        self.Content = models['Content']
        
        # Initialize components
        self.cache = SimilarityCache(cache_backend)
        self.metadata_extractor = ContentMetadataExtractor(db, models)
        self.story_analyzer = StoryContentAnalyzer()
        self.title_matcher = PerfectTitleMatcher()
        self.production_analyzer = ProductionAnalyzer()
        
        # Similarity weights for different content types
        self.weights = {
            'movie': {
                'language': 0.25,
                'story': 0.30,
                'genre': 0.15,
                'cast': 0.15,
                'production': 0.10,
                'quality': 0.05
            },
            'tv': {
                'language': 0.25,
                'story': 0.30,
                'genre': 0.15,
                'cast': 0.15,
                'production': 0.10,
                'quality': 0.05
            },
            'anime': {
                'language': 0.20,
                'story': 0.35,
                'genre': 0.20,
                'cast': 0.10,
                'production': 0.10,
                'quality': 0.05
            }
        }
    
    def find_similar_content(self, content_id: int, limit: int = 10, 
                           min_similarity: float = 0.3, 
                           language_priority: bool = True) -> List[Dict[str, Any]]:
        """Find similar content with comprehensive algorithm"""
        try:
            # Get base content
            base_content = self.Content.query.get(content_id)
            if not base_content:
                raise ValueError(f"Content {content_id} not found")
            
            # Extract metadata
            base_metadata = self.metadata_extractor.extract_metadata(base_content)
            
            # Get candidate content
            candidates = self._get_candidate_content(base_metadata, limit * 5)
            
            # Calculate similarities
            similarities = []
            for candidate in candidates:
                if candidate.id == content_id:
                    continue
                
                candidate_metadata = self.metadata_extractor.extract_metadata(candidate)
                similarity_score = self._calculate_comprehensive_similarity(
                    base_metadata, candidate_metadata
                )
                
                if similarity_score.overall_score >= min_similarity:
                    similarities.append({
                        'content': candidate,
                        'metadata': candidate_metadata,
                        'similarity_score': similarity_score
                    })
            
            # Sort by overall score
            similarities.sort(key=lambda x: x['similarity_score'].overall_score, reverse=True)
            
            # Apply language priority if enabled
            if language_priority:
                similarities = self._apply_language_priority(similarities, base_metadata)
            
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Similar content search error: {e}")
            return []
    
    def _get_candidate_content(self, base_metadata: ContentMetadata, limit: int) -> List:
        """Get candidate content for similarity comparison"""
        try:
            # Primary strategy: same content type and overlapping genres
            primary_candidates = self.Content.query.filter(
                self.Content.content_type == base_metadata.content_type,
                self.Content.id != base_metadata.id
            )
            
            # Add genre filtering if genres exist
            if base_metadata.genres:
                genre_conditions = []
                for genre in base_metadata.genres[:3]:  # Top 3 genres
                    genre_conditions.append(self.Content.genres.contains(genre))
                
                if genre_conditions:
                    primary_candidates = primary_candidates.filter(or_(*genre_conditions))
            
            primary_results = primary_candidates.order_by(
                self.Content.rating.desc(),
                self.Content.popularity.desc()
            ).limit(limit // 2).all()
            
            # Secondary strategy: same language family
            secondary_candidates = self.Content.query.filter(
                self.Content.content_type == base_metadata.content_type,
                self.Content.id != base_metadata.id
            )
            
            # Filter by language if available
            if base_metadata.languages and base_metadata.languages[0] != 'unknown':
                language_conditions = []
                for lang in base_metadata.languages:
                    language_conditions.append(self.Content.languages.contains(lang))
                
                if language_conditions:
                    secondary_candidates = secondary_candidates.filter(or_(*language_conditions))
            
            secondary_results = secondary_candidates.order_by(
                self.Content.rating.desc()
            ).limit(limit // 2).all()
            
            # Combine and deduplicate
            all_candidates = list({c.id: c for c in primary_results + secondary_results}.values())
            
            return all_candidates[:limit]
            
        except Exception as e:
            logger.error(f"Candidate selection error: {e}")
            return []
    
    def _calculate_comprehensive_similarity(self, content1: ContentMetadata, 
                                         content2: ContentMetadata) -> SimilarityScore:
        """Calculate comprehensive similarity score"""
        
        # Check cache first
        cached_score = self.cache.get_similarity(
            content1.id, content2.id, SimilarityType.HYBRID.value
        )
        if cached_score:
            return cached_score
        
        try:
            match_reasons = []
            
            # 1. Language similarity (HIGHEST PRIORITY)
            language_score = self._calculate_language_similarity(content1, content2)
            if language_score > 0.8:
                match_reasons.append(f"Strong language match ({language_score:.2f})")
            
            # 2. Story similarity
            story_score = self.story_analyzer.calculate_story_similarity(content1, content2)
            if story_score > 0.6:
                match_reasons.append(f"Similar story elements ({story_score:.2f})")
            
            # 3. Genre similarity
            genre_score = self._calculate_genre_similarity(content1, content2)
            if genre_score > 0.5:
                match_reasons.append(f"Genre compatibility ({genre_score:.2f})")
            
            # 4. Cast similarity
            cast_score = self.production_analyzer.calculate_cast_similarity(
                content1.main_cast, content2.main_cast
            )
            if cast_score > 0.3:
                match_reasons.append(f"Shared cast members ({cast_score:.2f})")
            
            # 5. Production similarity
            production_score = self.production_analyzer.calculate_crew_similarity(
                content1.directors, content2.directors,
                content1.writers, content2.writers
            )
            if production_score > 0.3:
                match_reasons.append(f"Same crew members ({production_score:.2f})")
            
            # 6. Quality similarity
            quality_score = self._calculate_quality_similarity(content1, content2)
            
            # Get weights for content type
            weights = self.weights.get(content1.content_type, self.weights['movie'])
            
            # Calculate weighted overall score
            overall_score = (
                language_score * weights['language'] +
                story_score * weights['story'] +
                genre_score * weights['genre'] +
                cast_score * weights['cast'] +
                production_score * weights['production'] +
                quality_score * weights['quality']
            )
            
            # Calculate confidence based on available data
            confidence = self._calculate_confidence(content1, content2)
            
            similarity_score = SimilarityScore(
                overall_score=overall_score,
                language_score=language_score,
                story_score=story_score,
                genre_score=genre_score,
                cast_score=cast_score,
                production_score=production_score,
                quality_score=quality_score,
                confidence=confidence,
                match_reasons=match_reasons
            )
            
            # Cache the result
            self.cache.set_similarity(
                content1.id, content2.id, SimilarityType.HYBRID.value, similarity_score
            )
            
            return similarity_score
            
        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return SimilarityScore(
                overall_score=0.0, language_score=0.0, story_score=0.0,
                genre_score=0.0, cast_score=0.0, production_score=0.0,
                quality_score=0.0, confidence=0.0
            )
    
    def _calculate_language_similarity(self, content1: ContentMetadata, content2: ContentMetadata) -> float:
        """Calculate language similarity with perfect accuracy"""
        # Exact language match gets highest score
        common_languages = set(content1.languages).intersection(set(content2.languages))
        if common_languages:
            exact_match_score = len(common_languages) / max(len(content1.languages), len(content2.languages))
            if exact_match_score == 1.0:
                return 1.0  # Perfect match
            else:
                return 0.8 + (exact_match_score * 0.2)  # High score for partial match
        
        # Language family match
        if content1.language_family == content2.language_family and content1.language_family != LanguageFamily.OTHER:
            return 0.6  # Good score for same family
        
        # No language match
        return 0.0
    
    def _calculate_genre_similarity(self, content1: ContentMetadata, content2: ContentMetadata) -> float:
        """Calculate genre similarity"""
        # Regular genres
        genre_sim = self._calculate_set_similarity(content1.genres, content2.genres)
        
        # Anime genres (if applicable)
        anime_genre_sim = 0.0
        if content1.anime_genres and content2.anime_genres:
            anime_genre_sim = self._calculate_set_similarity(content1.anime_genres, content2.anime_genres)
        
        # Combine scores
        if anime_genre_sim > 0:
            return (genre_sim + anime_genre_sim) / 2
        return genre_sim
    
    def _calculate_quality_similarity(self, content1: ContentMetadata, content2: ContentMetadata) -> float:
        """Calculate quality similarity based on ratings"""
        if content1.rating == 0 or content2.rating == 0:
            return 0.5  # Neutral score if no ratings
        
        rating_diff = abs(content1.rating - content2.rating)
        max_diff = 10.0  # Assuming 10-point scale
        
        return 1.0 - (rating_diff / max_diff)
    
    def _calculate_confidence(self, content1: ContentMetadata, content2: ContentMetadata) -> float:
        """Calculate confidence score based on available data"""
        factors = []
        
        # Overview availability
        if content1.overview and content2.overview:
            factors.append(0.3)
        elif content1.overview or content2.overview:
            factors.append(0.15)
        
        # Cast availability
        if content1.main_cast and content2.main_cast:
            factors.append(0.2)
        elif content1.main_cast or content2.main_cast:
            factors.append(0.1)
        
        # Crew availability
        if content1.directors and content2.directors:
            factors.append(0.15)
        
        # Genre availability
        if content1.genres and content2.genres:
            factors.append(0.2)
        
        # Language availability
        if content1.languages and content2.languages:
            factors.append(0.15)
        
        return sum(factors)
    
    def _calculate_set_similarity(self, set1: List[str], set2: List[str]) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0
        
        s1 = set(item.lower() for item in set1)
        s2 = set(item.lower() for item in set2)
        
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_language_priority(self, similarities: List[Dict], base_metadata: ContentMetadata) -> List[Dict]:
        """Apply language priority to similarity results"""
        # Group by language match level
        exact_matches = []
        family_matches = []
        other_matches = []
        
        for sim in similarities:
            candidate_metadata = sim['metadata']
            
            # Check for exact language match
            if set(base_metadata.languages).intersection(set(candidate_metadata.languages)):
                exact_matches.append(sim)
            # Check for language family match
            elif (base_metadata.language_family == candidate_metadata.language_family and 
                  base_metadata.language_family != LanguageFamily.OTHER):
                family_matches.append(sim)
            else:
                other_matches.append(sim)
        
        # Boost scores for language matches
        for sim in exact_matches:
            sim['similarity_score'].overall_score = min(1.0, sim['similarity_score'].overall_score * 1.2)
        
        for sim in family_matches:
            sim['similarity_score'].overall_score = min(1.0, sim['similarity_score'].overall_score * 1.1)
        
        # Re-sort and combine
        exact_matches.sort(key=lambda x: x['similarity_score'].overall_score, reverse=True)
        family_matches.sort(key=lambda x: x['similarity_score'].overall_score, reverse=True)
        other_matches.sort(key=lambda x: x['similarity_score'].overall_score, reverse=True)
        
        return exact_matches + family_matches + other_matches

class GenreExplorer:
    """Advanced genre exploration system"""
    
    def __init__(self, db, models, similarity_engine: AdvancedSimilarityEngine):
        self.db = db
        self.models = models
        self.Content = models['Content']
        self.similarity_engine = similarity_engine
        
        # Genre relationships and hierarchies
        self.genre_relationships = {
            'action': ['adventure', 'thriller', 'crime'],
            'drama': ['romance', 'family', 'biography'],
            'comedy': ['romantic comedy', 'family', 'animation'],
            'horror': ['thriller', 'supernatural', 'mystery'],
            'sci-fi': ['fantasy', 'adventure', 'thriller'],
            'fantasy': ['adventure', 'family', 'animation'],
            'romance': ['drama', 'comedy'],
            'thriller': ['action', 'crime', 'mystery'],
            'animation': ['family', 'comedy', 'adventure'],
            'documentary': ['biography', 'history'],
            'mystery': ['thriller', 'crime', 'drama'],
            'crime': ['action', 'thriller', 'drama']
        }
        
        # Anime genre relationships
        self.anime_genre_relationships = {
            'shonen': ['action', 'adventure', 'martial arts'],
            'shojo': ['romance', 'drama', 'slice of life'],
            'seinen': ['action', 'drama', 'psychological'],
            'josei': ['romance', 'drama', 'slice of life'],
            'mecha': ['action', 'sci-fi', 'military'],
            'isekai': ['fantasy', 'adventure', 'magic'],
            'slice of life': ['drama', 'comedy', 'school'],
            'supernatural': ['fantasy', 'horror', 'mystery']
        }
    
    def explore_genre(self, genre: str, content_type: str = 'movie', 
                     limit: int = 20, language_preference: List[str] = None,
                     quality_threshold: float = 6.0) -> Dict[str, Any]:
        """Explore content by genre with advanced filtering"""
        try:
            # Normalize genre
            genre_normalized = genre.lower().strip()
            
            # Get direct genre matches
            direct_matches = self._get_direct_genre_matches(
                genre_normalized, content_type, limit * 2, language_preference, quality_threshold
            )
            
            # Get related genre matches
            related_matches = self._get_related_genre_matches(
                genre_normalized, content_type, limit, language_preference, quality_threshold
            )
            
            # Combine and deduplicate
            all_matches = self._combine_and_deduplicate(direct_matches, related_matches)
            
            # Apply intelligent sorting
            sorted_matches = self._apply_intelligent_sorting(
                all_matches, genre_normalized, language_preference
            )
            
            # Extract metadata for response
            results = []
            for content in sorted_matches[:limit]:
                metadata = self.similarity_engine.metadata_extractor.extract_metadata(content)
                
                # Calculate genre relevance score
                relevance_score = self._calculate_genre_relevance(metadata, genre_normalized)
                
                results.append({
                    'content': content,
                    'metadata': metadata,
                    'genre_relevance': relevance_score,
                    'match_type': self._determine_match_type(metadata, genre_normalized)
                })
            
            return {
                'genre': genre,
                'content_type': content_type,
                'total_results': len(results),
                'results': results,
                'related_genres': self._get_related_genres(genre_normalized, content_type),
                'language_distribution': self._analyze_language_distribution(results),
                'quality_stats': self._analyze_quality_stats(results)
            }
            
        except Exception as e:
            logger.error(f"Genre exploration error: {e}")
            return {
                'genre': genre,
                'content_type': content_type,
                'total_results': 0,
                'results': [],
                'error': str(e)
            }
    
    def _get_direct_genre_matches(self, genre: str, content_type: str, limit: int,
                                language_preference: List[str] = None,
                                quality_threshold: float = 6.0) -> List:
        """Get content with direct genre matches"""
        query = self.Content.query.filter(
            self.Content.content_type == content_type,
            or_(
                self.Content.genres.contains(f'"{genre}"'),
                self.Content.genres.contains(genre.title()),
                self.Content.anime_genres.contains(f'"{genre}"'),
                self.Content.anime_genres.contains(genre.title())
            )
        )
        
        # Apply quality filter
        if quality_threshold > 0:
            query = query.filter(self.Content.rating >= quality_threshold)
        
        # Apply language preference
        if language_preference:
            language_conditions = []
            for lang in language_preference:
                language_conditions.append(self.Content.languages.contains(lang))
            query = query.filter(or_(*language_conditions))
        
        return query.order_by(
            self.Content.rating.desc(),
            self.Content.popularity.desc()
        ).limit(limit).all()
    
    def _get_related_genre_matches(self, genre: str, content_type: str, limit: int,
                                 language_preference: List[str] = None,
                                 quality_threshold: float = 6.0) -> List:
        """Get content with related genres"""
        related_genres = []
        
        # Get related genres from mappings
        if genre in self.genre_relationships:
            related_genres.extend(self.genre_relationships[genre])
        
        if content_type == 'anime' and genre in self.anime_genre_relationships:
            related_genres.extend(self.anime_genre_relationships[genre])
        
        if not related_genres:
            return []
        
        # Build query for related genres
        genre_conditions = []
        for related_genre in related_genres[:3]:  # Limit to top 3 related genres
            genre_conditions.extend([
                self.Content.genres.contains(f'"{related_genre}"'),
                self.Content.genres.contains(related_genre.title())
            ])
            
            if content_type == 'anime':
                genre_conditions.extend([
                    self.Content.anime_genres.contains(f'"{related_genre}"'),
                    self.Content.anime_genres.contains(related_genre.title())
                ])
        
        query = self.Content.query.filter(
            self.Content.content_type == content_type,
            or_(*genre_conditions)
        )
        
        # Apply quality filter
        if quality_threshold > 0:
            query = query.filter(self.Content.rating >= quality_threshold)
        
        # Apply language preference
        if language_preference:
            language_conditions = []
            for lang in language_preference:
                language_conditions.append(self.Content.languages.contains(lang))
            query = query.filter(or_(*language_conditions))
        
        return query.order_by(
            self.Content.rating.desc()
        ).limit(limit).all()
    
    def _combine_and_deduplicate(self, direct_matches: List, related_matches: List) -> List:
        """Combine and deduplicate matches"""
        seen_ids = set()
        combined = []
        
        # Add direct matches first (higher priority)
        for content in direct_matches:
            if content.id not in seen_ids:
                seen_ids.add(content.id)
                combined.append(content)
        
        # Add related matches
        for content in related_matches:
            if content.id not in seen_ids:
                seen_ids.add(content.id)
                combined.append(content)
        
        return combined
    
    def _apply_intelligent_sorting(self, matches: List, genre: str, 
                                 language_preference: List[str] = None) -> List:
        """Apply intelligent sorting based on multiple factors"""
        def sort_key(content):
            score = 0.0
            
            # Genre relevance (highest weight)
            try:
                genres = json.loads(content.genres or '[]')
                anime_genres = json.loads(content.anime_genres or '[]')
                all_genres = [g.lower() for g in genres + anime_genres]
                
                if genre in all_genres:
                    score += 100  # Direct match
                else:
                    # Check for partial matches
                    for g in all_genres:
                        if genre in g or g in genre:
                            score += 50
            except:
                pass
            
            # Language preference
            if language_preference:
                try:
                    languages = json.loads(content.languages or '[]')
                    for i, pref_lang in enumerate(language_preference):
                        if any(pref_lang.lower() in lang.lower() for lang in languages):
                            score += (10 - i) * 10  # Higher score for higher preference
                            break
                except:
                    pass
            
            # Quality score
            score += (content.rating or 0) * 5
            
            # Popularity score
            score += (content.popularity or 0) * 0.1
            
            return score
        
        return sorted(matches, key=sort_key, reverse=True)
    
    def _calculate_genre_relevance(self, metadata: ContentMetadata, genre: str) -> float:
        """Calculate how relevant content is to the genre"""
        relevance = 0.0
        
        # Direct genre match
        all_genres = [g.lower() for g in metadata.genres + metadata.anime_genres]
        if genre in all_genres:
            relevance += 1.0
        else:
            # Partial matches
            for g in all_genres:
                if genre in g or g in genre:
                    relevance += 0.5
        
        # Theme relevance
        theme_genre_mapping = {
            'action': ['action', 'adventure', 'power'],
            'romance': ['love', 'relationship'],
            'horror': ['fear', 'supernatural'],
            'comedy': ['humor', 'lighthearted'],
            'drama': ['emotional', 'character_driven'],
            'thriller': ['suspense', 'tension']
        }
        
        if genre in theme_genre_mapping:
            relevant_themes = theme_genre_mapping[genre]
            theme_matches = len(set(metadata.themes).intersection(set(relevant_themes)))
            relevance += theme_matches * 0.2
        
        return min(1.0, relevance)
    
    def _determine_match_type(self, metadata: ContentMetadata, genre: str) -> str:
        """Determine the type of genre match"""
        all_genres = [g.lower() for g in metadata.genres + metadata.anime_genres]
        
        if genre in all_genres:
            return 'direct_match'
        
        # Check related genres
        related_genres = self.genre_relationships.get(genre, [])
        if metadata.content_type == 'anime':
            related_genres.extend(self.anime_genre_relationships.get(genre, []))
        
        for related in related_genres:
            if related.lower() in all_genres:
                return 'related_genre'
        
        return 'theme_based'
    
    def _get_related_genres(self, genre: str, content_type: str) -> List[str]:
        """Get related genres for exploration"""
        related = []
        
        if genre in self.genre_relationships:
            related.extend(self.genre_relationships[genre])
        
        if content_type == 'anime' and genre in self.anime_genre_relationships:
            related.extend(self.anime_genre_relationships[genre])
        
        return list(set(related))
    
    def _analyze_language_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """Analyze language distribution in results"""
        language_counts = defaultdict(int)
        
        for result in results:
            for lang in result['metadata'].languages:
                if lang != 'unknown':
                    language_counts[lang] += 1
        
        return dict(language_counts)
    
    def _analyze_quality_stats(self, results: List[Dict]) -> Dict[str, float]:
        """Analyze quality statistics"""
        if not results:
            return {}
        
        ratings = [r['metadata'].rating for r in results if r['metadata'].rating > 0]
        
        if not ratings:
            return {}
        
        return {
            'average_rating': sum(ratings) / len(ratings),
            'min_rating': min(ratings),
            'max_rating': max(ratings),
            'total_with_ratings': len(ratings)
        }

# Service initialization function
def init_similarity_service(app, db, models, cache_backend=None):
    """Initialize the similarity service"""
    try:
        # Create similarity engine
        similarity_engine = AdvancedSimilarityEngine(db, models, cache_backend)
        
        # Create genre explorer
        genre_explorer = GenreExplorer(db, models, similarity_engine)
        
        logger.info("Similarity service initialized successfully")
        
        return {
            'similarity_engine': similarity_engine,
            'genre_explorer': genre_explorer
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize similarity service: {e}")
        return None

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor performance of similarity functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper

# Export main classes and functions
__all__ = [
    'AdvancedSimilarityEngine',
    'GenreExplorer', 
    'SimilarityScore',
    'SimilarityType',
    'LanguageFamily',
    'ContentMetadata',
    'init_similarity_service',
    'monitor_performance'
]