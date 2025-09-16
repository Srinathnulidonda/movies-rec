# backend/services/similar.py (Story-Centric Version)
import json
import logging
import time
import math
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import threading
import pickle
import hashlib

from sqlalchemy import and_, or_, func, text, desc
from sqlalchemy.orm import sessionmaker, joinedload
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import requests
import spacy

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except:
    logger.warning("NLTK download failed, using fallback methods")

class StoryElement(Enum):
    """Core story elements for analysis"""
    PLOT_STRUCTURE = "plot_structure"
    CHARACTER_ARCHETYPES = "character_archetypes"
    THEMES = "themes"
    EMOTIONAL_JOURNEY = "emotional_journey"
    CONFLICT_TYPE = "conflict_type"
    SETTING_CONTEXT = "setting_context"
    NARRATIVE_STYLE = "narrative_style"
    STORY_COMPLEXITY = "story_complexity"
    PACING = "pacing"
    TONE = "tone"
    MORAL_MESSAGE = "moral_message"
    STORY_BEATS = "story_beats"

class NarrativePattern(Enum):
    """Common narrative patterns and structures"""
    HEROS_JOURNEY = "heros_journey"
    THREE_ACT_STRUCTURE = "three_act_structure"
    REVENGE_ARC = "revenge_arc"
    LOVE_STORY = "love_story"
    COMING_OF_AGE = "coming_of_age"
    REDEMPTION = "redemption"
    FAMILY_SAGA = "family_saga"
    MYSTERY_INVESTIGATION = "mystery_investigation"
    SURVIVAL_STORY = "survival_story"
    POWER_STRUGGLE = "power_struggle"
    SACRIFICE_STORY = "sacrifice_story"
    TRANSFORMATION = "transformation"

@dataclass
class StoryAnalysis:
    """Comprehensive story analysis results"""
    content_id: int
    
    # Core story elements
    main_themes: Set[str] = field(default_factory=set)
    character_archetypes: Set[str] = field(default_factory=set)
    conflict_types: Set[str] = field(default_factory=set)
    emotional_beats: List[str] = field(default_factory=list)
    narrative_patterns: Set[NarrativePattern] = field(default_factory=set)
    
    # Story context
    setting_type: str = ""
    time_period: str = ""
    story_scope: str = ""  # personal, family, community, global
    
    # Emotional analysis
    emotional_tone: str = ""  # dark, light, bittersweet, uplifting
    emotional_intensity: float = 0.0
    emotional_complexity: float = 0.0
    
    # Narrative structure
    story_complexity_score: float = 0.0
    pacing_type: str = ""  # slow_burn, fast_paced, balanced
    narrative_style: str = ""  # linear, non_linear, multi_perspective
    
    # Content vectors
    plot_vector: Optional[np.ndarray] = None
    theme_vector: Optional[np.ndarray] = None
    character_vector: Optional[np.ndarray] = None
    dialogue_vector: Optional[np.ndarray] = None
    
    # Extracted keywords
    plot_keywords: Set[str] = field(default_factory=set)
    character_keywords: Set[str] = field(default_factory=set)
    emotion_keywords: Set[str] = field(default_factory=set)
    action_keywords: Set[str] = field(default_factory=set)

class AdvancedStoryAnalyzer:
    """Advanced story and content analysis engine"""
    
    def __init__(self):
        # Initialize NLP tools
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using NLTK fallback")
            self.nlp = None
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) if 'english' in stopwords.fileids() else set()
        
        # Story pattern definitions
        self.theme_patterns = {
            'love_romance': ['love', 'romance', 'relationship', 'marriage', 'wedding', 'couple', 'romantic', 'heart', 'passion', 'dating'],
            'family_bonds': ['family', 'father', 'mother', 'son', 'daughter', 'brother', 'sister', 'parent', 'child', 'siblings', 'generations'],
            'friendship': ['friend', 'friendship', 'loyal', 'companion', 'buddy', 'brotherhood', 'sisterhood', 'bond', 'trust'],
            'revenge_justice': ['revenge', 'vengeance', 'justice', 'payback', 'retribution', 'wronged', 'betrayal', 'enemy', 'villain'],
            'survival_struggle': ['survival', 'survive', 'escape', 'rescue', 'trapped', 'danger', 'threat', 'life_death', 'desperate'],
            'power_ambition': ['power', 'ambition', 'control', 'domination', 'empire', 'ruler', 'throne', 'politics', 'corruption'],
            'redemption': ['redemption', 'forgiveness', 'second_chance', 'atonement', 'guilt', 'regret', 'salvation', 'reformed'],
            'coming_of_age': ['growing_up', 'adolescent', 'teenager', 'youth', 'maturity', 'responsibility', 'adult', 'childhood'],
            'sacrifice_heroism': ['sacrifice', 'hero', 'heroic', 'brave', 'courage', 'selfless', 'duty', 'honor', 'noble'],
            'identity_discovery': ['identity', 'discovery', 'finding_self', 'purpose', 'destiny', 'calling', 'journey', 'awakening'],
            'social_issues': ['society', 'inequality', 'prejudice', 'discrimination', 'class', 'poverty', 'wealth', 'social_change'],
            'moral_dilemma': ['moral', 'ethics', 'right_wrong', 'choice', 'dilemma', 'conscience', 'values', 'principle']
        }
        
        self.character_archetypes = {
            'hero_protagonist': ['hero', 'protagonist', 'champion', 'savior', 'leader', 'warrior', 'defender'],
            'mentor_guide': ['mentor', 'teacher', 'guide', 'wise', 'elder', 'master', 'advisor', 'counselor'],
            'villain_antagonist': ['villain', 'antagonist', 'enemy', 'evil', 'corrupt', 'tyrant', 'criminal', 'monster'],
            'love_interest': ['love_interest', 'romantic', 'beloved', 'partner', 'soulmate', 'crush', 'attraction'],
            'comic_relief': ['comic', 'funny', 'humor', 'comedian', 'clown', 'joker', 'witty', 'amusing'],
            'sidekick_companion': ['sidekick', 'companion', 'friend', 'ally', 'partner', 'helper', 'supporter'],
            'victim_innocent': ['victim', 'innocent', 'vulnerable', 'helpless', 'pure', 'naive', 'defenseless'],
            'trickster': ['trickster', 'cunning', 'clever', 'mischievous', 'sly', 'deceiver', 'schemer']
        }
        
        self.conflict_types = {
            'man_vs_man': ['conflict', 'fight', 'battle', 'war', 'enemy', 'opponent', 'rivalry', 'competition'],
            'man_vs_self': ['internal', 'struggle', 'doubt', 'decision', 'guilt', 'fear', 'conscience', 'identity'],
            'man_vs_society': ['society', 'system', 'establishment', 'authority', 'government', 'law', 'tradition'],
            'man_vs_nature': ['nature', 'disaster', 'storm', 'survival', 'wilderness', 'elements', 'environment'],
            'man_vs_technology': ['technology', 'machine', 'artificial', 'robot', 'computer', 'digital', 'cyber'],
            'man_vs_supernatural': ['supernatural', 'ghost', 'spirit', 'magic', 'curse', 'demon', 'divine', 'mystical']
        }
        
        self.emotional_indicators = {
            'dark_intense': ['dark', 'intense', 'brutal', 'violent', 'tragic', 'devastating', 'horrific', 'gritty'],
            'uplifting_hopeful': ['uplifting', 'hopeful', 'inspiring', 'positive', 'optimistic', 'cheerful', 'joyful'],
            'bittersweet': ['bittersweet', 'melancholy', 'nostalgic', 'poignant', 'touching', 'emotional'],
            'suspenseful_thrilling': ['suspenseful', 'thrilling', 'tension', 'edge', 'gripping', 'nail_biting'],
            'funny_comedic': ['funny', 'comedic', 'hilarious', 'amusing', 'witty', 'humorous', 'satirical'],
            'romantic_passionate': ['romantic', 'passionate', 'tender', 'intimate', 'loving', 'sensual']
        }
        
        # Advanced vectorizers for different story aspects
        self.plot_vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        self.theme_vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            vocabulary=self._build_theme_vocabulary()
        )
        
        # Topic modeling for deeper analysis
        self.lda_model = LatentDirichletAllocation(
            n_components=20,
            random_state=42,
            max_iter=10
        )
        
        self._fitted = False
    
    def _build_theme_vocabulary(self) -> List[str]:
        """Build vocabulary focused on themes and story elements"""
        vocabulary = []
        for theme_words in self.theme_patterns.values():
            vocabulary.extend(theme_words)
        for archetype_words in self.character_archetypes.values():
            vocabulary.extend(archetype_words)
        for conflict_words in self.conflict_types.values():
            vocabulary.extend(conflict_words)
        return list(set(vocabulary))
    
    def fit_vectorizers(self, plot_texts: List[str]):
        """Fit vectorizers on the corpus"""
        try:
            if plot_texts:
                # Clean and preprocess texts
                cleaned_texts = [self._preprocess_text(text) for text in plot_texts if text]
                
                if cleaned_texts:
                    self.plot_vectorizer.fit(cleaned_texts)
                    
                    # Fit LDA model
                    plot_vectors = self.plot_vectorizer.transform(cleaned_texts)
                    self.lda_model.fit(plot_vectors)
                    
                    self._fitted = True
                    logger.info(f"Story analyzers fitted on {len(cleaned_texts)} plot texts")
                
        except Exception as e:
            logger.error(f"Error fitting story analyzers: {e}")
    
    def analyze_story(self, content_id: int, plot_text: str, title: str = "", genres: List[str] = None) -> StoryAnalysis:
        """Perform comprehensive story analysis"""
        
        analysis = StoryAnalysis(content_id=content_id)
        
        if not plot_text:
            return analysis
        
        try:
            # Preprocess text
            cleaned_text = self._preprocess_text(plot_text)
            
            # Extract themes
            analysis.main_themes = self._extract_themes(cleaned_text)
            
            # Extract character archetypes
            analysis.character_archetypes = self._extract_character_archetypes(cleaned_text)
            
            # Extract conflict types
            analysis.conflict_types = self._extract_conflict_types(cleaned_text)
            
            # Analyze emotional tone
            analysis.emotional_tone, analysis.emotional_intensity = self._analyze_emotional_tone(cleaned_text)
            
            # Extract narrative patterns
            analysis.narrative_patterns = self._identify_narrative_patterns(cleaned_text, title, genres)
            
            # Analyze story structure
            analysis.story_complexity_score = self._calculate_story_complexity(cleaned_text)
            analysis.pacing_type = self._analyze_pacing(cleaned_text)
            analysis.narrative_style = self._analyze_narrative_style(cleaned_text)
            
            # Extract setting and context
            analysis.setting_type, analysis.time_period = self._extract_setting_context(cleaned_text)
            analysis.story_scope = self._determine_story_scope(cleaned_text)
            
            # Generate vectors if fitted
            if self._fitted:
                analysis.plot_vector = self.plot_vectorizer.transform([cleaned_text]).toarray()[0]
                
                # Generate topic distribution
                plot_vec = self.plot_vectorizer.transform([cleaned_text])
                topic_dist = self.lda_model.transform(plot_vec)[0]
                analysis.theme_vector = topic_dist
            
            # Extract keywords
            analysis.plot_keywords = self._extract_plot_keywords(cleaned_text)
            analysis.character_keywords = self._extract_character_keywords(cleaned_text)
            analysis.emotion_keywords = self._extract_emotion_keywords(cleaned_text)
            analysis.action_keywords = self._extract_action_keywords(cleaned_text)
            
        except Exception as e:
            logger.error(f"Story analysis error for content {content_id}: {e}")
        
        return analysis
    
    def _preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing for story analysis"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and lemmatize if NLTK is available
        try:
            tokens = word_tokenize(text)
            # Remove stopwords but keep story-relevant words
            story_stopwords = self.stop_words - {
                'not', 'but', 'against', 'between', 'into', 'through', 'during', 'before', 'after'
            }
            tokens = [token for token in tokens if token not in story_stopwords and len(token) > 2]
            
            # Lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            return ' '.join(tokens)
        except:
            return text
    
    def _extract_themes(self, text: str) -> Set[str]:
        """Extract main themes from story text"""
        themes = set()
        text_words = set(text.split())
        
        for theme, keywords in self.theme_patterns.items():
            # Check for keyword matches
            matches = sum(1 for keyword in keywords if keyword in text_words)
            if matches >= 2:  # Require multiple matches for confidence
                themes.add(theme)
            elif matches == 1:
                # Single match needs context verification
                for keyword in keywords:
                    if keyword in text and self._verify_theme_context(text, keyword, theme):
                        themes.add(theme)
                        break
        
        return themes
    
    def _verify_theme_context(self, text: str, keyword: str, theme: str) -> bool:
        """Verify theme keyword appears in relevant context"""
        # Find sentences containing the keyword
        sentences = sent_tokenize(text)
        for sentence in sentences:
            if keyword in sentence.lower():
                # Check for supporting words in the same sentence
                supporting_words = self.theme_patterns[theme]
                sentence_words = set(sentence.lower().split())
                if len(sentence_words.intersection(supporting_words)) >= 2:
                    return True
        return False
    
    def _extract_character_archetypes(self, text: str) -> Set[str]:
        """Extract character archetypes present in the story"""
        archetypes = set()
        text_words = set(text.split())
        
        for archetype, keywords in self.character_archetypes.items():
            matches = sum(1 for keyword in keywords if keyword in text_words)
            if matches >= 1:
                archetypes.add(archetype)
        
        return archetypes
    
    def _extract_conflict_types(self, text: str) -> Set[str]:
        """Extract types of conflicts in the story"""
        conflicts = set()
        text_words = set(text.split())
        
        for conflict, keywords in self.conflict_types.items():
            matches = sum(1 for keyword in keywords if keyword in text_words)
            if matches >= 2:
                conflicts.add(conflict)
        
        return conflicts
    
    def _analyze_emotional_tone(self, text: str) -> Tuple[str, float]:
        """Analyze emotional tone and intensity"""
        text_words = set(text.split())
        emotion_scores = {}
        
        for emotion, keywords in self.emotional_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_words)
            if score > 0:
                emotion_scores[emotion] = score
        
        if not emotion_scores:
            return "neutral", 0.5
        
        # Get dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        
        # Calculate intensity based on keyword density
        total_words = len(text.split())
        intensity = min(1.0, sum(emotion_scores.values()) / max(total_words / 100, 1))
        
        return dominant_emotion, intensity
    
    def _identify_narrative_patterns(self, text: str, title: str = "", genres: List[str] = None) -> Set[NarrativePattern]:
        """Identify narrative patterns and story structures"""
        patterns = set()
        text_lower = text.lower()
        title_lower = title.lower() if title else ""
        genres_lower = [g.lower() for g in (genres or [])]
        
        # Hero's journey indicators
        hero_indicators = ['journey', 'quest', 'adventure', 'hero', 'destiny', 'call', 'mentor', 'return']
        if sum(1 for indicator in hero_indicators if indicator in text_lower) >= 3:
            patterns.add(NarrativePattern.HEROS_JOURNEY)
        
        # Revenge arc
        revenge_indicators = ['revenge', 'vengeance', 'payback', 'betrayal', 'wronged', 'justice']
        if sum(1 for indicator in revenge_indicators if indicator in text_lower) >= 2:
            patterns.add(NarrativePattern.REVENGE_ARC)
        
        # Love story
        love_indicators = ['love', 'romance', 'heart', 'marriage', 'couple', 'relationship']
        if (sum(1 for indicator in love_indicators if indicator in text_lower) >= 2 or
            'romance' in genres_lower):
            patterns.add(NarrativePattern.LOVE_STORY)
        
        # Coming of age
        coming_age_indicators = ['young', 'grow', 'adult', 'teenager', 'childhood', 'maturity']
        if sum(1 for indicator in coming_age_indicators if indicator in text_lower) >= 2:
            patterns.add(NarrativePattern.COMING_OF_AGE)
        
        # Family saga
        family_indicators = ['family', 'generation', 'father', 'mother', 'son', 'daughter', 'legacy']
        if sum(1 for indicator in family_indicators if indicator in text_lower) >= 3:
            patterns.add(NarrativePattern.FAMILY_SAGA)
        
        # Mystery investigation
        mystery_indicators = ['mystery', 'investigate', 'detective', 'clue', 'solve', 'crime', 'murder']
        if (sum(1 for indicator in mystery_indicators if indicator in text_lower) >= 2 or
            'mystery' in genres_lower or 'crime' in genres_lower):
            patterns.add(NarrativePattern.MYSTERY_INVESTIGATION)
        
        # Survival story
        survival_indicators = ['survive', 'survival', 'escape', 'trapped', 'rescue', 'danger']
        if sum(1 for indicator in survival_indicators if indicator in text_lower) >= 2:
            patterns.add(NarrativePattern.SURVIVAL_STORY)
        
        # Redemption
        redemption_indicators = ['redemption', 'forgive', 'second_chance', 'atonement', 'reformed']
        if sum(1 for indicator in redemption_indicators if indicator in text_lower) >= 2:
            patterns.add(NarrativePattern.REDEMPTION)
        
        return patterns
    
    def _calculate_story_complexity(self, text: str) -> float:
        """Calculate story complexity score"""
        complexity_score = 0.0
        
        # Sentence complexity
        sentences = sent_tokenize(text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        complexity_score += min(1.0, avg_sentence_length / 20)  # Normalize to 20 words
        
        # Vocabulary richness
        words = text.split()
        unique_words = set(words)
        vocabulary_richness = len(unique_words) / max(len(words), 1)
        complexity_score += vocabulary_richness
        
        # Character and plot element density
        plot_elements = len(self._extract_plot_keywords(text))
        character_elements = len(self._extract_character_keywords(text))
        element_density = (plot_elements + character_elements) / max(len(words) / 100, 1)
        complexity_score += min(1.0, element_density)
        
        return min(1.0, complexity_score / 3)
    
    def _analyze_pacing(self, text: str) -> str:
        """Analyze story pacing"""
        action_words = ['fight', 'run', 'chase', 'escape', 'battle', 'race', 'rush', 'quick', 'fast', 'sudden']
        slow_words = ['slow', 'gradual', 'peaceful', 'calm', 'quiet', 'gentle', 'steady', 'patient']
        
        action_count = sum(1 for word in action_words if word in text.lower())
        slow_count = sum(1 for word in slow_words if word in text.lower())
        
        if action_count > slow_count * 2:
            return "fast_paced"
        elif slow_count > action_count:
            return "slow_burn"
        else:
            return "balanced"
    
    def _analyze_narrative_style(self, text: str) -> str:
        """Analyze narrative style"""
        # Simple heuristics for narrative style
        time_indicators = ['then', 'next', 'after', 'before', 'meanwhile', 'suddenly', 'later']
        perspective_indicators = ['flashback', 'memory', 'perspective', 'viewpoint', 'told']
        
        time_count = sum(1 for indicator in time_indicators if indicator in text.lower())
        perspective_count = sum(1 for indicator in perspective_indicators if indicator in text.lower())
        
        if perspective_count > 2:
            return "multi_perspective"
        elif time_count > 3:
            return "non_linear"
        else:
            return "linear"
    
    def _extract_setting_context(self, text: str) -> Tuple[str, str]:
        """Extract setting type and time period"""
        # Setting types
        urban_words = ['city', 'urban', 'street', 'building', 'apartment', 'office']
        rural_words = ['village', 'farm', 'countryside', 'rural', 'forest', 'mountain']
        fantasy_words = ['kingdom', 'magic', 'castle', 'dragon', 'wizard', 'enchanted']
        sci_fi_words = ['space', 'future', 'robot', 'alien', 'technology', 'cyberpunk']
        
        setting_scores = {
            'urban': sum(1 for word in urban_words if word in text.lower()),
            'rural': sum(1 for word in rural_words if word in text.lower()),
            'fantasy': sum(1 for word in fantasy_words if word in text.lower()),
            'sci_fi': sum(1 for word in sci_fi_words if word in text.lower())
        }
        
        setting_type = max(setting_scores, key=setting_scores.get) if max(setting_scores.values()) > 0 else "contemporary"
        
        # Time period indicators
        historical_words = ['historical', 'ancient', 'medieval', 'century', 'era', 'period']
        future_words = ['future', 'futuristic', 'tomorrow', 'advanced', 'next_generation']
        
        if any(word in text.lower() for word in historical_words):
            time_period = "historical"
        elif any(word in text.lower() for word in future_words):
            time_period = "future"
        else:
            time_period = "contemporary"
        
        return setting_type, time_period
    
    def _determine_story_scope(self, text: str) -> str:
        """Determine the scope of the story"""
        personal_words = ['personal', 'individual', 'self', 'inner', 'private']
        family_words = ['family', 'household', 'relatives', 'parents', 'children']
        community_words = ['community', 'town', 'neighborhood', 'local', 'village']
        global_words = ['world', 'global', 'universal', 'humanity', 'civilization', 'planet']
        
        scope_scores = {
            'personal': sum(1 for word in personal_words if word in text.lower()),
            'family': sum(1 for word in family_words if word in text.lower()),
            'community': sum(1 for word in community_words if word in text.lower()),
            'global': sum(1 for word in global_words if word in text.lower())
        }
        
        return max(scope_scores, key=scope_scores.get) if max(scope_scores.values()) > 0 else "personal"
    
    def _extract_plot_keywords(self, text: str) -> Set[str]:
        """Extract important plot keywords"""
        # Extract nouns and verbs that indicate plot elements
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            plot_keywords = set()
            for word, pos in pos_tags:
                if (pos.startswith('NN') or pos.startswith('VB')) and len(word) > 3:
                    plot_keywords.add(word.lower())
            
            return plot_keywords
        except:
            # Fallback method
            words = text.split()
            return {word for word in words if len(word) > 4}
    
    def _extract_character_keywords(self, text: str) -> Set[str]:
        """Extract character-related keywords"""
        character_words = {'character', 'person', 'man', 'woman', 'boy', 'girl', 'hero', 'villain', 'friend', 'enemy'}
        text_words = set(text.lower().split())
        return text_words.intersection(character_words)
    
    def _extract_emotion_keywords(self, text: str) -> Set[str]:
        """Extract emotion-related keywords"""
        emotion_words = {
            'love', 'hate', 'anger', 'joy', 'sadness', 'fear', 'hope', 'despair',
            'happiness', 'sorrow', 'excitement', 'anxiety', 'peace', 'rage'
        }
        text_words = set(text.lower().split())
        return text_words.intersection(emotion_words)
    
    def _extract_action_keywords(self, text: str) -> Set[str]:
        """Extract action-related keywords"""
        action_words = {
            'fight', 'battle', 'run', 'chase', 'escape', 'rescue', 'attack', 'defend',
            'journey', 'travel', 'discover', 'explore', 'search', 'find', 'create', 'destroy'
        }
        text_words = set(text.lower().split())
        return text_words.intersection(action_words)

class StoryBasedSimilarityCalculator:
    """Story-centric similarity calculator - Main focus on content and narrative"""
    
    def __init__(self, db, models, cache=None):
        self.db = db
        self.models = models
        self.cache = cache
        self.story_analyzer = AdvancedStoryAnalyzer()
        self.story_analyses = {}  # Cache for story analyses
        
        # Initialize story analyzer with existing content
        self._initialize_story_corpus()
    
    def _initialize_story_corpus(self):
        """Initialize story analyzer with existing plot texts"""
        try:
            Content = self.models['Content']
            contents = Content.query.filter(
                Content.overview.isnot(None),
                func.length(Content.overview) > 50
            ).limit(1000).all()
            
            plot_texts = [content.overview for content in contents if content.overview]
            
            if plot_texts:
                self.story_analyzer.fit_vectorizers(plot_texts)
                logger.info(f"Story analyzer initialized with {len(plot_texts)} plots")
            
        except Exception as e:
            logger.error(f"Story corpus initialization error: {e}")
    
    def get_story_analysis(self, content_id: int) -> StoryAnalysis:
        """Get or create story analysis for content"""
        
        # Check cache first
        if content_id in self.story_analyses:
            return self.story_analyses[content_id]
        
        try:
            Content = self.models['Content']
            content = Content.query.get(content_id)
            
            if not content or not content.overview:
                return StoryAnalysis(content_id=content_id)
            
            # Parse genres
            genres = []
            if content.genres:
                try:
                    genres = json.loads(content.genres)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Analyze story
            analysis = self.story_analyzer.analyze_story(
                content_id=content_id,
                plot_text=content.overview,
                title=content.title,
                genres=genres
            )
            
            # Cache the analysis
            self.story_analyses[content_id] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Story analysis error for content {content_id}: {e}")
            return StoryAnalysis(content_id=content_id)
    
    def calculate_story_similarity(
        self, 
        source_content_id: int, 
        target_content_id: int
    ) -> Dict[str, float]:
        """Calculate comprehensive story similarity between two content items"""
        
        source_analysis = self.get_story_analysis(source_content_id)
        target_analysis = self.get_story_analysis(target_content_id)
        
        similarities = {}
        
        # 1. Theme Similarity (25% weight)
        theme_similarity = self._calculate_theme_similarity(source_analysis, target_analysis)
        similarities['theme_similarity'] = theme_similarity
        
        # 2. Narrative Pattern Similarity (20% weight)
        pattern_similarity = self._calculate_pattern_similarity(source_analysis, target_analysis)
        similarities['pattern_similarity'] = pattern_similarity
        
        # 3. Character Archetype Similarity (15% weight)
        character_similarity = self._calculate_character_similarity(source_analysis, target_analysis)
        similarities['character_similarity'] = character_similarity
        
        # 4. Emotional Journey Similarity (15% weight)
        emotional_similarity = self._calculate_emotional_similarity(source_analysis, target_analysis)
        similarities['emotional_similarity'] = emotional_similarity
        
        # 5. Conflict Type Similarity (10% weight)
        conflict_similarity = self._calculate_conflict_similarity(source_analysis, target_analysis)
        similarities['conflict_similarity'] = conflict_similarity
        
        # 6. Plot Vector Similarity (10% weight) - Deep content analysis
        plot_vector_similarity = self._calculate_plot_vector_similarity(source_analysis, target_analysis)
        similarities['plot_vector_similarity'] = plot_vector_similarity
        
        # 7. Story Structure Similarity (5% weight)
        structure_similarity = self._calculate_structure_similarity(source_analysis, target_analysis)
        similarities['structure_similarity'] = structure_similarity
        
        # Calculate weighted total score (Story-focused weights)
        total_story_score = (
            theme_similarity * 0.25 +
            pattern_similarity * 0.20 +
            character_similarity * 0.15 +
            emotional_similarity * 0.15 +
            conflict_similarity * 0.10 +
            plot_vector_similarity * 0.10 +
            structure_similarity * 0.05
        )
        
        similarities['total_story_score'] = total_story_score
        
        return similarities
    
    def _calculate_theme_similarity(self, source: StoryAnalysis, target: StoryAnalysis) -> float:
        """Calculate theme similarity - Most important factor"""
        
        if not source.main_themes or not target.main_themes:
            return 0.0
        
        # Exact theme matches
        common_themes = source.main_themes.intersection(target.main_themes)
        total_themes = source.main_themes.union(target.main_themes)
        
        exact_similarity = len(common_themes) / len(total_themes) if total_themes else 0.0
        
        # Theme vector similarity if available
        vector_similarity = 0.0
        if source.theme_vector is not None and target.theme_vector is not None:
            vector_similarity = cosine_similarity(
                source.theme_vector.reshape(1, -1),
                target.theme_vector.reshape(1, -1)
            )[0][0]
        
        # Combine exact and vector similarities
        return max(exact_similarity, vector_similarity)
    
    def _calculate_pattern_similarity(self, source: StoryAnalysis, target: StoryAnalysis) -> float:
        """Calculate narrative pattern similarity"""
        
        if not source.narrative_patterns or not target.narrative_patterns:
            return 0.0
        
        common_patterns = source.narrative_patterns.intersection(target.narrative_patterns)
        total_patterns = source.narrative_patterns.union(target.narrative_patterns)
        
        return len(common_patterns) / len(total_patterns) if total_patterns else 0.0
    
    def _calculate_character_similarity(self, source: StoryAnalysis, target: StoryAnalysis) -> float:
        """Calculate character archetype similarity"""
        
        if not source.character_archetypes or not target.character_archetypes:
            return 0.0
        
        common_archetypes = source.character_archetypes.intersection(target.character_archetypes)
        total_archetypes = source.character_archetypes.union(target.character_archetypes)
        
        return len(common_archetypes) / len(total_archetypes) if total_archetypes else 0.0
    
    def _calculate_emotional_similarity(self, source: StoryAnalysis, target: StoryAnalysis) -> float:
        """Calculate emotional journey and tone similarity"""
        
        emotional_score = 0.0
        
        # Emotional tone similarity
        if source.emotional_tone == target.emotional_tone:
            emotional_score += 0.5
        
        # Emotional intensity similarity
        if source.emotional_intensity > 0 and target.emotional_intensity > 0:
            intensity_diff = abs(source.emotional_intensity - target.emotional_intensity)
            intensity_similarity = max(0, 1 - intensity_diff)
            emotional_score += intensity_similarity * 0.3
        
        # Emotion keywords similarity
        if source.emotion_keywords and target.emotion_keywords:
            common_emotions = source.emotion_keywords.intersection(target.emotion_keywords)
            total_emotions = source.emotion_keywords.union(target.emotion_keywords)
            emotion_keyword_similarity = len(common_emotions) / len(total_emotions) if total_emotions else 0.0
            emotional_score += emotion_keyword_similarity * 0.2
        
        return min(1.0, emotional_score)
    
    def _calculate_conflict_similarity(self, source: StoryAnalysis, target: StoryAnalysis) -> float:
        """Calculate conflict type similarity"""
        
        if not source.conflict_types or not target.conflict_types:
            return 0.0
        
        common_conflicts = source.conflict_types.intersection(target.conflict_types)
        total_conflicts = source.conflict_types.union(target.conflict_types)
        
        return len(common_conflicts) / len(total_conflicts) if total_conflicts else 0.0
    
    def _calculate_plot_vector_similarity(self, source: StoryAnalysis, target: StoryAnalysis) -> float:
        """Calculate deep plot content similarity using vectors"""
        
        if source.plot_vector is None or target.plot_vector is None:
            # Fallback to keyword similarity
            return self._calculate_keyword_similarity(source, target)
        
        try:
            similarity = cosine_similarity(
                source.plot_vector.reshape(1, -1),
                target.plot_vector.reshape(1, -1)
            )[0][0]
            
            return max(0.0, similarity)
        except Exception as e:
            logger.warning(f"Plot vector similarity calculation error: {e}")
            return self._calculate_keyword_similarity(source, target)
    
    def _calculate_keyword_similarity(self, source: StoryAnalysis, target: StoryAnalysis) -> float:
        """Fallback keyword similarity calculation"""
        
        # Plot keywords similarity
        plot_similarity = 0.0
        if source.plot_keywords and target.plot_keywords:
            common_plot = source.plot_keywords.intersection(target.plot_keywords)
            total_plot = source.plot_keywords.union(target.plot_keywords)
            plot_similarity = len(common_plot) / len(total_plot) if total_plot else 0.0
        
        # Action keywords similarity
        action_similarity = 0.0
        if source.action_keywords and target.action_keywords:
            common_action = source.action_keywords.intersection(target.action_keywords)
            total_action = source.action_keywords.union(target.action_keywords)
            action_similarity = len(common_action) / len(total_action) if total_action else 0.0
        
        return (plot_similarity + action_similarity) / 2
    
    def _calculate_structure_similarity(self, source: StoryAnalysis, target: StoryAnalysis) -> float:
        """Calculate story structure similarity"""
        
        structure_score = 0.0
        
        # Complexity similarity
        if source.story_complexity_score > 0 and target.story_complexity_score > 0:
            complexity_diff = abs(source.story_complexity_score - target.story_complexity_score)
            structure_score += max(0, 1 - complexity_diff) * 0.4
        
        # Pacing similarity
        if source.pacing_type == target.pacing_type:
            structure_score += 0.3
        
        # Narrative style similarity
        if source.narrative_style == target.narrative_style:
            structure_score += 0.3
        
        return structure_score

@dataclass
class StoryBasedSimilarityScore:
    """Story-focused similarity score with detailed breakdown"""
    content_id: int
    total_story_score: float
    theme_similarity: float = 0.0
    pattern_similarity: float = 0.0
    character_similarity: float = 0.0
    emotional_similarity: float = 0.0
    conflict_similarity: float = 0.0
    plot_content_similarity: float = 0.0
    structure_similarity: float = 0.0
    
    # Secondary factors (much lower weight)
    language_bonus: float = 0.0
    genre_bonus: float = 0.0
    quality_bonus: float = 0.0
    
    # Story insights
    shared_themes: List[str] = field(default_factory=list)
    shared_patterns: List[str] = field(default_factory=list)
    story_match_reasons: List[str] = field(default_factory=list)
    story_confidence: float = 0.0
    
    def __post_init__(self):
        """Calculate story confidence and insights"""
        # Story confidence based on multiple story signals
        story_signals = [
            self.theme_similarity, self.pattern_similarity, 
            self.character_similarity, self.emotional_similarity,
            self.plot_content_similarity
        ]
        
        strong_signals = sum(1 for signal in story_signals if signal > 0.5)
        self.story_confidence = min(100, strong_signals * 25)

class UltimateStoryBasedSimilarityService:
    """Ultimate story-focused similarity service - Content is King"""
    
    def __init__(self, app, db, models, cache=None):
        self.app = app
        self.db = db
        self.models = models
        self.cache = cache
        self.story_calculator = StoryBasedSimilarityCalculator(db, models, cache)
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'story_analysis_cache_hits': 0,
            'high_confidence_recommendations': 0,
            'average_story_confidence': 0.0
        }
    
    def get_story_based_similar_content(
        self,
        content_id: int,
        limit: int = 10,
        min_story_similarity: float = 0.3,
        user_id: Optional[int] = None,
        include_story_analysis: bool = False
    ) -> Dict[str, Any]:
        """Get similar content based primarily on story and content analysis"""
        
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # Get base content
            Content = self.models['Content']
            base_content = Content.query.get(content_id)
            if not base_content:
                raise ValueError(f"Content {content_id} not found")
            
            # Check cache
            cache_key = f"story_similar_v3:{content_id}:{limit}:{min_story_similarity}"
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    return cached_result
            
            # Get candidate content (same type, with plots)
            candidates = Content.query.filter(
                Content.id != content_id,
                Content.content_type == base_content.content_type,
                Content.overview.isnot(None),
                func.length(Content.overview) > 50
            ).limit(200).all()  # Pre-filter for performance
            
            if not candidates:
                return self._empty_result(content_id, "No candidates with sufficient plot information")
            
            # Calculate story similarities
            similarities = []
            
            for candidate in candidates:
                try:
                    # Calculate comprehensive story similarity
                    story_similarities = self.story_calculator.calculate_story_similarity(
                        content_id, candidate.id
                    )
                    
                    total_story_score = story_similarities['total_story_score']
                    
                    # Only include if meets minimum story similarity
                    if total_story_score >= min_story_similarity:
                        # Add small bonuses for secondary factors (max 10% boost)
                        language_bonus = self._calculate_language_bonus(base_content, candidate) * 0.05
                        genre_bonus = self._calculate_genre_bonus(base_content, candidate) * 0.03
                        quality_bonus = self._calculate_quality_bonus(base_content, candidate) * 0.02
                        
                        # Final score is story-dominated
                        final_score = min(1.0, total_story_score + language_bonus + genre_bonus + quality_bonus)
                        
                        # Create detailed similarity score
                        similarity_score = StoryBasedSimilarityScore(
                            content_id=candidate.id,
                            total_story_score=final_score,
                            theme_similarity=story_similarities['theme_similarity'],
                            pattern_similarity=story_similarities['pattern_similarity'],
                            character_similarity=story_similarities['character_similarity'],
                            emotional_similarity=story_similarities['emotional_similarity'],
                            conflict_similarity=story_similarities['conflict_similarity'],
                            plot_content_similarity=story_similarities['plot_vector_similarity'],
                            structure_similarity=story_similarities['structure_similarity'],
                            language_bonus=language_bonus,
                            genre_bonus=genre_bonus,
                            quality_bonus=quality_bonus
                        )
                        
                        # Generate story insights
                        self._add_story_insights(similarity_score, content_id, candidate.id)
                        
                        similarities.append(similarity_score)
                        
                except Exception as e:
                    logger.warning(f"Error calculating similarity for candidate {candidate.id}: {e}")
                    continue
            
            # Sort by story score (primary) and confidence (secondary)
            similarities.sort(key=lambda x: (x.total_story_score, x.story_confidence), reverse=True)
            
            # Take top results
            top_similarities = similarities[:limit]
            
            # Format results with detailed story information
            results = []
            for sim_score in top_similarities:
                content = Content.query.get(sim_score.content_id)
                if content:
                    # Ensure slug
                    if not content.slug:
                        content.slug = f"content-{content.id}"
                    
                    result_item = {
                        'id': content.id,
                        'slug': content.slug,
                        'title': content.title,
                        'content_type': content.content_type,
                        'genres': json.loads(content.genres or '[]'),
                        'languages': json.loads(content.languages or '[]'),
                        'rating': content.rating,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                        'overview': content.overview[:200] + '...' if content.overview else '',
                        'youtube_trailer': f"https://www.youtube.com/watch?v={content.youtube_trailer_id}" if content.youtube_trailer_id else None,
                        
                        # Story-focused similarity metrics
                        'story_similarity_score': round(sim_score.total_story_score, 4),
                        'story_confidence': round(sim_score.story_confidence, 1),
                        
                        # Detailed story breakdown
                        'story_analysis': {
                            'theme_match': round(sim_score.theme_similarity, 3),
                            'narrative_pattern_match': round(sim_score.pattern_similarity, 3),
                            'character_similarity': round(sim_score.character_similarity, 3),
                            'emotional_similarity': round(sim_score.emotional_similarity, 3),
                            'conflict_similarity': round(sim_score.conflict_similarity, 3),
                            'plot_content_match': round(sim_score.plot_content_similarity, 3),
                            'story_structure_match': round(sim_score.structure_similarity, 3)
                        },
                        
                        'story_match_reasons': sim_score.story_match_reasons,
                        'shared_themes': sim_score.shared_themes,
                        'shared_patterns': sim_score.shared_patterns,
                        
                        # Secondary factors (much smaller influence)
                        'secondary_factors': {
                            'language_bonus': round(sim_score.language_bonus, 3),
                            'genre_bonus': round(sim_score.genre_bonus, 3),
                            'quality_bonus': round(sim_score.quality_bonus, 3)
                        }
                    }
                    
                    # Include detailed story analysis if requested
                    if include_story_analysis:
                        target_analysis = self.story_calculator.get_story_analysis(content.id)
                        result_item['detailed_story_analysis'] = {
                            'themes': list(target_analysis.main_themes),
                            'narrative_patterns': [p.value for p in target_analysis.narrative_patterns],
                            'character_archetypes': list(target_analysis.character_archetypes),
                            'emotional_tone': target_analysis.emotional_tone,
                            'emotional_intensity': target_analysis.emotional_intensity,
                            'story_complexity': target_analysis.story_complexity_score,
                            'pacing_type': target_analysis.pacing_type,
                            'setting_type': target_analysis.setting_type,
                            'story_scope': target_analysis.story_scope
                        }
                    
                    results.append(result_item)
            
            # Calculate performance metrics
            high_confidence = sum(1 for r in results if r['story_confidence'] > 75)
            self.stats['high_confidence_recommendations'] += high_confidence
            
            if results:
                avg_confidence = sum(r['story_confidence'] for r in results) / len(results)
                self.stats['average_story_confidence'] = (
                    (self.stats['average_story_confidence'] * (self.stats['total_requests'] - 1) + avg_confidence) /
                    self.stats['total_requests']
                )
            
            # Get base content story analysis for context
            base_story_analysis = self.story_calculator.get_story_analysis(content_id)
            
            response = {
                'base_content': {
                    'id': base_content.id,
                    'slug': base_content.slug or f"content-{base_content.id}",
                    'title': base_content.title,
                    'content_type': base_content.content_type,
                    'overview': base_content.overview,
                    'story_profile': {
                        'main_themes': list(base_story_analysis.main_themes),
                        'narrative_patterns': [p.value for p in base_story_analysis.narrative_patterns],
                        'emotional_tone': base_story_analysis.emotional_tone,
                        'story_complexity': base_story_analysis.story_complexity_score
                    }
                },
                'similar_content': results,
                'algorithm_info': {
                    'primary_focus': 'story_and_content_analysis',
                    'story_weight': '90%',
                    'secondary_factors_weight': '10%',
                    'analysis_depth': 'comprehensive',
                    'similarity_factors': [
                        'theme_analysis', 'narrative_patterns', 'character_archetypes',
                        'emotional_journey', 'conflict_types', 'plot_content',
                        'story_structure', 'pacing', 'tone'
                    ]
                },
                'metadata': {
                    'total_results': len(results),
                    'candidates_analyzed': len(candidates),
                    'story_similarity_threshold': min_story_similarity,
                    'high_confidence_results': high_confidence,
                    'average_story_confidence': round(avg_confidence, 1) if results else 0,
                    'response_time_ms': round((time.time() - start_time) * 1000, 2),
                    'algorithm_version': '3.0_story_centric',
                    'cache_used': False,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            # Cache the result
            if self.cache:
                self.cache.set(cache_key, response, timeout=3600)  # 1 hour cache
            
            return response
            
        except Exception as e:
            logger.error(f"Story-based similarity service error: {e}")
            return self._empty_result(content_id, str(e))
    
    def _calculate_language_bonus(self, base_content, target_content) -> float:
        """Small bonus for language match (max 5% of total score)"""
        try:
            base_langs = set(json.loads(base_content.languages or '[]'))
            target_langs = set(json.loads(target_content.languages or '[]'))
            
            if base_langs.intersection(target_langs):
                return 1.0
            
            # Language family bonus
            dravidian = {'telugu', 'tamil', 'kannada', 'malayalam'}
            if (base_langs.intersection(dravidian) and target_langs.intersection(dravidian)):
                return 0.6
            
            return 0.0
        except:
            return 0.0
    
    def _calculate_genre_bonus(self, base_content, target_content) -> float:
        """Small bonus for genre match (max 3% of total score)"""
        try:
            base_genres = set(json.loads(base_content.genres or '[]'))
            target_genres = set(json.loads(target_content.genres or '[]'))
            
            if not base_genres or not target_genres:
                return 0.0
            
            overlap = len(base_genres.intersection(target_genres))
            total = len(base_genres.union(target_genres))
            
            return overlap / total if total > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_quality_bonus(self, base_content, target_content) -> float:
        """Small bonus for similar quality (max 2% of total score)"""
        try:
            if not base_content.rating or not target_content.rating:
                return 0.0
            
            rating_diff = abs(base_content.rating - target_content.rating)
            return max(0, 1 - (rating_diff / 10))
        except:
            return 0.0
    
    def _add_story_insights(self, similarity_score: StoryBasedSimilarityScore, source_id: int, target_id: int):
        """Add detailed story insights and match reasons"""
        
        source_analysis = self.story_calculator.get_story_analysis(source_id)
        target_analysis = self.story_calculator.get_story_analysis(target_id)
        
        # Shared themes
        similarity_score.shared_themes = list(source_analysis.main_themes.intersection(target_analysis.main_themes))
        
        # Shared patterns
        shared_pattern_values = source_analysis.narrative_patterns.intersection(target_analysis.narrative_patterns)
        similarity_score.shared_patterns = [p.value for p in shared_pattern_values]
        
        # Generate story match reasons
        reasons = []
        
        if similarity_score.theme_similarity > 0.7:
            reasons.append(f"Very similar themes: {', '.join(similarity_score.shared_themes[:3])}")
        elif similarity_score.theme_similarity > 0.4:
            reasons.append("Similar thematic elements")
        
        if similarity_score.pattern_similarity > 0.5:
            reasons.append(f"Same narrative structure: {', '.join(similarity_score.shared_patterns[:2])}")
        
        if similarity_score.emotional_similarity > 0.6:
            if source_analysis.emotional_tone == target_analysis.emotional_tone:
                reasons.append(f"Matching emotional tone: {source_analysis.emotional_tone}")
        
        if similarity_score.character_similarity > 0.5:
            common_archetypes = source_analysis.character_archetypes.intersection(target_analysis.character_archetypes)
            if common_archetypes:
                reasons.append(f"Similar character types: {', '.join(list(common_archetypes)[:2])}")
        
        if similarity_score.plot_content_similarity > 0.5:
            reasons.append("Similar plot elements and storytelling")
        
        if similarity_score.conflict_similarity > 0.5:
            common_conflicts = source_analysis.conflict_types.intersection(target_analysis.conflict_types)
            if common_conflicts:
                reasons.append(f"Same conflict types: {', '.join(list(common_conflicts))}")
        
        if not reasons:
            reasons.append("Similar storytelling approach and content")
        
        similarity_score.story_match_reasons = reasons
    
    def _empty_result(self, content_id: int, reason: str) -> Dict[str, Any]:
        """Return empty result with error info"""
        return {
            'base_content': {'id': content_id},
            'similar_content': [],
            'error': reason,
            'algorithm_info': {
                'primary_focus': 'story_and_content_analysis',
                'story_weight': '90%'
            },
            'metadata': {
                'total_results': 0,
                'algorithm_version': '3.0_story_centric',
                'timestamp': datetime.utcnow().isoformat()
            }
        }
    
    def get_service_performance(self) -> Dict[str, Any]:
        """Get service performance statistics"""
        return {
            'performance_stats': self.stats,
            'story_focus_percentage': 90,
            'high_confidence_rate': (
                self.stats['high_confidence_recommendations'] / max(self.stats['total_requests'] * 10, 1) * 100
            ),
            'service_version': '3.0_story_centric',
            'primary_algorithms': [
                'theme_analysis', 'narrative_pattern_matching', 'emotional_journey_analysis',
                'character_archetype_matching', 'plot_content_vectorization',
                'conflict_type_analysis', 'story_structure_analysis'
            ]
        }

def init_story_based_similarity_service(app, db, models, cache=None):
    """Initialize the ultimate story-based similarity service"""
    try:
        service = UltimateStoryBasedSimilarityService(app, db, models, cache)
        logger.info("Ultimate story-based similarity service initialized successfully")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize story-based similarity service: {e}")
        return None