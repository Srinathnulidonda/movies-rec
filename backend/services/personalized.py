# backend/services/personalized.py
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import pandas as pd
import json
import logging
import jwt
import math
import random
import heapq
from functools import wraps, lru_cache
import hashlib
from sqlalchemy import func, and_, or_, desc, text, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import joinedload, relationship
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
from typing import Dict, List, Tuple, Optional, Any, Set
import re
from itertools import combinations
import networkx as nx
from textblob import TextBlob
import spacy
import warnings
warnings.filterwarnings('ignore')

# Load spaCy model for NLP
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Create personalized blueprint
personalized_bp = Blueprint('personalized', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Global variables - will be initialized by main app
db = None
cache = None
User = None
Content = None
UserInteraction = None
AnonymousInteraction = None
ContentPerson = None
Person = None
Review = None
UserPreference = None
RecommendationFeedback = None
UserSession = None
ContentSimilarity = None
StoryProfile = None
UserStoryPreference = None
app = None
services = None

def init_personalized(flask_app, database, models, app_services, app_cache):
    """Initialize personalized module with app context and models"""
    global db, cache, User, Content, UserInteraction, AnonymousInteraction
    global ContentPerson, Person, Review, UserPreference, RecommendationFeedback
    global UserSession, ContentSimilarity, StoryProfile, UserStoryPreference, app, services
    
    app = flask_app
    db = database
    cache = app_cache
    services = app_services
    
    # Initialize existing models
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    AnonymousInteraction = models['AnonymousInteraction']
    ContentPerson = models.get('ContentPerson')
    Person = models.get('Person')
    Review = models.get('Review')
    
    # Create enhanced models if they don't exist
    if 'UserPreference' not in models:
        UserPreference = create_user_preference_model(db)
        models['UserPreference'] = UserPreference
    else:
        UserPreference = models['UserPreference']
    
    if 'RecommendationFeedback' not in models:
        RecommendationFeedback = create_recommendation_feedback_model(db)
        models['RecommendationFeedback'] = RecommendationFeedback
    else:
        RecommendationFeedback = models['RecommendationFeedback']
    
    if 'UserSession' not in models:
        UserSession = create_user_session_model(db)
        models['UserSession'] = UserSession
    else:
        UserSession = models['UserSession']
    
    if 'ContentSimilarity' not in models:
        ContentSimilarity = create_content_similarity_model(db)
        models['ContentSimilarity'] = ContentSimilarity
    else:
        ContentSimilarity = models['ContentSimilarity']
    
    if 'StoryProfile' not in models:
        StoryProfile = create_story_profile_model(db)
        models['StoryProfile'] = StoryProfile
    else:
        StoryProfile = models['StoryProfile']
    
    if 'UserStoryPreference' not in models:
        UserStoryPreference = create_user_story_preference_model(db)
        models['UserStoryPreference'] = UserStoryPreference
    else:
        UserStoryPreference = models['UserStoryPreference']
    
    # Create tables if they don't exist
    with flask_app.app_context():
        db.create_all()

def create_user_preference_model(db):
    """Create enhanced UserPreference model"""
    
    class UserPreference(db.Model):
        __tablename__ = 'user_preferences'
        
        id = db.Column(db.Integer, primary_key=True)
        user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
        
        # Core preferences
        genre_preferences = db.Column(db.Text)  # JSON: detailed genre weights
        language_preferences = db.Column(db.Text)
        content_type_preferences = db.Column(db.Text)
        quality_preferences = db.Column(db.Text)
        runtime_preferences = db.Column(db.Text)
        
        # Story preferences
        story_themes = db.Column(db.Text)  # JSON: preferred themes
        narrative_styles = db.Column(db.Text)  # JSON: preferred narrative styles
        emotional_tones = db.Column(db.Text)  # JSON: preferred emotional tones
        plot_complexity = db.Column(db.Float, default=0.5)  # 0-1 scale
        character_depth = db.Column(db.Float, default=0.5)  # 0-1 scale
        
        # Advanced patterns
        viewing_patterns = db.Column(db.Text)
        search_patterns = db.Column(db.Text)
        sequence_patterns = db.Column(db.Text)
        time_patterns = db.Column(db.Text)  # When user watches what
        mood_patterns = db.Column(db.Text)  # Mood-based preferences
        
        # Actor/Director preferences
        cast_crew_preferences = db.Column(db.Text)
        director_preferences = db.Column(db.Text)
        studio_preferences = db.Column(db.Text)
        franchise_preferences = db.Column(db.Text)
        
        # Behavioral metrics
        exploration_tendency = db.Column(db.Float, default=0.5)
        diversity_preference = db.Column(db.Float, default=0.5)
        recency_bias = db.Column(db.Float, default=0.5)
        quality_threshold = db.Column(db.Float, default=6.0)
        binge_tendency = db.Column(db.Float, default=0.5)
        
        # Profile metadata
        profile_strength = db.Column(db.Float, default=0.0)
        confidence_score = db.Column(db.Float, default=0.0)
        personalization_level = db.Column(db.Integer, default=1)  # 1-5 levels
        
        # User state
        current_mode = db.Column(db.String(50), default='discovery_mode')
        current_mood = db.Column(db.String(50))  # happy, sad, excited, relaxed, etc.
        
        # Timestamps
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        last_calculated = db.Column(db.DateTime, default=datetime.utcnow)
        
    return UserPreference

def create_recommendation_feedback_model(db):
    """Create enhanced RecommendationFeedback model"""
    
    class RecommendationFeedback(db.Model):
        __tablename__ = 'recommendation_feedback'
        
        id = db.Column(db.Integer, primary_key=True)
        user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
        content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
        
        # Recommendation details
        recommendation_score = db.Column(db.Float)
        recommendation_method = db.Column(db.String(100))
        recommendation_reason = db.Column(db.Text)
        recommendation_rank = db.Column(db.Integer)
        story_match_score = db.Column(db.Float)  # How well story matched
        
        # User feedback
        feedback_type = db.Column(db.String(50))
        user_rating = db.Column(db.Float)
        watch_duration = db.Column(db.Integer)
        completion_rate = db.Column(db.Float)
        rewatched = db.Column(db.Boolean, default=False)
        
        # Engagement metrics
        engagement_score = db.Column(db.Float)
        satisfaction_score = db.Column(db.Float)
        
        # Context
        device_type = db.Column(db.String(50))
        time_of_day = db.Column(db.Integer)
        day_of_week = db.Column(db.Integer)
        user_mode = db.Column(db.String(50))
        user_mood = db.Column(db.String(50))
        
        # Success metrics
        was_successful = db.Column(db.Boolean, default=False)
        led_to_similar = db.Column(db.Boolean, default=False)  # Led to watching similar content
        
        # Timestamps
        recommended_at = db.Column(db.DateTime, default=datetime.utcnow)
        feedback_at = db.Column(db.DateTime)
        
        __table_args__ = (
            db.Index('idx_user_feedback', 'user_id', 'feedback_at'),
            db.Index('idx_method_performance', 'recommendation_method', 'was_successful'),
        )
        
    return RecommendationFeedback

def create_user_session_model(db):
    """Create UserSession model for tracking sessions"""
    
    class UserSession(db.Model):
        __tablename__ = 'user_sessions'
        
        id = db.Column(db.Integer, primary_key=True)
        user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
        session_id = db.Column(db.String(100), unique=True, nullable=False)
        
        # Session data
        start_time = db.Column(db.DateTime, default=datetime.utcnow)
        end_time = db.Column(db.DateTime)
        duration = db.Column(db.Integer)
        
        # Session activity
        interactions_count = db.Column(db.Integer, default=0)
        content_viewed = db.Column(db.Text)  # JSON: list of content IDs
        genres_explored = db.Column(db.Text)  # JSON: genres in session
        search_queries = db.Column(db.Text)  # JSON: search queries in session
        
        # Session characteristics
        session_type = db.Column(db.String(50))
        user_mode = db.Column(db.String(50))
        dominant_mood = db.Column(db.String(50))
        
        # Device and context
        device_type = db.Column(db.String(50))
        ip_address = db.Column(db.String(45))
        user_agent = db.Column(db.Text)
        
        __table_args__ = (
            db.Index('idx_user_sessions', 'user_id', 'start_time'),
        )
        
    return UserSession

def create_content_similarity_model(db):
    """Create ContentSimilarity model for pre-computed similarities"""
    
    class ContentSimilarity(db.Model):
        __tablename__ = 'content_similarities'
        
        id = db.Column(db.Integer, primary_key=True)
        content_id_1 = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
        content_id_2 = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
        
        # Similarity scores
        overall_similarity = db.Column(db.Float)
        story_similarity = db.Column(db.Float)
        genre_similarity = db.Column(db.Float)
        theme_similarity = db.Column(db.Float)
        style_similarity = db.Column(db.Float)
        cast_similarity = db.Column(db.Float)
        
        # Computed at
        computed_at = db.Column(db.DateTime, default=datetime.utcnow)
        
        __table_args__ = (
            db.UniqueConstraint('content_id_1', 'content_id_2'),
            db.Index('idx_content_similarity', 'content_id_1', 'overall_similarity'),
        )
        
    return ContentSimilarity

def create_story_profile_model(db):
    """Create StoryProfile model for detailed story analysis"""
    
    class StoryProfile(db.Model):
        __tablename__ = 'story_profiles'
        
        id = db.Column(db.Integer, primary_key=True)
        content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False, unique=True)
        
        # Story elements
        themes = db.Column(db.Text)  # JSON: main themes
        sub_themes = db.Column(db.Text)  # JSON: sub-themes
        narrative_style = db.Column(db.String(100))  # linear, non-linear, episodic, etc.
        emotional_tone = db.Column(db.String(100))  # dark, uplifting, neutral, etc.
        plot_keywords = db.Column(db.Text)  # JSON: important plot keywords
        
        # Story complexity
        plot_complexity = db.Column(db.Float)  # 0-1 scale
        character_complexity = db.Column(db.Float)  # 0-1 scale
        narrative_depth = db.Column(db.Float)  # 0-1 scale
        
        # Emotional journey
        emotional_arc = db.Column(db.Text)  # JSON: emotional progression
        intensity_level = db.Column(db.Float)  # 0-1 scale
        
        # Target audience
        maturity_level = db.Column(db.String(50))
        cultural_elements = db.Column(db.Text)  # JSON: cultural references
        
        # Story type
        story_type = db.Column(db.String(100))  # hero's journey, tragedy, comedy, etc.
        ending_type = db.Column(db.String(50))  # happy, sad, open, twist
        
        # Embeddings
        story_embedding = db.Column(db.Text)  # JSON: vector representation
        
        # Analysis metadata
        analyzed_at = db.Column(db.DateTime, default=datetime.utcnow)
        confidence_score = db.Column(db.Float)
        
    return StoryProfile

def create_user_story_preference_model(db):
    """Create UserStoryPreference model for tracking story preferences"""
    
    class UserStoryPreference(db.Model):
        __tablename__ = 'user_story_preferences'
        
        id = db.Column(db.Integer, primary_key=True)
        user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
        
        # Preferred story elements
        preferred_themes = db.Column(db.Text)  # JSON: weighted themes
        avoided_themes = db.Column(db.Text)  # JSON: themes to avoid
        preferred_narratives = db.Column(db.Text)  # JSON: narrative styles
        preferred_endings = db.Column(db.Text)  # JSON: ending preferences
        
        # Complexity preferences
        ideal_plot_complexity = db.Column(db.Float, default=0.5)
        ideal_character_depth = db.Column(db.Float, default=0.5)
        ideal_narrative_depth = db.Column(db.Float, default=0.5)
        
        # Emotional preferences
        preferred_emotional_tones = db.Column(db.Text)  # JSON
        emotional_intensity_preference = db.Column(db.Float, default=0.5)
        mood_based_preferences = db.Column(db.Text)  # JSON: mood -> content mapping
        
        # Story type preferences
        story_type_weights = db.Column(db.Text)  # JSON: weighted story types
        
        # Time-based preferences
        time_based_preferences = db.Column(db.Text)  # JSON: time -> preference mapping
        
        # Computed embeddings
        preference_embedding = db.Column(db.Text)  # JSON: vector representation
        
        # Metadata
        last_updated = db.Column(db.DateTime, default=datetime.utcnow)
        confidence_score = db.Column(db.Float, default=0.0)
        
    return UserStoryPreference

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': 'Invalid token'}), 401
                
            # Update last active
            current_user.last_active = datetime.utcnow()
            db.session.commit()
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return jsonify({'error': 'Authentication failed'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

class StoryAnalyzer:
    """Advanced story analysis using NLP and ML"""
    
    def __init__(self):
        self.theme_extractor = TfidfVectorizer(max_features=100, ngram_range=(1, 3))
        self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
        self.emotion_analyzer = TextBlob
        
    def analyze_story(self, content: Content) -> Dict:
        """Perform deep story analysis on content"""
        try:
            # Check if analysis already exists
            story_profile = StoryProfile.query.filter_by(content_id=content.id).first()
            if story_profile and story_profile.analyzed_at > datetime.utcnow() - timedelta(days=30):
                return self._load_story_profile(story_profile)
            
            # Prepare text for analysis
            text = self._prepare_text(content)
            
            # Extract themes
            themes = self._extract_themes(text, content)
            
            # Analyze narrative style
            narrative_style = self._analyze_narrative_style(text, content)
            
            # Analyze emotional tone
            emotional_analysis = self._analyze_emotions(text)
            
            # Analyze complexity
            complexity = self._analyze_complexity(text, content)
            
            # Identify story type
            story_type = self._identify_story_type(text, content)
            
            # Create embeddings
            story_embedding = self._create_story_embedding(
                themes, narrative_style, emotional_analysis, complexity
            )
            
            # Store analysis
            profile_data = {
                'themes': themes['main_themes'],
                'sub_themes': themes['sub_themes'],
                'narrative_style': narrative_style,
                'emotional_tone': emotional_analysis['primary_tone'],
                'emotional_arc': emotional_analysis['arc'],
                'plot_complexity': complexity['plot'],
                'character_complexity': complexity['character'],
                'narrative_depth': complexity['narrative'],
                'story_type': story_type['type'],
                'ending_type': story_type['ending'],
                'intensity_level': emotional_analysis['intensity'],
                'story_embedding': story_embedding,
                'confidence_score': self._calculate_confidence(text)
            }
            
            self._save_story_profile(content.id, profile_data)
            
            return profile_data
            
        except Exception as e:
            logger.error(f"Error analyzing story for content {content.id}: {e}")
            return self._get_default_story_profile()
    
    def _prepare_text(self, content: Content) -> str:
        """Prepare content text for analysis"""
        text_parts = []
        
        if content.title:
            text_parts.append(content.title)
        
        if content.overview:
            text_parts.append(content.overview)
        
        # Add genre information
        try:
            genres = json.loads(content.genres or '[]')
            text_parts.extend(genres)
        except:
            pass
        
        return ' '.join(text_parts).lower()
    
    def _extract_themes(self, text: str, content: Content) -> Dict:
        """Extract main and sub-themes from content"""
        try:
            # Use NLP to extract key themes
            doc = nlp(text)
            
            # Extract noun phrases as potential themes
            themes = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Keep short phrases
                    themes.append(chunk.text)
            
            # Extract entities as themes
            entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']]
            
            # Genre-based themes
            genre_themes = self._get_genre_themes(content)
            
            # Combine and rank themes
            all_themes = themes + entities + genre_themes
            theme_counts = Counter(all_themes)
            
            main_themes = [theme for theme, _ in theme_counts.most_common(5)]
            sub_themes = [theme for theme, _ in theme_counts.most_common()[5:10]]
            
            return {
                'main_themes': main_themes,
                'sub_themes': sub_themes,
                'all_themes': dict(theme_counts)
            }
            
        except Exception as e:
            logger.error(f"Error extracting themes: {e}")
            return {'main_themes': [], 'sub_themes': [], 'all_themes': {}}
    
    def _get_genre_themes(self, content: Content) -> List[str]:
        """Get common themes based on genres"""
        genre_theme_mapping = {
            'action': ['adventure', 'heroism', 'conflict', 'survival'],
            'romance': ['love', 'relationships', 'passion', 'heartbreak'],
            'comedy': ['humor', 'laughter', 'satire', 'fun'],
            'drama': ['emotion', 'conflict', 'relationships', 'life'],
            'horror': ['fear', 'suspense', 'supernatural', 'survival'],
            'sci-fi': ['future', 'technology', 'space', 'innovation'],
            'thriller': ['suspense', 'mystery', 'danger', 'conspiracy'],
            'fantasy': ['magic', 'adventure', 'mythology', 'quest']
        }
        
        themes = []
        try:
            genres = json.loads(content.genres or '[]')
            for genre in genres:
                genre_lower = genre.lower()
                if genre_lower in genre_theme_mapping:
                    themes.extend(genre_theme_mapping[genre_lower])
        except:
            pass
        
        return themes
    
    def _analyze_narrative_style(self, text: str, content: Content) -> str:
        """Analyze the narrative style of content"""
        styles = {
            'linear': ['beginning', 'middle', 'end', 'chronological'],
            'non_linear': ['flashback', 'parallel', 'reverse', 'fragmented'],
            'episodic': ['episode', 'series', 'season', 'chapter'],
            'character_driven': ['character', 'development', 'personal', 'growth'],
            'plot_driven': ['action', 'event', 'twist', 'climax']
        }
        
        text_lower = text.lower()
        style_scores = {}
        
        for style, keywords in styles.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            style_scores[style] = score
        
        # Default to most likely style
        if max(style_scores.values()) == 0:
            return 'linear'
        
        return max(style_scores, key=style_scores.get)
    
    def _analyze_emotions(self, text: str) -> Dict:
        """Analyze emotional content and arc"""
        try:
            blob = TextBlob(text)
            
            # Get overall sentiment
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Determine emotional tone
            if polarity > 0.3:
                tone = 'uplifting'
            elif polarity < -0.3:
                tone = 'dark'
            elif subjectivity > 0.6:
                tone = 'emotional'
            else:
                tone = 'neutral'
            
            # Analyze emotional intensity
            intensity = abs(polarity) * subjectivity
            
            # Emotional arc (simplified)
            sentences = blob.sentences[:10] if len(blob.sentences) > 10 else blob.sentences
            arc = []
            for sentence in sentences:
                arc.append(sentence.sentiment.polarity)
            
            return {
                'primary_tone': tone,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'intensity': min(intensity, 1.0),
                'arc': arc
            }
            
        except Exception as e:
            logger.error(f"Error analyzing emotions: {e}")
            return {
                'primary_tone': 'neutral',
                'polarity': 0,
                'subjectivity': 0.5,
                'intensity': 0.5,
                'arc': []
            }
    
    def _analyze_complexity(self, text: str, content: Content) -> Dict:
        """Analyze story complexity"""
        try:
            doc = nlp(text)
            
            # Plot complexity (based on entity and event diversity)
            entities = set([ent.label_ for ent in doc.ents])
            plot_complexity = min(len(entities) / 10, 1.0)
            
            # Character complexity (based on person entities)
            characters = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
            character_complexity = min(len(set(characters)) / 5, 1.0)
            
            # Narrative depth (based on sentence complexity)
            avg_sentence_length = np.mean([len(sent.text.split()) for sent in doc.sents])
            narrative_depth = min(avg_sentence_length / 20, 1.0)
            
            return {
                'plot': plot_complexity,
                'character': character_complexity,
                'narrative': narrative_depth
            }
            
        except Exception as e:
            logger.error(f"Error analyzing complexity: {e}")
            return {'plot': 0.5, 'character': 0.5, 'narrative': 0.5}
    
    def _identify_story_type(self, text: str, content: Content) -> Dict:
        """Identify the type of story"""
        story_types = {
            'heros_journey': ['hero', 'quest', 'adventure', 'journey', 'triumph'],
            'tragedy': ['death', 'loss', 'tragedy', 'sacrifice', 'downfall'],
            'comedy': ['funny', 'humor', 'laugh', 'joke', 'comedy'],
            'romance': ['love', 'romance', 'relationship', 'couple', 'heart'],
            'mystery': ['mystery', 'detective', 'solve', 'clue', 'investigation'],
            'coming_of_age': ['grow', 'youth', 'adolescent', 'mature', 'learn']
        }
        
        text_lower = text.lower()
        type_scores = {}
        
        for story_type, keywords in story_types.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            type_scores[story_type] = score
        
        # Determine ending type based on emotional analysis
        ending_type = 'open'  # default
        if 'happy' in text_lower or 'triumph' in text_lower:
            ending_type = 'happy'
        elif 'death' in text_lower or 'tragedy' in text_lower:
            ending_type = 'sad'
        elif 'twist' in text_lower or 'surprise' in text_lower:
            ending_type = 'twist'
        
        return {
            'type': max(type_scores, key=type_scores.get) if max(type_scores.values()) > 0 else 'general',
            'ending': ending_type,
            'type_scores': type_scores
        }
    
    def _create_story_embedding(self, themes: Dict, narrative_style: str, 
                              emotional_analysis: Dict, complexity: Dict) -> List[float]:
        """Create a vector embedding representing the story"""
        embedding = []
        
        # Theme vector (simplified)
        theme_vector = [0] * 20
        for i, theme in enumerate(themes.get('main_themes', [])[:20]):
            theme_vector[i] = 1
        embedding.extend(theme_vector)
        
        # Style vector
        styles = ['linear', 'non_linear', 'episodic', 'character_driven', 'plot_driven']
        style_vector = [1 if s == narrative_style else 0 for s in styles]
        embedding.extend(style_vector)
        
        # Emotional vector
        embedding.append(emotional_analysis.get('polarity', 0))
        embedding.append(emotional_analysis.get('subjectivity', 0.5))
        embedding.append(emotional_analysis.get('intensity', 0.5))
        
        # Complexity vector
        embedding.append(complexity.get('plot', 0.5))
        embedding.append(complexity.get('character', 0.5))
        embedding.append(complexity.get('narrative', 0.5))
        
        return embedding
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence in analysis"""
        # Based on text length and quality
        text_length = len(text.split())
        
        if text_length < 20:
            return 0.3
        elif text_length < 50:
            return 0.5
        elif text_length < 100:
            return 0.7
        else:
            return 0.9
    
    def _save_story_profile(self, content_id: int, profile_data: Dict):
        """Save story profile to database"""
        try:
            story_profile = StoryProfile.query.filter_by(content_id=content_id).first()
            
            if not story_profile:
                story_profile = StoryProfile(content_id=content_id)
                db.session.add(story_profile)
            
            # Update fields
            story_profile.themes = json.dumps(profile_data.get('themes', []))
            story_profile.sub_themes = json.dumps(profile_data.get('sub_themes', []))
            story_profile.narrative_style = profile_data.get('narrative_style')
            story_profile.emotional_tone = profile_data.get('emotional_tone')
            story_profile.emotional_arc = json.dumps(profile_data.get('emotional_arc', []))
            story_profile.plot_complexity = profile_data.get('plot_complexity', 0.5)
            story_profile.character_complexity = profile_data.get('character_complexity', 0.5)
            story_profile.narrative_depth = profile_data.get('narrative_depth', 0.5)
            story_profile.story_type = profile_data.get('story_type')
            story_profile.ending_type = profile_data.get('ending_type')
            story_profile.intensity_level = profile_data.get('intensity_level', 0.5)
            story_profile.story_embedding = json.dumps(profile_data.get('story_embedding', []))
            story_profile.confidence_score = profile_data.get('confidence_score', 0.5)
            story_profile.analyzed_at = datetime.utcnow()
            
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Error saving story profile: {e}")
            db.session.rollback()
    
    def _load_story_profile(self, story_profile: StoryProfile) -> Dict:
        """Load story profile from database"""
        return {
            'themes': json.loads(story_profile.themes or '[]'),
            'sub_themes': json.loads(story_profile.sub_themes or '[]'),
            'narrative_style': story_profile.narrative_style,
            'emotional_tone': story_profile.emotional_tone,
            'emotional_arc': json.loads(story_profile.emotional_arc or '[]'),
            'plot_complexity': story_profile.plot_complexity,
            'character_complexity': story_profile.character_complexity,
            'narrative_depth': story_profile.narrative_depth,
            'story_type': story_profile.story_type,
            'ending_type': story_profile.ending_type,
            'intensity_level': story_profile.intensity_level,
            'story_embedding': json.loads(story_profile.story_embedding or '[]'),
            'confidence_score': story_profile.confidence_score
        }
    
    def _get_default_story_profile(self) -> Dict:
        """Return default story profile"""
        return {
            'themes': [],
            'sub_themes': [],
            'narrative_style': 'linear',
            'emotional_tone': 'neutral',
            'emotional_arc': [],
            'plot_complexity': 0.5,
            'character_complexity': 0.5,
            'narrative_depth': 0.5,
            'story_type': 'general',
            'ending_type': 'open',
            'intensity_level': 0.5,
            'story_embedding': [0.5] * 31,
            'confidence_score': 0.3
        }

class DeepUserProfiler:
    """Ultra-advanced user profiling with story preference learning"""
    
    def __init__(self):
        self.story_analyzer = StoryAnalyzer()
        
        # Enhanced interaction weights
        self.interaction_weights = {
            'rating': {'weight': 1.0, 'confidence': 0.95, 'story_impact': 0.9},
            'favorite': {'weight': 0.95, 'confidence': 0.9, 'story_impact': 0.85},
            'watchlist': {'weight': 0.8, 'confidence': 0.8, 'story_impact': 0.7},
            'like': {'weight': 0.7, 'confidence': 0.75, 'story_impact': 0.65},
            'view': {'weight': 0.5, 'confidence': 0.6, 'story_impact': 0.5},
            'rewatch': {'weight': 1.2, 'confidence': 1.0, 'story_impact': 1.0},
            'search': {'weight': 0.3, 'confidence': 0.4, 'story_impact': 0.3},
            'click': {'weight': 0.2, 'confidence': 0.3, 'story_impact': 0.2}
        }
        
        # Time decay with context
        self.temporal_decay = {
            'immediate': {'weight': 1.0, 'days': 7},
            'recent': {'weight': 0.9, 'days': 30},
            'moderate': {'weight': 0.7, 'days': 90},
            'old': {'weight': 0.5, 'days': 180},
            'ancient': {'weight': 0.3, 'days': 365}
        }
    
    def build_deep_profile(self, user_id: int, force_update: bool = False) -> Dict:
        """Build comprehensive user profile with story preferences"""
        try:
            # Check cache
            if not force_update:
                cache_key = f"deep_profile:{user_id}"
                if cache:
                    cached = cache.get(cache_key)
                    if cached:
                        return cached
            
            # Get all interactions with content
            interactions_data = self._get_user_interactions(user_id)
            
            if not interactions_data:
                return self._get_default_deep_profile(user_id)
            
            # Build profile components
            profile = {
                'user_id': user_id,
                'profile_version': '3.0',
                'interaction_count': len(interactions_data),
                
                # Core preferences
                'content_preferences': self._analyze_content_preferences(interactions_data),
                'story_preferences': self._analyze_story_preferences(interactions_data),
                'behavioral_patterns': self._analyze_behavioral_patterns(interactions_data),
                'temporal_patterns': self._analyze_temporal_patterns(interactions_data),
                
                # Advanced insights
                'psychological_profile': self._build_psychological_profile(interactions_data),
                'mood_patterns': self._analyze_mood_patterns(interactions_data),
                'discovery_profile': self._analyze_discovery_patterns(interactions_data),
                
                # Predictions
                'next_likely_content': self._predict_next_content(interactions_data),
                'interest_evolution': self._track_interest_evolution(interactions_data),
                
                # Metadata
                'profile_strength': 0.0,
                'confidence_score': 0.0,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            # Calculate profile strength
            profile['profile_strength'] = self._calculate_profile_strength(profile)
            profile['confidence_score'] = self._calculate_confidence_score(profile)
            
            # Store in database
            self._store_profile(user_id, profile)
            
            # Cache profile
            if cache:
                cache.set(f"deep_profile:{user_id}", profile, timeout=1800)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error building deep profile for user {user_id}: {e}")
            return self._get_default_deep_profile(user_id)
    
    def _get_user_interactions(self, user_id: int) -> List[Dict]:
        """Get all user interactions with rich content data"""
        try:
            # Get interactions with content join
            interactions = db.session.query(UserInteraction, Content).join(
                Content, UserInteraction.content_id == Content.id
            ).filter(UserInteraction.user_id == user_id).order_by(
                UserInteraction.timestamp.desc()
            ).all()
            
            # Enrich with story profiles
            interactions_data = []
            for interaction, content in interactions:
                # Get story profile
                story_profile = StoryProfile.query.filter_by(content_id=content.id).first()
                if not story_profile:
                    # Analyze story if not exists
                    story_data = self.story_analyzer.analyze_story(content)
                else:
                    story_data = self.story_analyzer._load_story_profile(story_profile)
                
                interactions_data.append({
                    'interaction': interaction,
                    'content': content,
                    'story': story_data
                })
            
            return interactions_data
            
        except Exception as e:
            logger.error(f"Error getting user interactions: {e}")
            return []
    
    def _analyze_content_preferences(self, interactions_data: List[Dict]) -> Dict:
        """Analyze detailed content preferences"""
        preferences = {
            'genres': defaultdict(float),
            'languages': defaultdict(float),
            'content_types': defaultdict(float),
            'quality_metrics': {},
            'runtime_preferences': {},
            'release_preferences': {},
            'cast_crew': defaultdict(float),
            'directors': defaultdict(float),
            'studios': defaultdict(float)
        }
        
        total_weight = 0
        quality_ratings = []
        runtimes = []
        
        for data in interactions_data:
            interaction = data['interaction']
            content = data['content']
            
            # Calculate weight
            weight = self._calculate_interaction_weight(interaction)
            total_weight += weight
            
            # Genre preferences with decay
            try:
                genres = json.loads(content.genres or '[]')
                for i, genre in enumerate(genres[:5]):
                    genre_weight = weight * (1.0 / (i + 1))  # Position-based decay
                    preferences['genres'][genre.lower()] += genre_weight
            except:
                pass
            
            # Language preferences
            try:
                languages = json.loads(content.languages or '[]')
                for lang in languages:
                    preferences['languages'][lang.lower()] += weight
            except:
                pass
            
            # Content type
            preferences['content_types'][content.content_type] += weight
            
            # Quality metrics
            if content.rating:
                quality_ratings.append((content.rating, weight))
            
            # Runtime
            if content.runtime:
                runtimes.append((content.runtime, weight))
            
            # Cast and crew preferences
            if ContentPerson:
                cast_crew = ContentPerson.query.filter_by(content_id=content.id).limit(10).all()
                for cp in cast_crew:
                    if cp.role_type == 'cast':
                        person = Person.query.get(cp.person_id)
                        if person:
                            preferences['cast_crew'][person.name] += weight * 0.5
                    elif cp.role_type == 'crew' and cp.job == 'Director':
                        person = Person.query.get(cp.person_id)
                        if person:
                            preferences['directors'][person.name] += weight * 0.7
        
        # Normalize preferences
        if total_weight > 0:
            for category in ['genres', 'languages', 'content_types', 'cast_crew', 'directors']:
                for key in preferences[category]:
                    preferences[category][key] /= total_weight
        
        # Calculate quality preferences
        if quality_ratings:
            weighted_ratings = [r * w for r, w in quality_ratings]
            total_weight = sum(w for _, w in quality_ratings)
            preferences['quality_metrics'] = {
                'average_rating': sum(weighted_ratings) / total_weight if total_weight > 0 else 0,
                'min_acceptable': min(r for r, _ in quality_ratings),
                'preferred_range': [
                    np.percentile([r for r, _ in quality_ratings], 25),
                    np.percentile([r for r, _ in quality_ratings], 75)
                ]
            }
        
        # Calculate runtime preferences
        if runtimes:
            weighted_runtimes = [r * w for r, w in runtimes]
            total_weight = sum(w for _, w in runtimes)
            preferences['runtime_preferences'] = {
                'average': sum(weighted_runtimes) / total_weight if total_weight > 0 else 120,
                'range': [
                    np.percentile([r for r, _ in runtimes], 25),
                    np.percentile([r for r, _ in runtimes], 75)
                ]
            }
        
        return dict(preferences)
    
    def _analyze_story_preferences(self, interactions_data: List[Dict]) -> Dict:
        """Analyze story and narrative preferences"""
        story_prefs = {
            'themes': defaultdict(float),
            'narrative_styles': defaultdict(float),
            'emotional_tones': defaultdict(float),
            'story_types': defaultdict(float),
            'ending_preferences': defaultdict(float),
            'complexity_preferences': {
                'plot': [],
                'character': [],
                'narrative': []
            },
            'emotional_intensity': []
        }
        
        for data in interactions_data:
            interaction = data['interaction']
            story = data['story']
            
            weight = self._calculate_interaction_weight(interaction, story_focused=True)
            
            # Theme preferences
            for theme in story.get('themes', []):
                story_prefs['themes'][theme] += weight
            
            # Narrative style
            style = story.get('narrative_style', 'linear')
            story_prefs['narrative_styles'][style] += weight
            
            # Emotional tone
            tone = story.get('emotional_tone', 'neutral')
            story_prefs['emotional_tones'][tone] += weight
            
            # Story type
            story_type = story.get('story_type', 'general')
            story_prefs['story_types'][story_type] += weight
            
            # Ending preference
            ending = story.get('ending_type', 'open')
            story_prefs['ending_preferences'][ending] += weight
            
            # Complexity preferences (weighted)
            story_prefs['complexity_preferences']['plot'].append(
                (story.get('plot_complexity', 0.5), weight)
            )
            story_prefs['complexity_preferences']['character'].append(
                (story.get('character_complexity', 0.5), weight)
            )
            story_prefs['complexity_preferences']['narrative'].append(
                (story.get('narrative_depth', 0.5), weight)
            )
            
            # Emotional intensity
            story_prefs['emotional_intensity'].append(
                (story.get('intensity_level', 0.5), weight)
            )
        
        # Calculate weighted averages for complexity
        for aspect in ['plot', 'character', 'narrative']:
            values = story_prefs['complexity_preferences'][aspect]
            if values:
                weighted_sum = sum(v * w for v, w in values)
                total_weight = sum(w for _, w in values)
                story_prefs['complexity_preferences'][aspect] = weighted_sum / total_weight if total_weight > 0 else 0.5
            else:
                story_prefs['complexity_preferences'][aspect] = 0.5
        
        # Calculate weighted average for emotional intensity
        if story_prefs['emotional_intensity']:
            weighted_sum = sum(v * w for v, w in story_prefs['emotional_intensity'])
            total_weight = sum(w for _, w in story_prefs['emotional_intensity'])
            story_prefs['emotional_intensity'] = weighted_sum / total_weight if total_weight > 0 else 0.5
        else:
            story_prefs['emotional_intensity'] = 0.5
        
        return dict(story_prefs)
    
    def _analyze_behavioral_patterns(self, interactions_data: List[Dict]) -> Dict:
        """Analyze user behavioral patterns"""
        patterns = {
            'viewing_habits': {},
            'interaction_patterns': {},
            'search_behavior': {},
            'rating_behavior': {},
            'binge_patterns': {},
            'rewatch_patterns': {}
        }
        
        # Group interactions by type
        interaction_groups = defaultdict(list)
        for data in interactions_data:
            interaction = data['interaction']
            interaction_groups[interaction.interaction_type].append(data)
        
        # Viewing habits
        view_times = []
        for data in interaction_groups.get('view', []):
            view_times.append(data['interaction'].timestamp)
        
        if view_times:
            patterns['viewing_habits'] = {
                'total_views': len(view_times),
                'avg_daily_views': len(view_times) / max((view_times[0] - view_times[-1]).days, 1),
                'peak_viewing_hour': Counter([t.hour for t in view_times]).most_common(1)[0][0] if view_times else 0,
                'preferred_day': Counter([t.strftime('%A') for t in view_times]).most_common(1)[0][0] if view_times else 'Unknown'
            }
        
        # Rating behavior
        ratings = [(data['interaction'].rating, data['content']) 
                  for data in interactions_data if data['interaction'].rating]
        
        if ratings:
            patterns['rating_behavior'] = {
                'total_ratings': len(ratings),
                'average_rating': np.mean([r for r, _ in ratings]),
                'rating_std': np.std([r for r, _ in ratings]),
                'harsh_critic': np.mean([r for r, _ in ratings]) < 3.5,
                'generous_rater': np.mean([r for r, _ in ratings]) > 4.0
            }
        
        # Binge patterns
        patterns['binge_patterns'] = self._detect_binge_patterns(interactions_data)
        
        # Rewatch patterns
        content_views = defaultdict(int)
        for data in interactions_data:
            if data['interaction'].interaction_type == 'view':
                content_views[data['content'].id] += 1
        
        rewatched = [cid for cid, count in content_views.items() if count > 1]
        patterns['rewatch_patterns'] = {
            'rewatch_count': len(rewatched),
            'rewatch_rate': len(rewatched) / len(content_views) if content_views else 0,
            'rewatched_content': rewatched[:10]
        }
        
        return patterns
    
    def _analyze_temporal_patterns(self, interactions_data: List[Dict]) -> Dict:
        """Analyze temporal viewing patterns"""
        patterns = {
            'hourly_distribution': defaultdict(int),
            'daily_distribution': defaultdict(int),
            'monthly_distribution': defaultdict(int),
            'seasonal_preferences': defaultdict(list),
            'time_based_content_preferences': defaultdict(list)
        }
        
        for data in interactions_data:
            interaction = data['interaction']
            content = data['content']
            timestamp = interaction.timestamp
            
            # Basic temporal distribution
            patterns['hourly_distribution'][timestamp.hour] += 1
            patterns['daily_distribution'][timestamp.strftime('%A')] += 1
            patterns['monthly_distribution'][timestamp.month] += 1
            
            # Seasonal preferences
            season = self._get_season(timestamp.month)
            patterns['seasonal_preferences'][season].append(content.content_type)
            
            # Time-based content preferences
            time_period = self._get_time_period(timestamp.hour)
            try:
                genres = json.loads(content.genres or '[]')
                patterns['time_based_content_preferences'][time_period].extend(genres)
            except:
                pass
        
        # Process time-based preferences
        for time_period in patterns['time_based_content_preferences']:
            genres = patterns['time_based_content_preferences'][time_period]
            if genres:
                patterns['time_based_content_preferences'][time_period] = Counter(genres).most_common(3)
        
        return dict(patterns)
    
    def _build_psychological_profile(self, interactions_data: List[Dict]) -> Dict:
        """Build psychological profile based on content choices"""
        profile = {
            'personality_traits': {},
            'emotional_preferences': {},
            'cognitive_style': {},
            'social_preferences': {}
        }
        
        # Analyze content choices for personality insights
        genres = []
        themes = []
        emotional_tones = []
        
        for data in interactions_data:
            story = data['story']
            content = data['content']
            
            try:
                genres.extend(json.loads(content.genres or '[]'))
            except:
                pass
            
            themes.extend(story.get('themes', []))
            emotional_tones.append(story.get('emotional_tone', 'neutral'))
        
        # Personality traits based on preferences
        genre_counts = Counter([g.lower() for g in genres])
        
        # Openness (variety seeking)
        profile['personality_traits']['openness'] = len(set(genres)) / max(len(genres), 1)
        
        # Emotional stability (preference for intense content)
        dark_content = sum(1 for tone in emotional_tones if tone in ['dark', 'intense'])
        profile['personality_traits']['emotional_stability'] = 1 - (dark_content / max(len(emotional_tones), 1))
        
        # Extraversion (preference for social content)
        social_genres = ['comedy', 'romance', 'family']
        social_count = sum(genre_counts.get(g, 0) for g in social_genres)
        profile['personality_traits']['extraversion'] = social_count / max(sum(genre_counts.values()), 1)
        
        # Cognitive style
        complexity_scores = [data['story'].get('plot_complexity', 0.5) for data in interactions_data]
        profile['cognitive_style'] = {
            'complexity_preference': np.mean(complexity_scores) if complexity_scores else 0.5,
            'analytical': np.mean(complexity_scores) > 0.6 if complexity_scores else False,
            'intuitive': np.mean(complexity_scores) < 0.4 if complexity_scores else False
        }
        
        return profile
    
    def _analyze_mood_patterns(self, interactions_data: List[Dict]) -> Dict:
        """Analyze mood-based viewing patterns"""
        mood_patterns = {
            'mood_content_mapping': defaultdict(list),
            'time_mood_correlation': {},
            'mood_transitions': []
        }
        
        # Infer mood from content emotional tones
        for i, data in enumerate(interactions_data):
            story = data['story']
            timestamp = data['interaction'].timestamp
            
            # Map emotional tone to mood
            tone = story.get('emotional_tone', 'neutral')
            mood = self._tone_to_mood(tone)
            
            # Time-mood correlation
            hour = timestamp.hour
            time_period = self._get_time_period(hour)
            
            if time_period not in mood_patterns['time_mood_correlation']:
                mood_patterns['time_mood_correlation'][time_period] = []
            mood_patterns['time_mood_correlation'][time_period].append(mood)
            
            # Track mood transitions
            if i > 0:
                prev_tone = interactions_data[i-1]['story'].get('emotional_tone', 'neutral')
                prev_mood = self._tone_to_mood(prev_tone)
                mood_patterns['mood_transitions'].append((prev_mood, mood))
        
        # Process time-mood correlations
        for time_period in mood_patterns['time_mood_correlation']:
            moods = mood_patterns['time_mood_correlation'][time_period]
            if moods:
                mood_patterns['time_mood_correlation'][time_period] = Counter(moods).most_common(1)[0][0]
        
        return mood_patterns
    
    def _analyze_discovery_patterns(self, interactions_data: List[Dict]) -> Dict:
        """Analyze content discovery patterns"""
        discovery = {
            'exploration_rate': 0.0,
            'genre_diversity': 0.0,
            'language_diversity': 0.0,
            'novelty_seeking': 0.0,
            'comfort_zone': [],
            'exploration_areas': []
        }
        
        # Calculate diversity metrics
        all_genres = []
        all_languages = []
        all_themes = []
        
        for data in interactions_data:
            content = data['content']
            story = data['story']
            
            try:
                all_genres.extend(json.loads(content.genres or '[]'))
                all_languages.extend(json.loads(content.languages or '[]'))
            except:
                pass
            
            all_themes.extend(story.get('themes', []))
        
        # Calculate diversity scores
        if all_genres:
            discovery['genre_diversity'] = len(set(all_genres)) / len(all_genres)
        
        if all_languages:
            discovery['language_diversity'] = len(set(all_languages)) / len(all_languages)
        
        # Identify comfort zone (most common elements)
        if all_genres:
            discovery['comfort_zone'] = Counter(all_genres).most_common(3)
        
        # Identify exploration areas (rare elements)
        genre_counts = Counter(all_genres)
        exploration_genres = [g for g, count in genre_counts.items() if count == 1]
        discovery['exploration_areas'] = exploration_genres[:5]
        
        # Calculate exploration rate
        discovery['exploration_rate'] = len(exploration_genres) / max(len(all_genres), 1)
        
        # Novelty seeking (preference for new releases)
        recent_content = sum(1 for data in interactions_data 
                           if data['content'].is_new_release)
        discovery['novelty_seeking'] = recent_content / max(len(interactions_data), 1)
        
        return discovery
    
    def _predict_next_content(self, interactions_data: List[Dict]) -> List[str]:
        """Predict what user might watch next"""
        if not interactions_data:
            return []
        
        # Get recent interaction patterns
        recent = interactions_data[:10]
        
        predictions = []
        
        # Pattern-based prediction
        recent_genres = []
        recent_themes = []
        
        for data in recent:
            try:
                genres = json.loads(data['content'].genres or '[]')
                recent_genres.extend(genres)
            except:
                pass
            
            recent_themes.extend(data['story'].get('themes', []))
        
        # Most likely genres
        if recent_genres:
            likely_genres = Counter(recent_genres).most_common(3)
            predictions.extend([f"{genre} content" for genre, _ in likely_genres])
        
        # Most likely themes
        if recent_themes:
            likely_themes = Counter(recent_themes).most_common(2)
            predictions.extend([f"Stories about {theme}" for theme, _ in likely_themes])
        
        return predictions[:5]
    
    def _track_interest_evolution(self, interactions_data: List[Dict]) -> Dict:
        """Track how user interests evolve over time"""
        evolution = {
            'trending_up': [],
            'trending_down': [],
            'stable_interests': [],
            'timeline': []
        }
        
        # Split interactions into time windows
        if not interactions_data:
            return evolution
        
        # Recent window (last 30 days)
        recent_date = datetime.utcnow() - timedelta(days=30)
        old_date = datetime.utcnow() - timedelta(days=90)
        
        recent_interactions = [d for d in interactions_data 
                             if d['interaction'].timestamp >= recent_date]
        old_interactions = [d for d in interactions_data 
                          if old_date <= d['interaction'].timestamp < recent_date]
        
        # Count genres in each window
        recent_genres = []
        old_genres = []
        
        for data in recent_interactions:
            try:
                genres = json.loads(data['content'].genres or '[]')
                recent_genres.extend(genres)
            except:
                pass
        
        for data in old_interactions:
            try:
                genres = json.loads(data['content'].genres or '[]')
                old_genres.extend(genres)
            except:
                pass
        
        recent_counts = Counter(recent_genres)
        old_counts = Counter(old_genres)
        
        # Identify trends
        all_genres = set(recent_counts.keys()) | set(old_counts.keys())
        
        for genre in all_genres:
            recent = recent_counts.get(genre, 0)
            old = old_counts.get(genre, 0)
            
            if recent > old * 1.5:
                evolution['trending_up'].append(genre)
            elif recent < old * 0.5:
                evolution['trending_down'].append(genre)
            elif recent > 0 and old > 0:
                evolution['stable_interests'].append(genre)
        
        return evolution
    
    def _calculate_interaction_weight(self, interaction: UserInteraction, story_focused: bool = False) -> float:
        """Calculate weight for an interaction"""
        # Base weight from interaction type
        interaction_data = self.interaction_weights.get(
            interaction.interaction_type,
            {'weight': 0.1, 'confidence': 0.2, 'story_impact': 0.1}
        )
        
        if story_focused:
            base_weight = interaction_data['story_impact']
        else:
            base_weight = interaction_data['weight']
        
        # Temporal decay
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        temporal_weight = self._get_temporal_weight(days_ago)
        
        # Rating boost
        rating_boost = 1.0
        if interaction.rating:
            rating_boost = 1 + (interaction.rating - 3) / 5  # Boost for high ratings
        
        return base_weight * temporal_weight * rating_boost
    
    def _get_temporal_weight(self, days_ago: int) -> float:
        """Get temporal decay weight"""
        for period, data in self.temporal_decay.items():
            if days_ago <= data['days']:
                return data['weight']
        return 0.2  # Very old
    
    def _detect_binge_patterns(self, interactions_data: List[Dict]) -> Dict:
        """Detect binge-watching patterns"""
        binge_sessions = []
        current_session = []
        
        for i, data in enumerate(interactions_data):
            if data['interaction'].interaction_type != 'view':
                continue
            
            if not current_session:
                current_session.append(data)
            else:
                # Check time gap
                time_gap = (current_session[-1]['interaction'].timestamp - 
                          data['interaction'].timestamp).total_seconds() / 3600
                
                if time_gap <= 3:  # Within 3 hours
                    current_session.append(data)
                else:
                    if len(current_session) >= 3:
                        binge_sessions.append(current_session)
                    current_session = [data]
        
        if len(current_session) >= 3:
            binge_sessions.append(current_session)
        
        return {
            'binge_count': len(binge_sessions),
            'avg_binge_length': np.mean([len(s) for s in binge_sessions]) if binge_sessions else 0,
            'max_binge_length': max([len(s) for s in binge_sessions]) if binge_sessions else 0,
            'binge_rate': len(binge_sessions) / max(len(interactions_data), 1)
        }
    
    def _get_season(self, month: int) -> str:
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _get_time_period(self, hour: int) -> str:
        """Get time period from hour"""
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    
    def _tone_to_mood(self, tone: str) -> str:
        """Convert emotional tone to mood"""
        mood_mapping = {
            'uplifting': 'happy',
            'dark': 'serious',
            'emotional': 'sentimental',
            'neutral': 'relaxed',
            'intense': 'excited'
        }
        return mood_mapping.get(tone, 'neutral')
    
    def _calculate_profile_strength(self, profile: Dict) -> float:
        """Calculate overall profile strength"""
        factors = []
        
        # Interaction count factor
        interaction_count = profile.get('interaction_count', 0)
        factors.append(min(interaction_count / 100, 1.0))
        
        # Diversity factor
        if 'content_preferences' in profile:
            genre_count = len(profile['content_preferences'].get('genres', {}))
            factors.append(min(genre_count / 20, 1.0))
        
        # Story preference completeness
        if 'story_preferences' in profile:
            story_data = profile['story_preferences']
            if story_data.get('themes'):
                factors.append(0.8)
        
        # Behavioral pattern richness
        if 'behavioral_patterns' in profile:
            patterns = profile['behavioral_patterns']
            if patterns.get('rating_behavior'):
                factors.append(0.7)
        
        return np.mean(factors) * 100 if factors else 0
    
    def _calculate_confidence_score(self, profile: Dict) -> float:
        """Calculate confidence in profile accuracy"""
        interaction_count = profile.get('interaction_count', 0)
        
        # Base confidence from interactions
        base = min(interaction_count / 50, 1.0)
        
        # Adjust for profile completeness
        strength = profile.get('profile_strength', 0) / 100
        
        return (base * 0.7 + strength * 0.3)
    
    def _store_profile(self, user_id: int, profile: Dict):
        """Store profile in database"""
        try:
            user_pref = UserPreference.query.filter_by(user_id=user_id).first()
            
            if not user_pref:
                user_pref = UserPreference(user_id=user_id)
                db.session.add(user_pref)
            
            # Store components
            if 'content_preferences' in profile:
                content_prefs = profile['content_preferences']
                user_pref.genre_preferences = json.dumps(dict(content_prefs.get('genres', {})))
                user_pref.language_preferences = json.dumps(dict(content_prefs.get('languages', {})))
                user_pref.content_type_preferences = json.dumps(dict(content_prefs.get('content_types', {})))
                user_pref.quality_preferences = json.dumps(content_prefs.get('quality_metrics', {}))
                user_pref.runtime_preferences = json.dumps(content_prefs.get('runtime_preferences', {}))
                user_pref.cast_crew_preferences = json.dumps(dict(content_prefs.get('cast_crew', {})))
                user_pref.director_preferences = json.dumps(dict(content_prefs.get('directors', {})))
            
            if 'story_preferences' in profile:
                story_prefs = profile['story_preferences']
                user_pref.story_themes = json.dumps(dict(story_prefs.get('themes', {})))
                user_pref.narrative_styles = json.dumps(dict(story_prefs.get('narrative_styles', {})))
                user_pref.emotional_tones = json.dumps(dict(story_prefs.get('emotional_tones', {})))
                user_pref.plot_complexity = story_prefs.get('complexity_preferences', {}).get('plot', 0.5)
                user_pref.character_depth = story_prefs.get('complexity_preferences', {}).get('character', 0.5)
            
            if 'behavioral_patterns' in profile:
                behavioral = profile['behavioral_patterns']
                user_pref.viewing_patterns = json.dumps(behavioral.get('viewing_habits', {}))
                user_pref.search_patterns = json.dumps(behavioral.get('search_behavior', {}))
            
            if 'temporal_patterns' in profile:
                user_pref.time_patterns = json.dumps(profile['temporal_patterns'])
            
            if 'mood_patterns' in profile:
                user_pref.mood_patterns = json.dumps(profile['mood_patterns'])
            
            # Update metadata
            user_pref.profile_strength = profile.get('profile_strength', 0)
            user_pref.confidence_score = profile.get('confidence_score', 0)
            user_pref.last_calculated = datetime.utcnow()
            
            # Also store story preferences
            self._store_story_preferences(user_id, profile)
            
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Error storing profile: {e}")
            db.session.rollback()
    
    def _store_story_preferences(self, user_id: int, profile: Dict):
        """Store user story preferences"""
        try:
            story_pref = UserStoryPreference.query.filter_by(user_id=user_id).first()
            
            if not story_pref:
                story_pref = UserStoryPreference(user_id=user_id)
                db.session.add(story_pref)
            
            if 'story_preferences' in profile:
                story_data = profile['story_preferences']
                story_pref.preferred_themes = json.dumps(dict(story_data.get('themes', {})))
                story_pref.preferred_narratives = json.dumps(dict(story_data.get('narrative_styles', {})))
                story_pref.preferred_endings = json.dumps(dict(story_data.get('ending_preferences', {})))
                story_pref.ideal_plot_complexity = story_data.get('complexity_preferences', {}).get('plot', 0.5)
                story_pref.ideal_character_depth = story_data.get('complexity_preferences', {}).get('character', 0.5)
                story_pref.ideal_narrative_depth = story_data.get('complexity_preferences', {}).get('narrative', 0.5)
                story_pref.preferred_emotional_tones = json.dumps(dict(story_data.get('emotional_tones', {})))
                story_pref.emotional_intensity_preference = story_data.get('emotional_intensity', 0.5)
            
            if 'mood_patterns' in profile:
                story_pref.mood_based_preferences = json.dumps(profile['mood_patterns'])
            
            story_pref.confidence_score = profile.get('confidence_score', 0)
            story_pref.last_updated = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error storing story preferences: {e}")
    
    def _get_default_deep_profile(self, user_id: int) -> Dict:
        """Get default profile for new users"""
        return {
            'user_id': user_id,
            'profile_version': '3.0',
            'interaction_count': 0,
            'content_preferences': {
                'genres': {},
                'languages': {'english': 0.5},
                'content_types': {'movie': 0.5, 'tv': 0.3, 'anime': 0.2}
            },
            'story_preferences': {
                'themes': {},
                'narrative_styles': {'linear': 0.5},
                'emotional_tones': {'neutral': 0.5},
                'story_types': {'general': 0.5},
                'complexity_preferences': {'plot': 0.5, 'character': 0.5, 'narrative': 0.5}
            },
            'behavioral_patterns': {},
            'temporal_patterns': {},
            'psychological_profile': {},
            'mood_patterns': {},
            'discovery_profile': {'exploration_rate': 0.5},
            'profile_strength': 0.0,
            'confidence_score': 0.0,
            'last_updated': datetime.utcnow().isoformat()
        }

class StoryMatchingEngine:
    """Engine for matching content based on story similarities"""
    
    def __init__(self):
        self.story_analyzer = StoryAnalyzer()
        
    def find_story_matches(self, user_id: int, base_content_id: Optional[int] = None, 
                          limit: int = 20) -> List[Dict]:
        """Find content with similar stories"""
        try:
            # Get user story preferences
            user_story_pref = UserStoryPreference.query.filter_by(user_id=user_id).first()
            
            if not user_story_pref:
                return []
            
            # Get preferred themes and styles
            preferred_themes = json.loads(user_story_pref.preferred_themes or '{}')
            preferred_narratives = json.loads(user_story_pref.preferred_narratives or '{}')
            preferred_tones = json.loads(user_story_pref.preferred_emotional_tones or '{}')
            
            # Build query
            query = db.session.query(Content, StoryProfile).join(
                StoryProfile, Content.id == StoryProfile.content_id
            )
            
            # Exclude already watched
            watched_ids = db.session.query(UserInteraction.content_id).filter_by(
                user_id=user_id
            ).subquery()
            query = query.filter(~Content.id.in_(watched_ids))
            
            # Get candidates
            candidates = query.limit(limit * 5).all()
            
            # Score candidates
            scored_matches = []
            for content, story_profile in candidates:
                score = self._calculate_story_match_score(
                    story_profile, preferred_themes, preferred_narratives, preferred_tones,
                    user_story_pref
                )
                
                if score > 0.3:  # Minimum threshold
                    scored_matches.append({
                        'content': content,
                        'story_profile': story_profile,
                        'match_score': score,
                        'match_reasons': self._get_match_reasons(
                            story_profile, preferred_themes, preferred_narratives
                        )
                    })
            
            # Sort by score
            scored_matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            return scored_matches[:limit]
            
        except Exception as e:
            logger.error(f"Error finding story matches: {e}")
            return []
    
    def _calculate_story_match_score(self, story_profile: StoryProfile, 
                                    preferred_themes: Dict, preferred_narratives: Dict,
                                    preferred_tones: Dict, user_story_pref: UserStoryPreference) -> float:
        """Calculate how well a story matches user preferences"""
        score = 0.0
        weights = {
            'themes': 0.35,
            'narrative': 0.20,
            'emotional': 0.20,
            'complexity': 0.15,
            'type': 0.10
        }
        
        # Theme matching
        try:
            story_themes = json.loads(story_profile.themes or '[]')
            theme_score = 0
            for theme in story_themes:
                if theme in preferred_themes:
                    theme_score += preferred_themes[theme]
            score += min(theme_score, 1.0) * weights['themes']
        except:
            pass
        
        # Narrative style matching
        if story_profile.narrative_style in preferred_narratives:
            score += preferred_narratives[story_profile.narrative_style] * weights['narrative']
        
        # Emotional tone matching
        if story_profile.emotional_tone in preferred_tones:
            score += preferred_tones[story_profile.emotional_tone] * weights['emotional']
        
        # Complexity matching
        complexity_diff = abs(story_profile.plot_complexity - user_story_pref.ideal_plot_complexity)
        complexity_match = 1 - complexity_diff
        score += complexity_match * weights['complexity']
        
        return min(score, 1.0)
    
    def _get_match_reasons(self, story_profile: StoryProfile, 
                          preferred_themes: Dict, preferred_narratives: Dict) -> List[str]:
        """Generate reasons for story match"""
        reasons = []
        
        try:
            story_themes = json.loads(story_profile.themes or '[]')
            matching_themes = [t for t in story_themes if t in preferred_themes]
            if matching_themes:
                reasons.append(f"Similar themes: {', '.join(matching_themes[:2])}")
        except:
            pass
        
        if story_profile.narrative_style in preferred_narratives:
            reasons.append(f"{story_profile.narrative_style.replace('_', ' ').title()} narrative style")
        
        if story_profile.emotional_tone:
            reasons.append(f"{story_profile.emotional_tone.title()} emotional tone")
        
        return reasons

class UltraPersonalizedRecommendationEngine:
    """Ultra-advanced recommendation engine with all enhancements"""
    
    def __init__(self):
        self.deep_profiler = DeepUserProfiler()
        self.story_matcher = StoryMatchingEngine()
        self.story_analyzer = StoryAnalyzer()
        
        # Component weights for different user states
        self.recommendation_strategies = {
            'discovery_mode': {
                'story_matching': 0.25,
                'collaborative': 0.20,
                'content_based': 0.20,
                'trending': 0.15,
                'exploration': 0.20
            },
            'comfort_mode': {
                'story_matching': 0.40,
                'collaborative': 0.25,
                'content_based': 0.20,
                'trending': 0.05,
                'exploration': 0.10
            },
            'binge_mode': {
                'story_matching': 0.35,
                'collaborative': 0.30,
                'content_based': 0.25,
                'trending': 0.05,
                'exploration': 0.05
            },
            'selective_mode': {
                'story_matching': 0.45,
                'collaborative': 0.25,
                'content_based': 0.20,
                'trending': 0.05,
                'exploration': 0.05
            }
        }
    
    def get_ultra_personalized_recommendations(self, user_id: int, 
                                              content_type: Optional[str] = None,
                                              mood: Optional[str] = None,
                                              limit: int = 20) -> Dict:
        """Get ultra-personalized recommendations with story matching"""
        try:
            # Build deep user profile
            user_profile = self.deep_profiler.build_deep_profile(user_id)
            
            # Determine user mode and mood
            user_mode = self._determine_user_mode(user_profile, mood)
            current_mood = mood or self._infer_current_mood(user_profile)
            
            # Get recommendation strategy
            strategy = self.recommendation_strategies.get(
                user_mode, 
                self.recommendation_strategies['discovery_mode']
            )
            
            # Collect recommendations from different sources
            all_recommendations = []
            
            # 1. Story-based matching
            if strategy['story_matching'] > 0:
                story_recs = self._get_story_based_recommendations(
                    user_id, user_profile, limit * 2
                )
                all_recommendations.extend(story_recs)
            
            # 2. Collaborative filtering
            if strategy['collaborative'] > 0:
                collab_recs = self._get_collaborative_recommendations(
                    user_id, user_profile, limit * 2
                )
                all_recommendations.extend(collab_recs)
            
            # 3. Content-based filtering
            if strategy['content_based'] > 0:
                content_recs = self._get_content_based_recommendations(
                    user_id, user_profile, limit * 2
                )
                all_recommendations.extend(content_recs)
            
            # 4. Trending content
            if strategy['trending'] > 0:
                trending_recs = self._get_trending_recommendations(
                    user_profile, limit
                )
                all_recommendations.extend(trending_recs)
            
            # 5. Exploration recommendations
            if strategy['exploration'] > 0:
                exploration_recs = self._get_exploration_recommendations(
                    user_id, user_profile, limit
                )
                all_recommendations.extend(exploration_recs)
            
            # Fuse and rank recommendations
            final_recommendations = self._fuse_and_rank_recommendations(
                all_recommendations, user_profile, strategy, current_mood
            )
            
            # Apply filters
            if content_type:
                final_recommendations = [
                    r for r in final_recommendations 
                    if r['content'].content_type == content_type
                ]
            
            # Format recommendations
            formatted_recs = self._format_recommendations(
                final_recommendations[:limit], user_profile
            )
            
            # Track recommendations
            self._track_recommendations(user_id, formatted_recs, user_mode, current_mood)
            
            return {
                'recommendations': formatted_recs,
                'user_insights': {
                    'profile_strength': user_profile.get('profile_strength', 0),
                    'confidence_score': user_profile.get('confidence_score', 0),
                    'user_mode': user_mode,
                    'current_mood': current_mood,
                    'top_themes': self._get_top_themes(user_profile),
                    'narrative_preference': self._get_narrative_preference(user_profile),
                    'discovery_level': user_profile.get('discovery_profile', {}).get('exploration_rate', 0.5)
                },
                'recommendation_metadata': {
                    'total_candidates': len(all_recommendations),
                    'strategy_used': strategy,
                    'personalization_level': 'ultra_high',
                    'story_matching_enabled': True,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting ultra-personalized recommendations: {e}")
            return {
                'recommendations': [],
                'user_insights': {},
                'recommendation_metadata': {'error': str(e)}
            }
    
    def _determine_user_mode(self, user_profile: Dict, requested_mood: Optional[str]) -> str:
        """Determine user's current mode"""
        if requested_mood:
            mood_to_mode = {
                'adventurous': 'discovery_mode',
                'relaxed': 'comfort_mode',
                'focused': 'binge_mode',
                'selective': 'selective_mode'
            }
            return mood_to_mode.get(requested_mood, 'discovery_mode')
        
        # Infer from recent behavior
        behavioral = user_profile.get('behavioral_patterns', {})
        
        # Check for binge pattern
        binge_data = behavioral.get('binge_patterns', {})
        if binge_data.get('binge_rate', 0) > 0.3:
            return 'binge_mode'
        
        # Check exploration tendency
        discovery = user_profile.get('discovery_profile', {})
        if discovery.get('exploration_rate', 0.5) > 0.7:
            return 'discovery_mode'
        
        # Check rating behavior
        rating_behavior = behavioral.get('rating_behavior', {})
        if rating_behavior.get('harsh_critic', False):
            return 'selective_mode'
        
        return 'comfort_mode'
    
    def _infer_current_mood(self, user_profile: Dict) -> str:
        """Infer user's current mood from patterns"""
        # Get current time-based mood
        hour = datetime.utcnow().hour
        time_period = self._get_time_period(hour)
        
        mood_patterns = user_profile.get('mood_patterns', {})
        time_mood = mood_patterns.get('time_mood_correlation', {})
        
        if time_period in time_mood:
            return time_mood[time_period]
        
        # Default based on time
        if 5 <= hour < 12:
            return 'energetic'
        elif 12 <= hour < 17:
            return 'focused'
        elif 17 <= hour < 21:
            return 'relaxed'
        else:
            return 'contemplative'
    
    def _get_story_based_recommendations(self, user_id: int, user_profile: Dict, 
                                        limit: int) -> List[Dict]:
        """Get recommendations based on story matching"""
        try:
            # Get story matches
            story_matches = self.story_matcher.find_story_matches(user_id, limit=limit)
            
            recommendations = []
            for match in story_matches:
                recommendations.append({
                    'content': match['content'],
                    'score': match['match_score'],
                    'method': 'story_matching',
                    'match_reasons': match['match_reasons'],
                    'story_profile': match['story_profile']
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in story-based recommendations: {e}")
            return []
    
    def _get_collaborative_recommendations(self, user_id: int, user_profile: Dict, 
                                          limit: int) -> List[Dict]:
        """Get collaborative filtering recommendations"""
        try:
            # Find similar users
            similar_users = self._find_similar_users(user_id, user_profile)
            
            if not similar_users:
                return []
            
            # Get content liked by similar users
            recommendations = []
            
            for similar_user_id, similarity_score in similar_users[:10]:
                # Get their highly rated content
                liked_content = db.session.query(UserInteraction, Content).join(
                    Content, UserInteraction.content_id == Content.id
                ).filter(
                    UserInteraction.user_id == similar_user_id,
                    or_(
                        UserInteraction.interaction_type == 'favorite',
                        and_(
                            UserInteraction.interaction_type == 'rating',
                            UserInteraction.rating >= 4
                        )
                    )
                ).limit(10).all()
                
                for interaction, content in liked_content:
                    # Check if user hasn't seen it
                    already_seen = UserInteraction.query.filter_by(
                        user_id=user_id,
                        content_id=content.id
                    ).first()
                    
                    if not already_seen:
                        recommendations.append({
                            'content': content,
                            'score': similarity_score * 0.8,
                            'method': 'collaborative',
                            'similar_user': similar_user_id
                        })
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in collaborative recommendations: {e}")
            return []
    
    def _get_content_based_recommendations(self, user_id: int, user_profile: Dict, 
                                          limit: int) -> List[Dict]:
        """Get content-based recommendations"""
        try:
            # Get user's favorite content
            favorite_content = db.session.query(UserInteraction, Content).join(
                Content, UserInteraction.content_id == Content.id
            ).filter(
                UserInteraction.user_id == user_id,
                or_(
                    UserInteraction.interaction_type == 'favorite',
                    and_(
                        UserInteraction.interaction_type == 'rating',
                        UserInteraction.rating >= 4.5
                    )
                )
            ).order_by(UserInteraction.timestamp.desc()).limit(10).all()
            
            if not favorite_content:
                return []
            
            recommendations = []
            
            for interaction, base_content in favorite_content:
                # Find similar content
                similar_content = self._find_similar_content(base_content, limit=5)
                
                for similar, similarity_score in similar_content:
                    # Check if user hasn't seen it
                    already_seen = UserInteraction.query.filter_by(
                        user_id=user_id,
                        content_id=similar.id
                    ).first()
                    
                    if not already_seen:
                        recommendations.append({
                            'content': similar,
                            'score': similarity_score,
                            'method': 'content_based',
                            'based_on': base_content.title
                        })
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
            return []
    
    def _get_trending_recommendations(self, user_profile: Dict, limit: int) -> List[Dict]:
        """Get trending content matching user preferences"""
        try:
            # Get user's preferred genres and languages
            content_prefs = user_profile.get('content_preferences', {})
            preferred_genres = list(content_prefs.get('genres', {}).keys())[:5]
            preferred_languages = list(content_prefs.get('languages', {}).keys())[:3]
            
            # Query trending content
            query = Content.query.filter(
                or_(
                    Content.is_trending == True,
                    Content.is_new_release == True
                )
            )
            
            # Filter by preferences
            if preferred_genres:
                genre_filters = []
                for genre in preferred_genres:
                    genre_filters.append(Content.genres.contains(genre))
                query = query.filter(or_(*genre_filters))
            
            trending = query.order_by(Content.popularity.desc()).limit(limit).all()
            
            recommendations = []
            for content in trending:
                score = 0.5  # Base score for trending
                
                # Boost for matching preferences
                try:
                    content_genres = json.loads(content.genres or '[]')
                    for genre in content_genres:
                        if genre.lower() in preferred_genres:
                            score += 0.1
                except:
                    pass
                
                recommendations.append({
                    'content': content,
                    'score': min(score, 1.0),
                    'method': 'trending'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in trending recommendations: {e}")
            return []
    
    def _get_exploration_recommendations(self, user_id: int, user_profile: Dict, 
                                        limit: int) -> List[Dict]:
        """Get exploration recommendations for discovering new content"""
        try:
            # Get user's comfort zone
            content_prefs = user_profile.get('content_preferences', {})
            known_genres = set(content_prefs.get('genres', {}).keys())
            known_languages = set(content_prefs.get('languages', {}).keys())
            
            # Find content outside comfort zone but high quality
            query = Content.query.filter(
                Content.rating >= 7.5,
                Content.vote_count >= 100
            )
            
            # Exclude already watched
            watched_ids = db.session.query(UserInteraction.content_id).filter_by(
                user_id=user_id
            ).subquery()
            query = query.filter(~Content.id.in_(watched_ids))
            
            exploration_content = query.order_by(func.random()).limit(limit * 2).all()
            
            recommendations = []
            for content in exploration_content:
                exploration_score = 0.3  # Base exploration score
                
                # Check for new elements
                try:
                    content_genres = json.loads(content.genres or '[]')
                    new_genres = [g for g in content_genres if g.lower() not in known_genres]
                    if new_genres:
                        exploration_score += 0.2
                except:
                    pass
                
                # Quality bonus
                if content.rating and content.rating >= 8.0:
                    exploration_score += 0.2
                
                recommendations.append({
                    'content': content,
                    'score': exploration_score,
                    'method': 'exploration'
                })
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in exploration recommendations: {e}")
            return []
    
    def _find_similar_users(self, user_id: int, user_profile: Dict) -> List[Tuple[int, float]]:
        """Find users with similar preferences"""
        try:
            # Get user's interactions
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            user_content_ids = set([i.content_id for i in user_interactions])
            
            if len(user_content_ids) < 5:
                return []
            
            # Find users with overlapping content
            similar_users = db.session.query(
                UserInteraction.user_id,
                func.count().label('common_content')
            ).filter(
                UserInteraction.content_id.in_(user_content_ids),
                UserInteraction.user_id != user_id
            ).group_by(UserInteraction.user_id).having(
                func.count() >= 5
            ).order_by(desc('common_content')).limit(50).all()
            
            # Calculate similarity scores
            user_similarities = []
            
            for other_user_id, common_count in similar_users:
                # Get other user's interactions
                other_interactions = UserInteraction.query.filter_by(
                    user_id=other_user_id
                ).all()
                other_content_ids = set([i.content_id for i in other_interactions])
                
                # Jaccard similarity
                intersection = len(user_content_ids & other_content_ids)
                union = len(user_content_ids | other_content_ids)
                
                if union > 0:
                    similarity = intersection / union
                    
                    # Boost for rating similarity
                    user_ratings = {i.content_id: i.rating for i in user_interactions if i.rating}
                    other_ratings = {i.content_id: i.rating for i in other_interactions if i.rating}
                    
                    common_rated = set(user_ratings.keys()) & set(other_ratings.keys())
                    if common_rated:
                        rating_diffs = [abs(user_ratings[cid] - other_ratings[cid]) 
                                      for cid in common_rated]
                        avg_diff = np.mean(rating_diffs)
                        rating_similarity = 1 - (avg_diff / 5)  # Normalize to 0-1
                        similarity = (similarity + rating_similarity) / 2
                    
                    user_similarities.append((other_user_id, similarity))
            
            # Sort by similarity
            user_similarities.sort(key=lambda x: x[1], reverse=True)
            
            return user_similarities[:20]
            
        except Exception as e:
            logger.error(f"Error finding similar users: {e}")
            return []
    
    def _find_similar_content(self, base_content: Content, limit: int = 10) -> List[Tuple[Content, float]]:
        """Find content similar to given content"""
        try:
            # Check if we have pre-computed similarities
            similarities = ContentSimilarity.query.filter_by(
                content_id_1=base_content.id
            ).order_by(ContentSimilarity.overall_similarity.desc()).limit(limit).all()
            
            if similarities:
                similar_content = []
                for sim in similarities:
                    content = Content.query.get(sim.content_id_2)
                    if content:
                        similar_content.append((content, sim.overall_similarity))
                return similar_content
            
            # Compute similarities on the fly
            return self._compute_content_similarity(base_content, limit)
            
        except Exception as e:
            logger.error(f"Error finding similar content: {e}")
            return []
    
    def _compute_content_similarity(self, base_content: Content, limit: int) -> List[Tuple[Content, float]]:
        """Compute content similarity on the fly"""
        try:
            # Get base content features
            base_genres = set(json.loads(base_content.genres or '[]'))
            base_languages = set(json.loads(base_content.languages or '[]'))
            
            # Query similar content
            query = Content.query.filter(
                Content.id != base_content.id,
                Content.content_type == base_content.content_type
            )
            
            # Get candidates
            candidates = query.limit(limit * 5).all()
            
            similarities = []
            for candidate in candidates:
                similarity = 0.0
                
                # Genre similarity
                try:
                    candidate_genres = set(json.loads(candidate.genres or '[]'))
                    if base_genres and candidate_genres:
                        genre_sim = len(base_genres & candidate_genres) / len(base_genres | candidate_genres)
                        similarity += genre_sim * 0.4
                except:
                    pass
                
                # Language similarity
                try:
                    candidate_languages = set(json.loads(candidate.languages or '[]'))
                    if base_languages and candidate_languages:
                        lang_sim = len(base_languages & candidate_languages) / len(base_languages | candidate_languages)
                        similarity += lang_sim * 0.2
                except:
                    pass
                
                # Rating similarity
                if base_content.rating and candidate.rating:
                    rating_diff = abs(base_content.rating - candidate.rating)
                    rating_sim = 1 - (rating_diff / 10)
                    similarity += rating_sim * 0.2
                
                # Release date proximity
                if base_content.release_date and candidate.release_date:
                    year_diff = abs(base_content.release_date.year - candidate.release_date.year)
                    date_sim = max(0, 1 - (year_diff / 10))
                    similarity += date_sim * 0.2
                
                if similarity > 0.3:
                    similarities.append((candidate, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Error computing content similarity: {e}")
            return []
    
    def _fuse_and_rank_recommendations(self, all_recommendations: List[Dict], 
                                      user_profile: Dict, strategy: Dict, 
                                      current_mood: str) -> List[Dict]:
        """Fuse and rank recommendations from multiple sources"""
        try:
            # Group by content ID
            content_scores = defaultdict(lambda: {
                'scores': [],
                'methods': [],
                'content': None,
                'metadata': {}
            })
            
            for rec in all_recommendations:
                content_id = rec['content'].id
                content_scores[content_id]['scores'].append(rec['score'])
                content_scores[content_id]['methods'].append(rec['method'])
                content_scores[content_id]['content'] = rec['content']
                
                # Store method-specific metadata
                if rec['method'] == 'story_matching' and 'match_reasons' in rec:
                    content_scores[content_id]['metadata']['match_reasons'] = rec['match_reasons']
                elif rec['method'] == 'content_based' and 'based_on' in rec:
                    content_scores[content_id]['metadata']['based_on'] = rec['based_on']
            
            # Calculate fusion scores
            fused_recommendations = []
            
            for content_id, data in content_scores.items():
                # Calculate weighted score based on strategy
                final_score = 0.0
                total_weight = 0.0
                
                for method, score in zip(data['methods'], data['scores']):
                    weight = strategy.get(method, 0.1)
                    final_score += score * weight
                    total_weight += weight
                
                if total_weight > 0:
                    final_score /= total_weight
                
                # Apply mood-based adjustments
                final_score = self._apply_mood_adjustment(
                    final_score, data['content'], current_mood
                )
                
                # Apply diversity bonus
                final_score = self._apply_diversity_bonus(
                    final_score, data['content'], user_profile
                )
                
                fused_recommendations.append({
                    'content': data['content'],
                    'score': final_score,
                    'methods': list(set(data['methods'])),
                    'metadata': data['metadata']
                })
            
            # Sort by final score
            fused_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return fused_recommendations
            
        except Exception as e:
            logger.error(f"Error fusing recommendations: {e}")
            return []
    
    def _apply_mood_adjustment(self, score: float, content: Content, mood: str) -> float:
        """Adjust score based on current mood"""
        try:
            # Get content's emotional tone
            story_profile = StoryProfile.query.filter_by(content_id=content.id).first()
            if not story_profile:
                return score
            
            emotional_tone = story_profile.emotional_tone
            
            # Mood-content matching
            mood_content_match = {
                'happy': {'uplifting': 1.2, 'neutral': 1.0, 'dark': 0.8},
                'sad': {'emotional': 1.2, 'dark': 1.1, 'uplifting': 0.9},
                'excited': {'intense': 1.2, 'uplifting': 1.1, 'neutral': 0.9},
                'relaxed': {'neutral': 1.2, 'uplifting': 1.0, 'intense': 0.8},
                'contemplative': {'emotional': 1.2, 'dark': 1.1, 'neutral': 1.0}
            }
            
            if mood in mood_content_match and emotional_tone in mood_content_match[mood]:
                multiplier = mood_content_match[mood][emotional_tone]
                score *= multiplier
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error applying mood adjustment: {e}")
            return score
    
    def _apply_diversity_bonus(self, score: float, content: Content, user_profile: Dict) -> float:
        """Apply diversity bonus to encourage exploration"""
        try:
            discovery_profile = user_profile.get('discovery_profile', {})
            exploration_rate = discovery_profile.get('exploration_rate', 0.5)
            
            if exploration_rate > 0.6:
                # User likes exploration - boost diverse content
                comfort_zone = discovery_profile.get('comfort_zone', [])
                
                try:
                    content_genres = json.loads(content.genres or '[]')
                    
                    # Check if content is outside comfort zone
                    is_diverse = True
                    for genre, _ in comfort_zone:
                        if genre in [g.lower() for g in content_genres]:
                            is_diverse = False
                            break
                    
                    if is_diverse:
                        score *= 1.1  # 10% diversity bonus
                
                except:
                    pass
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error applying diversity bonus: {e}")
            return score
    
    def _format_recommendations(self, recommendations: List[Dict], user_profile: Dict) -> List[Dict]:
        """Format recommendations for API response"""
        formatted = []
        
        for rec in recommendations:
            content = rec['content']
            
            # Ensure slug exists
            if not content.slug:
                try:
                    content.ensure_slug()
                    db.session.commit()
                except:
                    content.slug = f"content-{content.id}"
            
            # Get story insights
            story_profile = StoryProfile.query.filter_by(content_id=content.id).first()
            story_insights = {}
            
            if story_profile:
                story_insights = {
                    'themes': json.loads(story_profile.themes or '[]')[:3],
                    'emotional_tone': story_profile.emotional_tone,
                    'narrative_style': story_profile.narrative_style,
                    'complexity': {
                        'plot': story_profile.plot_complexity,
                        'character': story_profile.character_complexity
                    }
                }
            
            # Generate recommendation reason
            reason = self._generate_recommendation_reason(rec, user_profile)
            
            # Predict user rating
            predicted_rating = self._predict_user_rating(content, user_profile)
            
            formatted_rec = {
                'id': content.id,
                'slug': content.slug,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'languages': json.loads(content.languages or '[]'),
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'runtime': content.runtime,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:200] + '...' if content.overview and len(content.overview) > 200 else content.overview,
                'recommendation_score': round(rec['score'], 3),
                'recommendation_reason': reason,
                'predicted_rating': predicted_rating,
                'methods_used': rec.get('methods', []),
                'story_insights': story_insights,
                'match_details': rec.get('metadata', {}),
                'is_trending': content.is_trending,
                'is_new_release': content.is_new_release
            }
            
            formatted.append(formatted_rec)
        
        return formatted
    
    def _generate_recommendation_reason(self, recommendation: Dict, user_profile: Dict) -> str:
        """Generate detailed recommendation reason"""
        reasons = []
        methods = recommendation.get('methods', [])
        metadata = recommendation.get('metadata', {})
        
        # Method-specific reasons
        if 'story_matching' in methods and 'match_reasons' in metadata:
            reasons.extend(metadata['match_reasons'][:2])
        
        if 'content_based' in methods and 'based_on' in metadata:
            reasons.append(f"Similar to {metadata['based_on']}")
        
        if 'collaborative' in methods:
            reasons.append("Users with similar taste loved this")
        
        if 'trending' in methods:
            reasons.append("Trending in your preferred genres")
        
        if 'exploration' in methods:
            reasons.append("Expand your horizons with this highly-rated content")
        
        # Content-specific reasons
        content = recommendation['content']
        
        # Check genre match
        try:
            content_genres = json.loads(content.genres or '[]')
            user_genres = user_profile.get('content_preferences', {}).get('genres', {})
            
            matching_genres = [g for g in content_genres if g.lower() in user_genres]
            if matching_genres:
                reasons.append(f"Features {matching_genres[0]}")
        except:
            pass
        
        # Quality reason
        if content.rating and content.rating >= 8.5:
            reasons.append("Exceptional quality")
        
        # Combine reasons
        if reasons:
            return " • ".join(reasons[:3])
        else:
            return "Specially selected based on your unique preferences"
    
    def _predict_user_rating(self, content: Content, user_profile: Dict) -> float:
        """Predict how user would rate this content"""
        try:
            base_rating = 3.5
            
            # Genre match impact
            try:
                content_genres = json.loads(content.genres or '[]')
                user_genres = user_profile.get('content_preferences', {}).get('genres', {})
                
                genre_match = 0
                for genre in content_genres:
                    if genre.lower() in user_genres:
                        genre_match += user_genres[genre.lower()]
                
                base_rating += genre_match * 2  # Up to +2 for perfect genre match
            except:
                pass
            
            # Story preference match
            story_prefs = user_profile.get('story_preferences', {})
            story_profile = StoryProfile.query.filter_by(content_id=content.id).first()
            
            if story_profile and story_prefs:
                # Complexity match
                ideal_complexity = story_prefs.get('complexity_preferences', {}).get('plot', 0.5)
                complexity_diff = abs(story_profile.plot_complexity - ideal_complexity)
                base_rating += (1 - complexity_diff) * 0.5
                
                # Emotional tone match
                preferred_tones = story_prefs.get('emotional_tones', {})
                if story_profile.emotional_tone in preferred_tones:
                    base_rating += preferred_tones[story_profile.emotional_tone] * 0.5
            
            # Quality preference alignment
            quality_prefs = user_profile.get('content_preferences', {}).get('quality_metrics', {})
            if content.rating and quality_prefs:
                user_avg = quality_prefs.get('average_rating', 7.0)
                if content.rating >= user_avg:
                    base_rating += 0.3
            
            return min(round(base_rating, 1), 5.0)
            
        except Exception as e:
            logger.error(f"Error predicting rating: {e}")
            return 3.5
    
    def _track_recommendations(self, user_id: int, recommendations: List[Dict], 
                              user_mode: str, mood: str):
        """Track recommendations for learning"""
        try:
            for idx, rec in enumerate(recommendations[:10]):
                feedback = RecommendationFeedback(
                    user_id=user_id,
                    content_id=rec['id'],
                    recommendation_score=rec['recommendation_score'],
                    recommendation_method=','.join(rec['methods_used']),
                    recommendation_reason=rec['recommendation_reason'],
                    recommendation_rank=idx + 1,
                    story_match_score=rec.get('story_insights', {}).get('match_score', 0),
                    user_mode=user_mode,
                    user_mood=mood,
                    time_of_day=datetime.utcnow().hour,
                    day_of_week=datetime.utcnow().weekday()
                )
                db.session.add(feedback)
            
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Error tracking recommendations: {e}")
            db.session.rollback()
    
    def _get_top_themes(self, user_profile: Dict) -> List[str]:
        """Get user's top themes"""
        story_prefs = user_profile.get('story_preferences', {})
        themes = story_prefs.get('themes', {})
        
        if themes:
            sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
            return [theme for theme, _ in sorted_themes[:5]]
        
        return []
    
    def _get_narrative_preference(self, user_profile: Dict) -> str:
        """Get user's narrative preference"""
        story_prefs = user_profile.get('story_preferences', {})
        narratives = story_prefs.get('narrative_styles', {})
        
        if narratives:
            return max(narratives, key=narratives.get)
        
        return 'linear'
    
    def _get_time_period(self, hour: int) -> str:
        """Get time period from hour"""
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'

# Initialize the ultra-personalized engine
ultra_engine = UltraPersonalizedRecommendationEngine()

# API function to be called from users.py
def get_personalized_recommendations_for_user(user_id: int, content_type: Optional[str] = None, 
                                             limit: int = 20, mood: Optional[str] = None) -> Dict:
    """Get ultra-personalized recommendations for a user"""
    return ultra_engine.get_ultra_personalized_recommendations(
        user_id, content_type, mood, limit
    )

def update_user_profile(user_id: int, force_update: bool = False) -> Dict:
    """Update user's deep profile"""
    return ultra_engine.deep_profiler.build_deep_profile(user_id, force_update)

def record_recommendation_feedback(user_id: int, content_id: int, feedback_type: str, 
                                  rating: Optional[float] = None) -> bool:
    """Record feedback for a recommendation"""
    try:
        feedback = RecommendationFeedback.query.filter_by(
            user_id=user_id,
            content_id=content_id
        ).order_by(RecommendationFeedback.recommended_at.desc()).first()
        
        if feedback:
            feedback.feedback_type = feedback_type
            feedback.user_rating = rating
            feedback.feedback_at = datetime.utcnow()
            feedback.was_successful = feedback_type in ['liked', 'watched', 'favorited']
            
            if feedback_type == 'watched' and rating:
                feedback.engagement_score = rating / 5.0
                feedback.satisfaction_score = rating / 5.0
            
            db.session.commit()
            
            # Trigger profile update
            update_user_profile(user_id, force_update=False)
            
            return True
            
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        db.session.rollback()
    
    return False