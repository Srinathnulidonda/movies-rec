# backend/services/auth.py
from flask import Blueprint, request, jsonify
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate
from email.header import Header
import jwt
import os
import logging
from functools import wraps
import re
import threading
import smtplib
import ssl
import uuid
import time
from typing import Dict, Optional, List
import hashlib
import json
import redis
from urllib.parse import urlparse
from collections import defaultdict, Counter
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
auth_bp = Blueprint('auth', __name__)

# Frontend URL configuration
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')
BACKEND_URL = os.environ.get('BACKEND_URL', 'https://backend-app-970m.onrender.com')

# Redis configuration
REDIS_URL = os.environ.get('REDIS_URL', 'redis://red-d2qlbuje5dus73c71qog:xp7inVzgblGCbo9I4taSGLdKUg0xY91I@red-d2qlbuje5dus73c71qog:6379')

# Email validation regex
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# These will be initialized by init_auth function
app = None
db = None
User = None
UserInteraction = None
Content = None
Review = None
mail = None
serializer = None
redis_client = None

# Password reset token salt
PASSWORD_RESET_SALT = 'password-reset-salt-cinebrain-2025'

def init_redis():
    """Initialize Redis connection with better error handling"""
    global redis_client
    try:
        # Parse Redis URL
        url = urlparse(REDIS_URL)
        
        redis_client = redis.StrictRedis(
            host=url.hostname,
            port=url.port,
            password=url.password,
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=10,
            retry_on_timeout=True,
            health_check_interval=30,
            max_connections=20
        )
        
        # Test connection
        redis_client.ping()
        logger.info("✅ Redis connected successfully")
        return redis_client
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        # Create a mock Redis client for fallback
        return MockRedisClient()

class MockRedisClient:
    """Mock Redis client for when Redis is unavailable"""
    def __init__(self):
        self.data = {}
        self.expires = {}
    
    def ping(self):
        return True
    
    def set(self, key, value):
        self.data[key] = value
        return True
    
    def setex(self, key, time, value):
        self.data[key] = value
        self.expires[key] = datetime.utcnow() + timedelta(seconds=time)
        return True
    
    def get(self, key):
        if key in self.expires and self.expires[key] < datetime.utcnow():
            del self.data[key]
            del self.expires[key]
            return None
        return self.data.get(key)
    
    def delete(self, key):
        self.data.pop(key, None)
        self.expires.pop(key, None)
        return True
    
    def incr(self, key):
        self.data[key] = int(self.data.get(key, 0)) + 1
        return self.data[key]
    
    def expire(self, key, time):
        if key in self.data:
            self.expires[key] = datetime.utcnow() + timedelta(seconds=time)
        return True
    
    def lpush(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].insert(0, value)
        return len(self.data[key])
    
    def rpush(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)
        return len(self.data[key])
    
    def lpop(self, key):
        if key in self.data and self.data[key]:
            return self.data[key].pop(0)
        return None
    
    def llen(self, key):
        return len(self.data.get(key, []))

class EnhancedUserAnalytics:
    """Enhanced user analytics with more insights"""
    
    @staticmethod
    def get_comprehensive_user_stats(user_id: int) -> dict:
        """Get comprehensive user statistics with advanced metrics"""
        try:
            base_stats = EnhancedUserAnalytics.get_user_stats(user_id)
            
            # Add advanced metrics
            advanced_stats = {
                'engagement_metrics': EnhancedUserAnalytics._get_engagement_metrics(user_id),
                'content_diversity': EnhancedUserAnalytics._get_content_diversity_score(user_id),
                'temporal_patterns': EnhancedUserAnalytics._get_temporal_patterns(user_id),
                'quality_preferences': EnhancedUserAnalytics._get_quality_preferences(user_id),
                'social_metrics': EnhancedUserAnalytics._get_social_metrics(user_id)
            }
            
            return {**base_stats, **advanced_stats}
            
        except Exception as e:
            logger.error(f"Error getting comprehensive user stats: {e}")
            return {}
    
    @staticmethod
    def get_user_stats(user_id: int) -> dict:
        """Get basic user statistics"""
        try:
            if not UserInteraction:
                return {
                    'total_interactions': 0,
                    'content_watched': 0,
                    'favorites': 0,
                    'watchlist_items': 0,
                    'ratings_given': 0,
                    'average_rating': 0,
                    'most_watched_genre': None,
                    'preferred_content_type': None,
                    'viewing_streak': 0,
                    'discovery_score': 0.0
                }
            
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            
            if not interactions:
                return {
                    'total_interactions': 0,
                    'content_watched': 0,
                    'favorites': 0,
                    'watchlist_items': 0,
                    'ratings_given': 0,
                    'average_rating': 0,
                    'most_watched_genre': None,
                    'preferred_content_type': None,
                    'viewing_streak': 0,
                    'discovery_score': 0.0
                }
            
            # Basic stats
            stats = {
                'total_interactions': len(interactions),
                'content_watched': len([i for i in interactions if i.interaction_type == 'view']),
                'favorites': len([i for i in interactions if i.interaction_type == 'favorite']),
                'watchlist_items': len([i for i in interactions if i.interaction_type == 'watchlist']),
                'ratings_given': len([i for i in interactions if i.interaction_type == 'rating']),
                'likes_given': len([i for i in interactions if i.interaction_type == 'like']),
                'searches_made': len([i for i in interactions if i.interaction_type == 'search'])
            }
            
            # Average rating
            ratings = [i.rating for i in interactions if i.rating is not None]
            stats['average_rating'] = round(sum(ratings) / len(ratings), 1) if ratings else 0
            
            # Get content for analysis
            if Content:
                content_ids = list(set([i.content_id for i in interactions]))
                contents = Content.query.filter(Content.id.in_(content_ids)).all()
                content_map = {c.id: c for c in contents}
                
                # Genre analysis
                genre_counts = defaultdict(int)
                content_type_counts = defaultdict(int)
                language_counts = defaultdict(int)
                
                for interaction in interactions:
                    content = content_map.get(interaction.content_id)
                    if content:
                        # Count content types
                        content_type_counts[content.content_type] += 1
                        
                        # Count genres
                        try:
                            genres = json.loads(content.genres or '[]')
                            for genre in genres:
                                genre_counts[genre.lower()] += 1
                        except (json.JSONDecodeError, TypeError):
                            pass
                        
                        # Count languages
                        try:
                            languages = json.loads(content.languages or '[]')
                            for language in languages:
                                language_counts[language.lower()] += 1
                        except (json.JSONDecodeError, TypeError):
                            pass
                
                stats['most_watched_genre'] = max(genre_counts, key=genre_counts.get) if genre_counts else None
                stats['preferred_content_type'] = max(content_type_counts, key=content_type_counts.get) if content_type_counts else None
                stats['preferred_language'] = max(language_counts, key=language_counts.get) if language_counts else None
                
                # Discovery score
                stats['discovery_score'] = EnhancedUserAnalytics._calculate_discovery_score(interactions, contents)
                
                # Content quality preference
                quality_ratings = [content_map[i.content_id].rating for i in interactions 
                                 if i.content_id in content_map and content_map[i.content_id].rating]
                stats['preferred_content_quality'] = round(sum(quality_ratings) / len(quality_ratings), 1) if quality_ratings else 0
            else:
                stats['most_watched_genre'] = None
                stats['preferred_content_type'] = None
                stats['preferred_language'] = None
                stats['discovery_score'] = 0.0
                stats['preferred_content_quality'] = 0
            
            # Viewing streak
            stats['viewing_streak'] = EnhancedUserAnalytics._calculate_viewing_streak(interactions)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {}
    
    @staticmethod
    def _get_engagement_metrics(user_id: int) -> dict:
        """Calculate user engagement metrics"""
        try:
            if not UserInteraction:
                return {}
            
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            if not interactions:
                return {}
            
            # Calculate engagement score
            interaction_weights = {
                'view': 1,
                'like': 2,
                'favorite': 3,
                'watchlist': 2,
                'rating': 3,
                'share': 2,
                'search': 0.5
            }
            
            total_score = sum(interaction_weights.get(i.interaction_type, 1) for i in interactions)
            engagement_score = min(total_score / 100, 1.0)  # Normalize to 0-1
            
            # Recent activity (last 7 days)
            recent_cutoff = datetime.utcnow() - timedelta(days=7)
            recent_interactions = [i for i in interactions if i.timestamp >= recent_cutoff]
            
            return {
                'engagement_score': round(engagement_score, 3),
                'total_weighted_score': total_score,
                'recent_activity_count': len(recent_interactions),
                'activity_consistency': EnhancedUserAnalytics._calculate_activity_consistency(interactions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating engagement metrics: {e}")
            return {}
    
    @staticmethod
    def _get_content_diversity_score(user_id: int) -> dict:
        """Calculate content diversity metrics"""
        try:
            if not UserInteraction or not Content:
                return {}
            
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            if not interactions:
                return {}
            
            content_ids = [i.content_id for i in interactions]
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            
            if not contents:
                return {}
            
            # Genre diversity
            all_genres = set()
            for content in contents:
                try:
                    genres = json.loads(content.genres or '[]')
                    all_genres.update([g.lower() for g in genres])
                except:
                    pass
            
            # Language diversity
            all_languages = set()
            for content in contents:
                try:
                    languages = json.loads(content.languages or '[]')
                    all_languages.update([l.lower() for l in languages])
                except:
                    pass
            
            # Content type diversity
            content_types = set(content.content_type for content in contents)
            
            return {
                'genre_diversity_count': len(all_genres),
                'language_diversity_count': len(all_languages),
                'content_type_diversity_count': len(content_types),
                'diversity_score': min((len(all_genres) + len(all_languages) + len(content_types)) / 30, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating content diversity: {e}")
            return {}
    
    @staticmethod
    def _get_temporal_patterns(user_id: int) -> dict:
        """Analyze user's temporal viewing patterns"""
        try:
            if not UserInteraction:
                return {}
            
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            if not interactions:
                return {}
            
            # Hour of day analysis
            hour_counts = defaultdict(int)
            day_counts = defaultdict(int)
            
            for interaction in interactions:
                hour_counts[interaction.timestamp.hour] += 1
                day_counts[interaction.timestamp.weekday()] += 1
            
            peak_hour = max(hour_counts, key=hour_counts.get) if hour_counts else None
            peak_day = max(day_counts, key=day_counts.get) if day_counts else None
            
            # Convert peak day to name
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            peak_day_name = day_names[peak_day] if peak_day is not None else None
            
            return {
                'peak_hour': peak_hour,
                'peak_day': peak_day_name,
                'total_active_hours': len(hour_counts),
                'total_active_days': len(day_counts),
                'viewing_pattern': 'night_owl' if peak_hour and peak_hour > 20 else 'early_bird' if peak_hour and peak_hour < 8 else 'regular'
            }
            
        except Exception as e:
            logger.error(f"Error calculating temporal patterns: {e}")
            return {}
    
    @staticmethod
    def _get_quality_preferences(user_id: int) -> dict:
        """Analyze user's quality preferences"""
        try:
            if not UserInteraction or not Content:
                return {}
            
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            if not interactions:
                return {}
            
            content_ids = [i.content_id for i in interactions]
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            
            if not contents:
                return {}
            
            # Rating preferences
            ratings = [c.rating for c in contents if c.rating]
            avg_content_rating = sum(ratings) / len(ratings) if ratings else 0
            
            # Popularity preferences
            popularities = [c.popularity for c in contents if c.popularity]
            avg_popularity = sum(popularities) / len(popularities) if popularities else 0
            
            # Quality tier classification
            high_quality_count = len([c for c in contents if c.rating and c.rating >= 8.0])
            total_with_ratings = len([c for c in contents if c.rating])
            
            quality_preference = 'high' if total_with_ratings > 0 and (high_quality_count / total_with_ratings) > 0.6 else 'balanced'
            
            return {
                'average_content_rating': round(avg_content_rating, 1),
                'average_popularity': round(avg_popularity, 1),
                'quality_preference': quality_preference,
                'high_quality_percentage': round((high_quality_count / total_with_ratings * 100), 1) if total_with_ratings > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating quality preferences: {e}")
            return {}
    
    @staticmethod
    def _get_social_metrics(user_id: int) -> dict:
        """Calculate social engagement metrics"""
        try:
            if not Review:
                return {}
            
            user_reviews = Review.query.filter_by(user_id=user_id).all()
            
            total_reviews = len(user_reviews)
            total_helpful_votes = sum(review.helpful_count for review in user_reviews)
            avg_helpful_votes = total_helpful_votes / total_reviews if total_reviews > 0 else 0
            
            # Review quality score based on helpful votes
            review_quality_score = min(avg_helpful_votes / 10, 1.0) if avg_helpful_votes > 0 else 0
            
            return {
                'total_reviews': total_reviews,
                'total_helpful_votes': total_helpful_votes,
                'average_helpful_votes': round(avg_helpful_votes, 1),
                'review_quality_score': round(review_quality_score, 3),
                'social_contribution_level': 'high' if review_quality_score > 0.7 else 'medium' if review_quality_score > 0.3 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error calculating social metrics: {e}")
            return {}
    
    @staticmethod
    def _calculate_viewing_streak(interactions):
        """Calculate consecutive days with viewing activity"""
        if not interactions:
            return 0
        
        # Group interactions by date
        dates = set()
        for interaction in interactions:
            dates.add(interaction.timestamp.date())
        
        if not dates:
            return 0
        
        # Sort dates and find streak
        sorted_dates = sorted(dates, reverse=True)
        streak = 1
        
        for i in range(1, len(sorted_dates)):
            if (sorted_dates[i-1] - sorted_dates[i]).days == 1:
                streak += 1
            else:
                break
        
        return streak
    
    @staticmethod
    def _calculate_discovery_score(interactions, contents):
        """Calculate how much new/diverse content user explores"""
        if not interactions or not contents:
            return 0.0
        
        # Calculate genre diversity
        all_genres = set()
        for content in contents:
            try:
                genres = json.loads(content.genres or '[]')
                all_genres.update([g.lower() for g in genres])
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Calculate popularity distribution
        popularities = [c.popularity for c in contents if c.popularity]
        avg_popularity = sum(popularities) / len(popularities) if popularities else 100
        
        # Discovery score based on genre diversity and exploring less popular content
        genre_diversity = len(all_genres) / 20.0 if all_genres else 0  # Normalize by expected max genres
        popularity_exploration = max(0, (200 - avg_popularity) / 200) if avg_popularity else 0.5
        
        return min((genre_diversity + popularity_exploration) / 2, 1.0)
    
    @staticmethod
    def _calculate_activity_consistency(interactions):
        """Calculate how consistently user engages with platform"""
        if not interactions or len(interactions) < 2:
            return 0.0
        
        # Group by day
        daily_counts = defaultdict(int)
        for interaction in interactions:
            daily_counts[interaction.timestamp.date()] += 1
        
        if len(daily_counts) < 2:
            return 0.0
        
        # Calculate consistency (lower variance = higher consistency)
        counts = list(daily_counts.values())
        mean_count = sum(counts) / len(counts)
        variance = sum((x - mean_count) ** 2 for x in counts) / len(counts)
        
        # Normalize consistency score (0-1)
        consistency = max(0, 1 - (variance / (mean_count + 1)))
        return round(consistency, 3)

class MultiProviderEmailService:
    """Enhanced email service with multiple providers and better error handling"""
    
    def __init__(self):
        self.providers = self._initialize_providers()
        self.current_provider = 0
        self.redis_client = redis_client
        self.max_retries = 3
        self.circuit_breaker = {}
        
        # Start email workers
        self.start_email_workers()
    
    def _initialize_providers(self) -> List[Dict]:
        """Initialize multiple email providers for redundancy"""
        providers = []
        
        # Gmail SMTP
        gmail_username = os.environ.get('GMAIL_USERNAME', 'projects.srinath@gmail.com')
        gmail_password = os.environ.get('GMAIL_APP_PASSWORD', 'wuus nsow nbee xewv')
        
        if gmail_username and gmail_password:
            providers.append({
                'name': 'gmail',
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': gmail_username,
                'password': gmail_password,
                'from_email': gmail_username,
                'from_name': 'CineBrain',
                'use_tls': True
            })
        
        # Backup SMTP (you can add more providers here)
        backup_smtp = os.environ.get('BACKUP_SMTP_SERVER')
        if backup_smtp:
            providers.append({
                'name': 'backup',
                'smtp_server': backup_smtp,
                'smtp_port': int(os.environ.get('BACKUP_SMTP_PORT', 587)),
                'username': os.environ.get('BACKUP_SMTP_USERNAME'),
                'password': os.environ.get('BACKUP_SMTP_PASSWORD'),
                'from_email': os.environ.get('BACKUP_FROM_EMAIL'),
                'from_name': 'CineBrain',
                'use_tls': True
            })
        
        return providers
    
    def start_email_workers(self):
        """Start background threads for sending emails"""
        def worker():
            while True:
                try:
                    if self.redis_client and hasattr(self.redis_client, 'lpop'):
                        email_json = self.redis_client.lpop('email_queue')
                        if email_json:
                            try:
                                email_data = json.loads(email_json)
                                self._send_email_with_fallback(email_data)
                            except json.JSONDecodeError as e:
                                logger.error(f"Invalid email data in queue: {e}")
                        else:
                            time.sleep(2)
                    else:
                        time.sleep(5)
                except Exception as e:
                    logger.error(f"Email worker error: {e}")
                    time.sleep(5)
        
        # Start 2 worker threads
        for i in range(2):
            thread = threading.Thread(target=worker, daemon=True, name=f"EmailWorker-{i}")
            thread.start()
            logger.info(f"Started email worker thread {i}")
    
    def _send_email_with_fallback(self, email_data: Dict):
        """Try sending email with multiple providers"""
        for attempt in range(len(self.providers)):
            provider = self.providers[(self.current_provider + attempt) % len(self.providers)]
            
            # Check circuit breaker
            if self._is_circuit_open(provider['name']):
                logger.warning(f"Circuit breaker open for {provider['name']}, skipping")
                continue
            
            try:
                self._send_via_provider(email_data, provider)
                logger.info(f"✅ Email sent successfully via {provider['name']} to {email_data['to']}")
                
                # Reset circuit breaker on success
                self._reset_circuit_breaker(provider['name'])
                
                # Update current provider for next email
                self.current_provider = (self.current_provider + attempt) % len(self.providers)
                
                # Store success in Redis
                self._store_email_status(email_data.get('id'), 'sent', provider['name'])
                return
                
            except Exception as e:
                logger.error(f"❌ Failed to send via {provider['name']}: {e}")
                self._record_provider_failure(provider['name'])
                
                # If it's a network error, try next provider immediately
                if 'Network is unreachable' in str(e) or 'Connection refused' in str(e):
                    continue
                
                # For other errors, might be worth retrying same provider
                if attempt == len(self.providers) - 1:  # Last attempt
                    self._handle_final_failure(email_data, str(e))
    
    def _send_via_provider(self, email_data: Dict, provider: Dict):
        """Send email via specific provider"""
        # Check if we can reach the SMTP server first
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((provider['smtp_server'], provider['smtp_port']))
            sock.close()
            
            if result != 0:
                raise ConnectionError(f"Cannot reach SMTP server {provider['smtp_server']}:{provider['smtp_port']}")
        except Exception as e:
            raise ConnectionError(f"Network connectivity check failed: {e}")
        
        # Create message
        msg = MIMEMultipart('alternative')
        
        # Professional headers
        msg['From'] = formataddr((provider['from_name'], provider['from_email']))
        msg['To'] = email_data['to']
        msg['Subject'] = email_data['subject']
        msg['Date'] = formatdate(localtime=True)
        msg['Message-ID'] = f"<{email_data.get('id', uuid.uuid4())}@cinebrain.com>"
        
        # Add content
        text_part = MIMEText(email_data['text'], 'plain', 'utf-8')
        html_part = MIMEText(email_data['html'], 'html', 'utf-8')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Send email
        context = ssl.create_default_context()
        
        with smtplib.SMTP(provider['smtp_server'], provider['smtp_port'], timeout=30) as server:
            server.ehlo()
            if provider.get('use_tls', True):
                server.starttls(context=context)
                server.ehlo()
            server.login(provider['username'], provider['password'])
            server.send_message(msg)
    
    def _is_circuit_open(self, provider_name: str) -> bool:
        """Check if circuit breaker is open for provider"""
        circuit = self.circuit_breaker.get(provider_name, {'failures': 0, 'last_failure': None})
        
        if circuit['failures'] >= 3:
            # Circuit is open for 5 minutes after 3 failures
            if circuit['last_failure'] and (datetime.utcnow() - circuit['last_failure']).seconds < 300:
                return True
            else:
                # Reset after timeout
                self.circuit_breaker[provider_name] = {'failures': 0, 'last_failure': None}
        
        return False
    
    def _record_provider_failure(self, provider_name: str):
        """Record provider failure for circuit breaker"""
        if provider_name not in self.circuit_breaker:
            self.circuit_breaker[provider_name] = {'failures': 0, 'last_failure': None}
        
        self.circuit_breaker[provider_name]['failures'] += 1
        self.circuit_breaker[provider_name]['last_failure'] = datetime.utcnow()
    
    def _reset_circuit_breaker(self, provider_name: str):
        """Reset circuit breaker on successful send"""
        self.circuit_breaker[provider_name] = {'failures': 0, 'last_failure': None}
    
    def _store_email_status(self, email_id: str, status: str, provider: str = None):
        """Store email status in Redis"""
        if self.redis_client and email_id:
            try:
                status_data = {
                    'status': status,
                    'timestamp': datetime.utcnow().isoformat(),
                    'provider': provider
                }
                self.redis_client.setex(f"email_{status}:{email_id}", 86400, json.dumps(status_data))
            except Exception as e:
                logger.error(f"Failed to store email status: {e}")
    
    def _handle_final_failure(self, email_data: Dict, error: str):
        """Handle final failure after all providers failed"""
        logger.error(f"❌ Failed to send email after trying all providers to {email_data['to']}")
        self._store_email_status(email_data.get('id'), 'failed')
        
        # Store in dead letter queue for manual intervention
        if self.redis_client:
            try:
                dead_letter_data = {
                    **email_data,
                    'final_error': error,
                    'failed_at': datetime.utcnow().isoformat()
                }
                self.redis_client.lpush('email_dead_letter_queue', json.dumps(dead_letter_data))
            except Exception as e:
                logger.error(f"Failed to store in dead letter queue: {e}")
    
    def queue_email(self, to: str, subject: str, html: str, text: str, priority: str = 'normal'):
        """Queue email for sending"""
        if not self.providers:
            logger.error("No email providers configured")
            return False
        
        email_id = str(uuid.uuid4())
        email_data = {
            'id': email_id,
            'to': to,
            'subject': subject,
            'html': html,
            'text': text,
            'priority': priority,
            'timestamp': datetime.utcnow().isoformat(),
            'retry_count': 0
        }
        
        try:
            if self.redis_client and hasattr(self.redis_client, 'lpush'):
                if priority == 'high':
                    self.redis_client.lpush('email_queue', json.dumps(email_data))
                else:
                    self.redis_client.rpush('email_queue', json.dumps(email_data))
                
                self._store_email_status(email_id, 'queued')
                logger.info(f"📧 Email queued for {to} - ID: {email_id}")
            else:
                # Fallback: send directly
                logger.warning("Redis not available, sending email directly")
                threading.Thread(
                    target=self._send_email_with_fallback,
                    args=(email_data,),
                    daemon=True
                ).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue email: {e}")
            return False
    
    def get_professional_template(self, content_type: str, **kwargs) -> tuple:
        """Get professional email templates"""
        # Base CSS for all emails
        base_css = """
        <style type="text/css">
            /* Reset and base styles */
            body, table, td, a { -webkit-text-size-adjust: 100%; -ms-text-size-adjust: 100%; }
            table, td { mso-table-lspace: 0pt; mso-table-rspace: 0pt; }
            img { -ms-interpolation-mode: bicubic; border: 0; outline: none; text-decoration: none; }
            
            body {
                margin: 0 !important; padding: 0 !important; width: 100% !important; min-width: 100% !important;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
                font-size: 14px; line-height: 1.6; color: #202124; background-color: #f8f9fa;
            }
            
            .email-wrapper { width: 100%; background-color: #f8f9fa; padding: 40px 20px; }
            .email-container { max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; 
                box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15); overflow: hidden; }
            
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 48px; text-align: center; }
            .header-logo { font-size: 32px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .header-tagline { font-size: 14px; color: rgba(255,255,255,0.95); margin: 8px 0 0 0; font-weight: 400; }
            
            .content { padding: 48px; background-color: #ffffff; }
            h1 { font-size: 24px; font-weight: 400; color: #202124; margin: 0 0 24px; line-height: 1.3; }
            p { margin: 0 0 16px; color: #5f6368; font-size: 14px; line-height: 1.6; }
            
            .btn { display: inline-block; padding: 12px 32px; font-size: 14px; font-weight: 500; text-decoration: none !important;
                text-align: center; border-radius: 24px; margin: 24px 0; cursor: pointer; }
            .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff !important;
                box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4); }
            
            .alert { padding: 16px; border-radius: 8px; margin: 24px 0; font-size: 14px; }
            .alert-warning { background-color: #fef7e0; border-left: 4px solid #fbbc04; color: #ea8600; }
            .alert-error { background-color: #fce8e6; border-left: 4px solid #ea4335; color: #d33b27; }
            
            .code-block { background-color: #f8f9fa; border: 1px solid #dadce0; border-radius: 8px; padding: 16px;
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Courier New', monospace; font-size: 13px; color: #202124;
                word-break: break-all; margin: 16px 0; }
            
            .info-box { background-color: #f8f9fa; border-radius: 8px; padding: 24px; margin: 24px 0; border: 1px solid #e8eaed; }
            .footer { background-color: #f8f9fa; padding: 32px 48px; text-align: center; border-top: 1px solid #e8eaed; }
            .footer-text { font-size: 12px; color: #80868b; margin: 0 0 8px; line-height: 1.5; }
            .footer-link { color: #1a73e8 !important; text-decoration: none; font-size: 12px; margin: 0 12px; }
            .divider { height: 1px; background-color: #e8eaed; margin: 32px 0; }
            
            @media screen and (max-width: 600px) {
                .email-wrapper { padding: 0 !important; }
                .email-container { width: 100% !important; border-radius: 0 !important; }
                .content, .footer, .header { padding: 32px 24px !important; }
                h1 { font-size: 20px !important; }
                .btn { display: block !important; width: 100% !important; }
            }
        </style>
        """
        
        if content_type == 'password_reset':
            return self._get_password_reset_template(base_css, **kwargs)
        elif content_type == 'password_changed':
            return self._get_password_changed_template(base_css, **kwargs)
        elif content_type == 'welcome':
            return self._get_welcome_template(base_css, **kwargs)
        elif content_type == 'email_verification':
            return self._get_email_verification_template(base_css, **kwargs)
        else:
            return self._get_generic_template(base_css, **kwargs)
    
    def _get_password_reset_template(self, base_css: str, **kwargs) -> tuple:
        """Password reset email template"""
        reset_url = kwargs.get('reset_url', '')
        user_name = kwargs.get('user_name', 'there')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reset your password - CineBrain</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header">
                        <h1 class="header-logo">🎬 CineBrain</h1>
                        <p class="header-tagline">Your AI-Powered Entertainment Companion</p>
                    </div>
                    
                    <div class="content">
                        <h1>Reset your password</h1>
                        <p>Hi {user_name},</p>
                        <p>We received a request to reset your CineBrain account password. Click the button below to create a new password:</p>
                        
                        <center>
                            <a href="{reset_url}" class="btn btn-primary">Reset Password</a>
                        </center>
                        
                        <div class="info-box">
                            <p style="margin: 0; font-size: 13px; color: #5f6368;">
                                <strong>Can't click the button?</strong><br>
                                Copy and paste this link into your browser:
                            </p>
                            <div class="code-block">{reset_url}</div>
                        </div>
                        
                        <div class="alert alert-warning">
                            <strong>⏰ This link expires in 1 hour</strong><br>
                            For security reasons, this password reset link will expire soon.
                        </div>
                        
                        <div class="divider"></div>
                        
                        <p style="font-size: 13px; color: #5f6368;">
                            <strong>Didn't request this?</strong><br>
                            If you didn't request a password reset, you can safely ignore this email. Your password won't be changed.
                        </p>
                    </div>
                    
                    <div class="footer">
                        <p class="footer-text">
                            © {datetime.now().year} CineBrain, Inc. All rights reserved.<br>
                            This email was sent to {kwargs.get('user_email', '')}
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"""
Reset your password

Hi {user_name},

We received a request to reset your CineBrain account password.

To reset your password, visit:
{reset_url}

This link expires in 1 hour.

If you didn't request this, you can safely ignore this email.

Best regards,
The CineBrain Team

© {datetime.now().year} CineBrain, Inc.
        """
        
        return html, text
    
    def _get_password_changed_template(self, base_css: str, **kwargs) -> tuple:
        """Password changed confirmation template"""
        user_name = kwargs.get('user_name', 'there')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Password changed - CineBrain</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header" style="background: linear-gradient(135deg, #34a853 0%, #0d8043 100%);">
                        <h1 class="header-logo">✅ Password Changed</h1>
                        <p class="header-tagline">Your account is now secured</p>
                    </div>
                    
                    <div class="content">
                        <h1>Password successfully changed</h1>
                        <p>Hi {user_name},</p>
                        <p>Your CineBrain account password was successfully changed.</p>
                        
                        <center>
                            <a href="{FRONTEND_URL}/login" class="btn btn-primary">Sign in to CineBrain</a>
                        </center>
                        
                        <div class="alert alert-error">
                            <strong>⚠️ Didn't make this change?</strong><br>
                            If you didn't change your password, 
                            <a href="{FRONTEND_URL}/security/recover" style="color: #ea4335; font-weight: bold;">secure your account immediately</a>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p class="footer-text">© {datetime.now().year} CineBrain, Inc.</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"""
Password Changed Successfully

Hi {user_name},

Your CineBrain account password was successfully changed.

If you didn't make this change, secure your account immediately:
{FRONTEND_URL}/security/recover

© {datetime.now().year} CineBrain, Inc.
        """
        
        return html, text
    
    def _get_welcome_template(self, base_css: str, **kwargs) -> tuple:
        """Welcome email template for new users"""
        user_name = kwargs.get('user_name', 'there')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Welcome to CineBrain - CineBrain</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header">
                        <h1 class="header-logo">🎬 Welcome to CineBrain!</h1>
                        <p class="header-tagline">Your AI-Powered Entertainment Journey Begins</p>
                    </div>
                    
                    <div class="content">
                        <h1>Welcome aboard, {user_name}! 🚀</h1>
                        <p>We're thrilled to have you join the CineBrain community. Get ready to discover your next favorite movie, TV show, or anime with our AI-powered recommendations.</p>
                        
                        <h2>🎯 What you can do:</h2>
                        <ul style="color: #5f6368; margin: 16px 0; padding-left: 20px;">
                            <li>Get personalized recommendations based on your taste</li>
                            <li>Discover trending content and hidden gems</li>
                            <li>Build your watchlist and favorites collection</li>
                            <li>Rate and review content to improve recommendations</li>
                            <li>Explore content in multiple languages including Telugu, Hindi, and English</li>
                        </ul>
                        
                        <center>
                            <a href="{FRONTEND_URL}" class="btn btn-primary">Start Exploring</a>
                        </center>
                        
                        <div class="info-box">
                            <p style="margin: 0; font-size: 13px; color: #5f6368;">
                                <strong>💡 Pro Tip:</strong><br>
                                The more you interact with content (like, favorite, rate), the better our AI becomes at understanding your preferences!
                            </p>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p class="footer-text">Happy watching!<br>The CineBrain Team</p>
                        <p class="footer-text">© {datetime.now().year} CineBrain, Inc.</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"""
Welcome to CineBrain!

Hi {user_name},

We're thrilled to have you join the CineBrain community!

What you can do:
- Get personalized recommendations based on your taste
- Discover trending content and hidden gems
- Build your watchlist and favorites collection
- Rate and review content to improve recommendations
- Explore content in multiple languages

Start exploring: {FRONTEND_URL}

Happy watching!
The CineBrain Team

© {datetime.now().year} CineBrain, Inc.
        """
        
        return html, text
    
    def _get_email_verification_template(self, base_css: str, **kwargs) -> tuple:
        """Email verification template"""
        verification_url = kwargs.get('verification_url', '')
        user_name = kwargs.get('user_name', 'there')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Verify your email - CineBrain</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header">
                        <h1 class="header-logo">📧 Verify Your Email</h1>
                        <p class="header-tagline">One more step to complete your registration</p>
                    </div>
                    
                    <div class="content">
                        <h1>Verify your email address</h1>
                        <p>Hi {user_name},</p>
                        <p>Thanks for signing up for CineBrain! To complete your registration and start getting personalized recommendations, please verify your email address.</p>
                        
                        <center>
                            <a href="{verification_url}" class="btn btn-primary">Verify Email</a>
                        </center>
                        
                        <div class="info-box">
                            <p style="margin: 0; font-size: 13px; color: #5f6368;">
                                <strong>Can't click the button?</strong><br>
                                Copy and paste this link into your browser:
                            </p>
                            <div class="code-block">{verification_url}</div>
                        </div>
                        
                        <div class="alert alert-warning">
                            <strong>⏰ This link expires in 24 hours</strong><br>
                            Please verify your email within 24 hours to activate your account.
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p class="footer-text">© {datetime.now().year} CineBrain, Inc.</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"""
Verify your email address

Hi {user_name},

Thanks for signing up for CineBrain! Please verify your email address to complete registration.

Verification link:
{verification_url}

This link expires in 24 hours.

© {datetime.now().year} CineBrain, Inc.
        """
        
        return html, text
    
    def _get_generic_template(self, base_css: str, **kwargs) -> tuple:
        """Generic email template"""
        subject = kwargs.get('subject', 'CineBrain')
        content = kwargs.get('content', '')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>{subject}</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header">
                        <h1 class="header-logo">🎬 CineBrain</h1>
                    </div>
                    <div class="content">{content}</div>
                    <div class="footer">
                        <p class="footer-text">© {datetime.now().year} CineBrain, Inc.</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"{subject}\n\n{content}\n\n© {datetime.now().year} CineBrain, Inc."
        
        return html, text

# Initialize email service
email_service = None

def init_auth(flask_app, database, user_model):
    """Initialize auth module with Flask app and models"""
    global app, db, User, UserInteraction, Content, Review, mail, serializer, email_service, redis_client
    
    app = flask_app
    db = database
    User = user_model
    
    # Try to get additional models for analytics
    try:
        UserInteraction = db.Model.registry._class_registry.get('UserInteraction')
        Content = db.Model.registry._class_registry.get('Content')
        Review = db.Model.registry._class_registry.get('Review')
    except:
        UserInteraction = None
        Content = None
        Review = None
        logger.warning("Additional models not available for analytics")
    
    # Initialize Redis
    redis_client = init_redis()
    
    # Initialize multi-provider email service
    email_service = MultiProviderEmailService()
    
    # Initialize token serializer
    serializer = URLSafeTimedSerializer(app.secret_key)
    
    logger.info("✅ Enhanced auth module initialized with multi-provider email service")

# Rate limiting with Redis
def check_rate_limit(identifier: str, max_requests: int = 5, window: int = 300) -> bool:
    """Check if rate limit is exceeded using Redis"""
    if not redis_client:
        return True
    
    try:
        key = f"rate_limit:{identifier}"
        current_count = redis_client.incr(key)
        
        if current_count == 1:
            redis_client.expire(key, window)
        
        if current_count > max_requests:
            logger.warning(f"Rate limit exceeded for {identifier}: {current_count}/{max_requests}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Rate limit check error: {e}")
        return True

# Enhanced token management
def generate_reset_token(email):
    """Generate a secure password reset token and cache in Redis"""
    token = serializer.dumps(email, salt=PASSWORD_RESET_SALT)
    
    # Cache token in Redis for quick validation
    if redis_client:
        try:
            redis_client.setex(f"reset_token:{token[:20]}", 3600, email)
        except Exception as e:
            logger.error(f"Failed to cache token in Redis: {e}")
    
    return token

def verify_reset_token(token, expiration=3600):
    """Verify password reset token with Redis cache"""
    if redis_client:
        try:
            cached_email = redis_client.get(f"reset_token:{token[:20]}")
            if cached_email:
                email = serializer.loads(token, salt=PASSWORD_RESET_SALT, max_age=expiration)
                if email == cached_email:
                    return email
        except Exception as e:
            logger.error(f"Redis token verification error: {e}")
    
    try:
        email = serializer.loads(token, salt=PASSWORD_RESET_SALT, max_age=expiration)
        return email
    except SignatureExpired:
        return None
    except BadTimeSignature:
        return None

def validate_password(password):
    """Enhanced password validation"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if len(password) > 128:
        return False, "Password cannot exceed 128 characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    
    # Check for common patterns
    common_patterns = ['123456', 'password', 'qwerty', 'abc123', '111111']
    if any(pattern in password.lower() for pattern in common_patterns):
        return False, "Password contains common patterns that are not secure"
    
    return True, "Valid password"

def get_request_info(request):
    """Enhanced request information extraction"""
    # Get real IP address
    ip_address = request.headers.get('X-Forwarded-For', 
                    request.headers.get('X-Real-IP', request.remote_addr))
    if ip_address and ',' in ip_address:
        ip_address = ip_address.split(',')[0].strip()
    
    user_agent = request.headers.get('User-Agent', 'Unknown')
    
    # Enhanced device detection
    device = "Unknown device"
    browser = ""
    
    if 'Mobile' in user_agent or 'Android' in user_agent:
        if 'Android' in user_agent:
            device = "Android device"
        elif 'iPhone' in user_agent:
            device = "iPhone"
        else:
            device = "Mobile device"
    elif 'iPad' in user_agent:
        device = "iPad"
    elif 'Tablet' in user_agent:
        device = "Tablet"
    elif 'Windows' in user_agent:
        device = "Windows PC"
    elif 'Macintosh' in user_agent:
        device = "Mac"
    elif 'Linux' in user_agent:
        device = "Linux PC"
    
    # Enhanced browser detection
    if 'Edg' in user_agent:
        browser = "Microsoft Edge"
    elif 'Chrome' in user_agent and 'Chromium' not in user_agent:
        browser = "Google Chrome"
    elif 'Firefox' in user_agent:
        browser = "Mozilla Firefox"
    elif 'Safari' in user_agent and 'Chrome' not in user_agent:
        browser = "Safari"
    elif 'Opera' in user_agent:
        browser = "Opera"
    
    if browser:
        device = f"{browser} on {device}"
    
    # Try to get location (you'd need a GeoIP service for real data)
    location = "Unknown location"
    
    return ip_address, location, device

# Enhanced Authentication Routes
@auth_bp.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Enhanced validation
        required_fields = ['username', 'email', 'password']
        if not all(field in data and data[field].strip() for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Username validation
        if len(data['username']) < 3 or len(data['username']) > 50:
            return jsonify({'error': 'Username must be between 3 and 50 characters'}), 400
        
        if not re.match(r'^[a-zA-Z0-9_]+$', data['username']):
            return jsonify({'error': 'Username can only contain letters, numbers, and underscores'}), 400
        
        # Email validation
        if not EMAIL_REGEX.match(data['email']):
            return jsonify({'error': 'Please provide a valid email address'}), 400
        
        # Password validation
        is_valid, message = validate_password(data['password'])
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Check if user exists
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        # Enhanced user creation
        user = User(
            username=data['username'].strip(),
            email=data['email'].strip().lower(),
            password_hash=generate_password_hash(data['password']),
            preferred_languages=json.dumps(data.get('preferred_languages', ['english', 'telugu'])),
            preferred_genres=json.dumps(data.get('preferred_genres', [])),
            location=data.get('location', ''),
            avatar_url=data.get('avatar_url', '')
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Generate token
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.secret_key, algorithm='HS256')
        
        # Send welcome email
        if email_service:
            html_content, text_content = email_service.get_professional_template(
                'welcome',
                user_name=user.username,
                user_email=user.email
            )
            
            email_service.queue_email(
                to=user.email,
                subject="Welcome to CineBrain! 🎬",
                html=html_content,
                text=text_content,
                priority='normal'
            )
        
        # Get initial user stats
        stats = EnhancedUserAnalytics.get_comprehensive_user_stats(user.id)
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]'),
                'location': user.location,
                'avatar_url': user.avatar_url,
                'created_at': user.created_at.isoformat(),
                'stats': stats
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@auth_bp.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Missing username or password'}), 400
        
        # Check rate limiting
        if not check_rate_limit(f"login:{data['username']}", max_requests=5, window=300):
            return jsonify({'error': 'Too many login attempts. Please try again in 5 minutes.'}), 429
        
        user = User.query.filter_by(username=data['username']).first()
        
        if not user or not check_password_hash(user.password_hash, data['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Update last active
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        # Generate token
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.secret_key, algorithm='HS256')
        
        # Get comprehensive user stats
        stats = EnhancedUserAnalytics.get_comprehensive_user_stats(user.id)
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]'),
                'location': user.location,
                'avatar_url': user.avatar_url,
                'last_active': user.last_active.isoformat() if user.last_active else None,
                'stats': stats
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

# Password reset routes (existing but enhanced)
@auth_bp.route('/api/auth/forgot-password', methods=['POST', 'OPTIONS'])
def forgot_password():
    """Enhanced password reset with better error handling"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        # Validate email format
        if not email or not EMAIL_REGEX.match(email):
            return jsonify({'error': 'Please provide a valid email address'}), 400
        
        # Check rate limiting
        if not check_rate_limit(f"forgot_password:{email}", max_requests=3, window=600):
            return jsonify({
                'error': 'Too many password reset requests. Please try again in 10 minutes.'
            }), 429
        
        # Check if user exists
        user = User.query.filter_by(email=email).first()
        
        # Always return success to prevent email enumeration
        if user:
            # Generate reset token
            token = generate_reset_token(email)
            reset_url = f"{FRONTEND_URL}/auth/reset-password.html?token={token}"
            
            # Get professional email template
            html_content, text_content = email_service.get_professional_template(
                'password_reset',
                reset_url=reset_url,
                user_name=user.username,
                user_email=email
            )
            
            # Queue email for sending
            success = email_service.queue_email(
                to=email,
                subject="Reset your password - CineBrain",
                html=html_content,
                text=text_content,
                priority='high'
            )
            
            if success:
                logger.info(f"Password reset requested for {email}")
            else:
                logger.error(f"Failed to queue password reset email for {email}")
        
        return jsonify({
            'success': True,
            'message': 'If an account exists with this email, you will receive password reset instructions shortly.'
        }), 200
        
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        return jsonify({'error': 'Failed to process password reset request'}), 500

@auth_bp.route('/api/auth/reset-password', methods=['POST', 'OPTIONS'])
def reset_password():
    """Enhanced password reset with token verification"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        token = data.get('token', '').strip()
        new_password = data.get('password', '')
        confirm_password = data.get('confirmPassword', '')
        
        # Validate input
        if not token:
            return jsonify({'error': 'Reset token is required'}), 400
        
        if not new_password or not confirm_password:
            return jsonify({'error': 'Password and confirmation are required'}), 400
        
        if new_password != confirm_password:
            return jsonify({'error': 'Passwords do not match'}), 400
        
        # Validate password strength
        is_valid, message = validate_password(new_password)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Verify token
        email = verify_reset_token(token)
        if not email:
            return jsonify({'error': 'Invalid or expired reset token'}), 400
        
        # Find user
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Update password
        user.password_hash = generate_password_hash(new_password)
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        # Clear token from Redis cache
        if redis_client:
            try:
                redis_client.delete(f"reset_token:{token[:20]}")
            except:
                pass
        
        # Get request info for security email
        ip_address, location, device = get_request_info(request)
        
        # Send confirmation email
        html_content, text_content = email_service.get_professional_template(
            'password_changed',
            user_name=user.username,
            user_email=email,
            ip_address=ip_address,
            location=location,
            device=device
        )
        
        email_service.queue_email(
            to=email,
            subject="Your password was changed - CineBrain",
            html=html_content,
            text=text_content,
            priority='high'
        )
        
        # Generate a new auth token for immediate login
        auth_token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30),
            'iat': datetime.utcnow()
        }, app.secret_key, algorithm='HS256')
        
        # Get user stats for response
        stats = EnhancedUserAnalytics.get_comprehensive_user_stats(user.id)
        
        logger.info(f"Password reset successful for {email}")
        
        return jsonify({
            'success': True,
            'message': 'Password has been reset successfully',
            'token': auth_token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]'),
                'location': user.location,
                'avatar_url': user.avatar_url,
                'stats': stats
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Reset password error: {e}")
        return jsonify({'error': 'Failed to reset password'}), 500

@auth_bp.route('/api/auth/verify-reset-token', methods=['POST', 'OPTIONS'])
def verify_token():
    """Verify if a reset token is valid"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        token = data.get('token', '').strip()
        
        if not token:
            return jsonify({'valid': False, 'error': 'No token provided'}), 400
        
        email = verify_reset_token(token)
        if email:
            user = User.query.filter_by(email=email).first()
            if user:
                return jsonify({
                    'valid': True,
                    'email': email,
                    'masked_email': email[:3] + '***' + email[email.index('@'):]
                }), 200
            else:
                return jsonify({'valid': False, 'error': 'User not found'}), 400
        else:
            return jsonify({'valid': False, 'error': 'Invalid or expired token'}), 400
            
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return jsonify({'valid': False, 'error': 'Failed to verify token'}), 500

@auth_bp.route('/api/auth/health', methods=['GET'])
def auth_health():
    """Enhanced auth service health check"""
    try:
        health_info = {
            'status': 'healthy',
            'service': 'authentication',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '2.0.0'
        }
        
        # Test database connection
        if User:
            try:
                User.query.limit(1).first()
                health_info['database'] = 'connected'
            except Exception as e:
                health_info['database'] = f'error: {str(e)}'
                health_info['status'] = 'degraded'
        
        # Test Redis connection
        redis_status = 'not_configured'
        if redis_client:
            try:
                redis_client.ping()
                redis_status = 'connected'
                
                # Get Redis stats
                queue_size = redis_client.llen('email_queue') if hasattr(redis_client, 'llen') else 0
                dead_letter_size = redis_client.llen('email_dead_letter_queue') if hasattr(redis_client, 'llen') else 0
                
                health_info['redis_stats'] = {
                    'email_queue_size': queue_size,
                    'dead_letter_queue_size': dead_letter_size
                }
            except Exception as e:
                redis_status = f'error: {str(e)}'
                health_info['status'] = 'degraded'
        
        health_info['redis_status'] = redis_status
        
        # Check email service
        email_configured = email_service is not None
        if email_service:
            provider_count = len(email_service.providers)
            health_info['email_service'] = {
                'configured': True,
                'provider_count': provider_count,
                'providers': [p['name'] for p in email_service.providers],
                'circuit_breaker_status': email_service.circuit_breaker
            }
        else:
            health_info['email_service'] = {'configured': False}
        
        health_info['features'] = {
            'multi_provider_email': True,
            'circuit_breaker': True,
            'rate_limiting': True,
            'enhanced_analytics': True,
            'password_validation': True,
            'welcome_emails': True
        }
        
        return jsonify(health_info), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'authentication',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# CORS headers for all responses
@auth_bp.after_request
def after_request(response):
    """Add CORS headers to responses"""
    origin = request.headers.get('Origin')
    allowed_origins = [FRONTEND_URL, 'http://127.0.0.1:5500', 'http://127.0.0.1:5501', 'http://localhost:3000']
    
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response

# Enhanced authentication decorator
def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return '', 200
            
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
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

# Export everything needed
__all__ = [
    'auth_bp',
    'init_auth', 
    'require_auth',
    'generate_reset_token',
    'verify_reset_token',
    'validate_password',
    'EnhancedUserAnalytics',
    'MultiProviderEmailService'
]