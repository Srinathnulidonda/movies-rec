# backend/services/users.py
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import logging
import jwt
from functools import wraps
from collections import defaultdict, Counter
from sqlalchemy import func, and_, or_, desc
import cloudinary
import cloudinary.uploader
import requests
import os
import hashlib
import secrets
import string
from urllib.parse import quote
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import ssl
import threading
import time
import pyotp
import qrcode
import io
import base64

users_bp = Blueprint('users', __name__)
logger = logging.getLogger(__name__)

db = None
User = None
Content = None
UserInteraction = None
AnonymousInteraction = None
Review = None
http_session = None
cache = None
app = None

cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME', 'demo'),
    api_key=os.environ.get('CLOUDINARY_API_KEY', 'demo'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET', 'demo')
)

def init_users(flask_app, database, models, services):
    global db, User, Content, UserInteraction, AnonymousInteraction, Review
    global http_session, cache, app
    
    app = flask_app
    db = database
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    AnonymousInteraction = models.get('AnonymousInteraction')
    Review = models.get('Review')
    
    http_session = services['http_session']
    cache = services.get('cache')

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
            
            current_user.last_active = datetime.utcnow()
            db.session.commit()
            
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

class UserAnalytics:
    @staticmethod
    def get_comprehensive_user_stats(user_id: int) -> dict:
        try:
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            
            if not interactions:
                return UserAnalytics._get_default_stats()
            
            stats = {
                'total_interactions': len(interactions),
                'content_watched': len([i for i in interactions if i.interaction_type == 'view']),
                'favorites': len([i for i in interactions if i.interaction_type == 'favorite']),
                'watchlist_items': len([i for i in interactions if i.interaction_type == 'watchlist']),
                'ratings_given': len([i for i in interactions if i.interaction_type == 'rating']),
                'likes_given': len([i for i in interactions if i.interaction_type == 'like']),
                'shares': len([i for i in interactions if i.interaction_type == 'share']),
                'rewatches': len([i for i in interactions if i.interaction_type == 'rewatch'])
            }
            
            ratings = [i.rating for i in interactions if i.rating is not None]
            stats['average_rating'] = round(sum(ratings) / len(ratings), 1) if ratings else 0
            
            content_ids = list(set([i.content_id for i in interactions]))
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            content_map = {c.id: c for c in contents}
            
            genre_counts = defaultdict(int)
            content_type_counts = defaultdict(int)
            language_counts = defaultdict(int)
            total_runtime = 0
            
            for interaction in interactions:
                content = content_map.get(interaction.content_id)
                if content:
                    content_type_counts[content.content_type] += 1
                    
                    if content.runtime:
                        total_runtime += content.runtime
                    
                    try:
                        genres = json.loads(content.genres or '[]')
                        for genre in genres:
                            genre_counts[genre.lower()] += 1
                    except (json.JSONDecodeError, TypeError):
                        pass
                    
                    try:
                        languages = json.loads(content.languages or '[]')
                        for lang in languages:
                            language_counts[lang.lower()] += 1
                    except (json.JSONDecodeError, TypeError):
                        pass
            
            stats['most_watched_genre'] = max(genre_counts, key=genre_counts.get) if genre_counts else None
            stats['preferred_content_type'] = max(content_type_counts, key=content_type_counts.get) if content_type_counts else None
            stats['most_watched_language'] = max(language_counts, key=language_counts.get) if language_counts else None
            
            stats['total_watch_time_minutes'] = total_runtime
            stats['total_watch_time_hours'] = round(total_runtime / 60, 1) if total_runtime else 0
            
            stats['viewing_streak'] = UserAnalytics._calculate_viewing_streak(interactions)
            stats['discovery_score'] = UserAnalytics._calculate_discovery_score(interactions, contents)
            stats['monthly_activity'] = UserAnalytics._get_monthly_activity(interactions)
            stats['genre_distribution'] = dict(genre_counts)
            stats['content_type_distribution'] = dict(content_type_counts)
            stats['language_distribution'] = dict(language_counts)
            
            quality_ratings = [content_map[i.content_id].rating for i in interactions 
                             if i.content_id in content_map and content_map[i.content_id].rating]
            stats['preferred_content_quality'] = round(sum(quality_ratings) / len(quality_ratings), 1) if quality_ratings else 0
            
            stats['binge_sessions'] = UserAnalytics._detect_binge_sessions(interactions, content_map)
            stats['viewing_patterns'] = UserAnalytics._analyze_viewing_patterns(interactions)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting comprehensive user stats: {e}")
            return UserAnalytics._get_default_stats()
    
    @staticmethod
    def _get_default_stats():
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
            'discovery_score': 0.0,
            'total_watch_time_hours': 0,
            'monthly_activity': []
        }
    
    @staticmethod
    def _calculate_viewing_streak(interactions):
        if not interactions:
            return 0
        
        dates = set()
        for interaction in interactions:
            if interaction.interaction_type in ['view', 'rating', 'favorite']:
                dates.add(interaction.timestamp.date())
        
        if not dates:
            return 0
        
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
        if not interactions or not contents:
            return 0.0
        
        all_genres = set()
        all_languages = set()
        content_types = set()
        
        for content in contents:
            content_types.add(content.content_type)
            try:
                genres = json.loads(content.genres or '[]')
                all_genres.update([g.lower() for g in genres])
            except (json.JSONDecodeError, TypeError):
                pass
            
            try:
                languages = json.loads(content.languages or '[]')
                all_languages.update([l.lower() for l in languages])
            except (json.JSONDecodeError, TypeError):
                pass
        
        popularities = [c.popularity for c in contents if c.popularity]
        avg_popularity = sum(popularities) / len(popularities) if popularities else 100
        
        genre_diversity = len(all_genres) / 25.0 if all_genres else 0
        language_diversity = len(all_languages) / 10.0 if all_languages else 0
        content_type_diversity = len(content_types) / 3.0 if content_types else 0
        popularity_exploration = max(0, (300 - avg_popularity) / 300) if avg_popularity else 0.5
        
        return min((genre_diversity + language_diversity + content_type_diversity + popularity_exploration) / 4, 1.0)
    
    @staticmethod
    def _get_monthly_activity(interactions):
        monthly_counts = defaultdict(int)
        
        for interaction in interactions:
            if interaction.interaction_type in ['view', 'rating', 'favorite']:
                month_key = interaction.timestamp.strftime('%Y-%m')
                monthly_counts[month_key] += 1
        
        current_date = datetime.utcnow()
        last_12_months = []
        
        for i in range(12):
            date = current_date - timedelta(days=30*i)
            month_key = date.strftime('%Y-%m')
            last_12_months.append({
                'month': month_key,
                'count': monthly_counts.get(month_key, 0)
            })
        
        return list(reversed(last_12_months))
    
    @staticmethod
    def _detect_binge_sessions(interactions, content_map):
        daily_sessions = defaultdict(list)
        
        for interaction in interactions:
            if interaction.interaction_type in ['view', 'rewatch']:
                date_key = interaction.timestamp.date()
                content = content_map.get(interaction.content_id)
                
                if content:
                    daily_sessions[date_key].append({
                        'content': content,
                        'timestamp': interaction.timestamp,
                        'type': interaction.interaction_type
                    })
        
        binge_sessions = []
        for date, sessions in daily_sessions.items():
            if len(sessions) >= 3:
                content_types = [s['content'].content_type for s in sessions]
                total_runtime = sum([s['content'].runtime or 120 for s in sessions])
                
                binge_sessions.append({
                    'date': date.isoformat(),
                    'content_count': len(sessions),
                    'total_runtime_hours': round(total_runtime / 60, 1),
                    'primary_type': Counter(content_types).most_common(1)[0][0],
                    'session_duration': 'long' if total_runtime > 360 else 'medium'
                })
        
        return sorted(binge_sessions, key=lambda x: x['date'], reverse=True)[:10]
    
    @staticmethod
    def _analyze_viewing_patterns(interactions):
        hourly_patterns = defaultdict(int)
        daily_patterns = defaultdict(int)
        
        for interaction in interactions:
            if interaction.interaction_type in ['view', 'rewatch']:
                hour = interaction.timestamp.hour
                day = interaction.timestamp.weekday()
                
                hourly_patterns[hour] += 1
                daily_patterns[day] += 1
        
        peak_hour = max(hourly_patterns, key=hourly_patterns.get) if hourly_patterns else None
        peak_day = max(daily_patterns, key=daily_patterns.get) if daily_patterns else None
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        return {
            'peak_hour': peak_hour,
            'peak_day': day_names[peak_day] if peak_day is not None else None,
            'hourly_distribution': dict(hourly_patterns),
            'daily_distribution': dict(daily_patterns),
            'is_weekend_viewer': sum([daily_patterns[5], daily_patterns[6]]) > sum([daily_patterns[i] for i in range(5)]),
            'is_night_owl': sum([hourly_patterns[i] for i in range(22, 24)] + [hourly_patterns[i] for i in range(0, 6)]) > sum([hourly_patterns[i] for i in range(6, 22)])
        }

class AchievementSystem:
    ACHIEVEMENTS = {
        'first_watch': {
            'name': 'First Steps',
            'description': 'Watched your first content',
            'icon': '🎬',
            'condition': lambda stats: stats['content_watched'] >= 1,
            'points': 10
        },
        'movie_buff': {
            'name': 'Movie Buff',
            'description': 'Watched 50 movies',
            'icon': '🍿',
            'condition': lambda stats: stats['content_type_distribution'].get('movie', 0) >= 50,
            'points': 100
        },
        'tv_addict': {
            'name': 'TV Series Addict',
            'description': 'Watched 25 TV series',
            'icon': '📺',
            'condition': lambda stats: stats['content_type_distribution'].get('tv', 0) >= 25,
            'points': 75
        },
        'anime_lover': {
            'name': 'Anime Lover',
            'description': 'Watched 30 anime series/movies',
            'icon': '🗾',
            'condition': lambda stats: stats['content_type_distribution'].get('anime', 0) >= 30,
            'points': 80
        },
        'binge_watcher': {
            'name': 'Binge Watcher',
            'description': 'Had 10 binge watching sessions',
            'icon': '🔥',
            'condition': lambda stats: len(stats.get('binge_sessions', [])) >= 10,
            'points': 50
        },
        'critic': {
            'name': 'Critic',
            'description': 'Rated 100 content items',
            'icon': '⭐',
            'condition': lambda stats: stats['ratings_given'] >= 100,
            'points': 75
        },
        'explorer': {
            'name': 'Explorer',
            'description': 'Discovery score above 0.8',
            'icon': '🧭',
            'condition': lambda stats: stats['discovery_score'] >= 0.8,
            'points': 90
        },
        'marathon_runner': {
            'name': 'Marathon Runner',
            'description': 'Watched 500+ hours of content',
            'icon': '🏃',
            'condition': lambda stats: stats['total_watch_time_hours'] >= 500,
            'points': 150
        },
        'streak_master': {
            'name': 'Streak Master',
            'description': 'Maintained 30-day viewing streak',
            'icon': '🔥',
            'condition': lambda stats: stats['viewing_streak'] >= 30,
            'points': 100
        },
        'quality_seeker': {
            'name': 'Quality Seeker',
            'description': 'Average content rating above 8.0',
            'icon': '💎',
            'condition': lambda stats: stats['preferred_content_quality'] >= 8.0,
            'points': 80
        },
        'social_butterfly': {
            'name': 'Social Butterfly',
            'description': 'Shared 50 content items',
            'icon': '🦋',
            'condition': lambda stats: stats.get('shares', 0) >= 50,
            'points': 60
        },
        'completionist': {
            'name': 'Completionist',
            'description': 'Completed profile with all preferences',
            'icon': '✅',
            'condition': lambda stats: True,
            'points': 25
        }
    }
    
    @staticmethod
    def check_achievements(user_id: int) -> dict:
        try:
            stats = UserAnalytics.get_comprehensive_user_stats(user_id)
            
            user = User.query.get(user_id)
            current_achievements = json.loads(getattr(user, 'achievements', None) or '{}')
            
            new_achievements = []
            total_points = current_achievements.get('total_points', 0)
            unlocked = current_achievements.get('unlocked', {})
            
            for achievement_id, achievement in AchievementSystem.ACHIEVEMENTS.items():
                if achievement_id not in unlocked:
                    if achievement['condition'](stats):
                        unlocked[achievement_id] = {
                            'unlocked_at': datetime.utcnow().isoformat(),
                            'name': achievement['name'],
                            'description': achievement['description'],
                            'icon': achievement['icon'],
                            'points': achievement['points']
                        }
                        total_points += achievement['points']
                        new_achievements.append(achievement_id)
            
            achievement_data = {
                'unlocked': unlocked,
                'total_points': total_points,
                'total_unlocked': len(unlocked),
                'total_available': len(AchievementSystem.ACHIEVEMENTS),
                'completion_percentage': round(len(unlocked) / len(AchievementSystem.ACHIEVEMENTS) * 100, 1)
            }
            
            if hasattr(user, 'achievements'):
                user.achievements = json.dumps(achievement_data)
                db.session.commit()
            
            return {
                'achievements': achievement_data,
                'new_achievements': new_achievements
            }
            
        except Exception as e:
            logger.error(f"Error checking achievements: {e}")
            return {'achievements': {}, 'new_achievements': []}

class NotificationService:
    @staticmethod
    def send_email_notification(user_email: str, subject: str, message: str):
        try:
            gmail_user = os.environ.get('GMAIL_USERNAME', 'projects.srinath@gmail.com')
            gmail_password = os.environ.get('GMAIL_APP_PASSWORD', 'wuus nsow nbee xewv')
            
            msg = MIMEMultipart()
            msg['From'] = gmail_user
            msg['To'] = user_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'html'))
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(gmail_user, gmail_password)
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            logger.error(f"Email notification error: {e}")
            return False
    
    @staticmethod
    def create_notification(user_id: int, notification_type: str, title: str, message: str, metadata: dict = None):
        try:
            notification_data = {
                'type': notification_type,
                'title': title,
                'message': message,
                'metadata': metadata or {},
                'created_at': datetime.utcnow().isoformat(),
                'read': False
            }
            
            user = User.query.get(user_id)
            if user:
                current_notifications = json.loads(getattr(user, 'notifications', None) or '[]')
                current_notifications.insert(0, notification_data)
                current_notifications = current_notifications[:50]
                
                if hasattr(user, 'notifications'):
                    user.notifications = json.dumps(current_notifications)
                    db.session.commit()
                
                user_preferences = json.loads(getattr(user, 'notification_preferences', None) or '{}')
                if user_preferences.get('email_notifications', True):
                    NotificationService.send_email_notification(user.email, title, message)
                
                return True
        except Exception as e:
            logger.error(f"Notification creation error: {e}")
            return False

class AvatarService:
    @staticmethod
    def upload_avatar(user_id: int, image_file) -> str:
        try:
            result = cloudinary.uploader.upload(
                image_file,
                folder=f"avatars/{user_id}",
                public_id=f"avatar_{user_id}",
                overwrite=True,
                format="jpg",
                transformation=[
                    {'width': 300, 'height': 300, 'crop': 'fill', 'gravity': 'face'},
                    {'quality': 'auto:good'}
                ]
            )
            return result['secure_url']
        except Exception as e:
            logger.error(f"Cloudinary upload error: {e}")
            return AvatarService.generate_default_avatar(user_id)
    
    @staticmethod
    def generate_default_avatar(user_id: int, username: str = None) -> str:
        try:
            user = User.query.get(user_id) if not username else None
            display_name = username or (user.username if user else f"User{user_id}")
            
            avatar_url = f"https://ui-avatars.com/api/?name={quote(display_name)}&size=300&background=667eea&color=ffffff&format=png&rounded=true&bold=true"
            return avatar_url
        except Exception as e:
            logger.error(f"Default avatar generation error: {e}")
            return "https://ui-avatars.com/api/?name=User&size=300&background=667eea&color=ffffff&format=png&rounded=true&bold=true"

class SecurityService:
    @staticmethod
    def generate_2fa_secret(user_id: int) -> dict:
        try:
            secret = pyotp.random_base32()
            user = User.query.get(user_id)
            
            if user:
                totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
                    name=user.email,
                    issuer_name="CineBrain"
                )
                
                qr = qrcode.QRCode(version=1, box_size=10, border=5)
                qr.add_data(totp_uri)
                qr.make(fit=True)
                
                qr_img = qr.make_image(fill_color="black", back_color="white")
                
                buffer = io.BytesIO()
                qr_img.save(buffer, format='PNG')
                qr_code_data = base64.b64encode(buffer.getvalue()).decode()
                
                return {
                    'secret': secret,
                    'qr_code': f"data:image/png;base64,{qr_code_data}",
                    'manual_entry_key': secret
                }
        except Exception as e:
            logger.error(f"2FA secret generation error: {e}")
            return {}
    
    @staticmethod
    def verify_2fa_token(secret: str, token: str) -> bool:
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)
        except Exception as e:
            logger.error(f"2FA verification error: {e}")
            return False
    
    @staticmethod
    def generate_session_token() -> str:
        return secrets.token_urlsafe(32)

@users_bp.route('/api/user/profile', methods=['GET'])
@require_auth
def get_user_profile(current_user):
    try:
        stats = UserAnalytics.get_comprehensive_user_stats(current_user.id)
        achievements = AchievementSystem.check_achievements(current_user.id)
        
        recent_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id
        ).order_by(UserInteraction.timestamp.desc()).limit(20).all()
        
        recent_activity = []
        for interaction in recent_interactions:
            content = Content.query.get(interaction.content_id)
            if content:
                recent_activity.append({
                    'content_id': content.id,
                    'content_title': content.title,
                    'content_type': content.content_type,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'interaction_type': interaction.interaction_type,
                    'timestamp': interaction.timestamp.isoformat(),
                    'rating': interaction.rating
                })
        
        profile_data = {
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'bio': getattr(current_user, 'bio', ''),
                'is_admin': current_user.is_admin,
                'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
                'preferred_genres': json.loads(current_user.preferred_genres or '[]'),
                'location': current_user.location,
                'avatar_url': current_user.avatar_url or AvatarService.generate_default_avatar(current_user.id, current_user.username),
                'theme_preference': getattr(current_user, 'theme_preference', 'dark'),
                'ui_language': getattr(current_user, 'ui_language', 'en'),
                'region': getattr(current_user, 'region', 'IN'),
                'created_at': current_user.created_at.isoformat(),
                'last_active': current_user.last_active.isoformat() if current_user.last_active else None,
                'two_factor_enabled': getattr(current_user, 'two_factor_enabled', False)
            },
            'stats': stats,
            'achievements': achievements['achievements'],
            'new_achievements': achievements['new_achievements'],
            'recent_activity': recent_activity,
            'profile_completion': {
                'percentage': self._calculate_profile_completion(current_user),
                'missing_fields': self._get_missing_profile_fields(current_user)
            }
        }
        
        return jsonify(profile_data), 200
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return jsonify({'error': 'Failed to get user profile'}), 500

def _calculate_profile_completion(user):
    completion = 0
    
    if user.username:
        completion += 15
    if user.email:
        completion += 15
    if getattr(user, 'bio', ''):
        completion += 10
    if user.avatar_url:
        completion += 10
    if user.preferred_languages and json.loads(user.preferred_languages or '[]'):
        completion += 15
    if user.preferred_genres and json.loads(user.preferred_genres or '[]'):
        completion += 15
    if user.location:
        completion += 10
    if getattr(user, 'theme_preference', ''):
        completion += 5
    if getattr(user, 'region', ''):
        completion += 5
    
    return completion

def _get_missing_profile_fields(user):
    missing = []
    
    if not getattr(user, 'bio', ''):
        missing.append('bio')
    if not user.avatar_url:
        missing.append('avatar')
    if not user.preferred_languages or not json.loads(user.preferred_languages or '[]'):
        missing.append('preferred_languages')
    if not user.preferred_genres or not json.loads(user.preferred_genres or '[]'):
        missing.append('preferred_genres')
    if not user.location:
        missing.append('location')
    
    return missing

@users_bp.route('/api/user/profile', methods=['PUT'])
@require_auth
def update_user_profile(current_user):
    try:
        data = request.get_json()
        
        if 'username' in data and data['username'] != current_user.username:
            existing_user = User.query.filter_by(username=data['username']).first()
            if existing_user:
                return jsonify({'error': 'Username already taken'}), 400
            current_user.username = data['username']
        
        if 'bio' in data:
            current_user.bio = data['bio'][:500]
        
        if 'preferred_languages' in data:
            current_user.preferred_languages = json.dumps(data['preferred_languages'])
        
        if 'preferred_genres' in data:
            current_user.preferred_genres = json.dumps(data['preferred_genres'])
        
        if 'location' in data:
            current_user.location = data['location']
        
        if 'theme_preference' in data:
            current_user.theme_preference = data['theme_preference']
        
        if 'ui_language' in data:
            current_user.ui_language = data['ui_language']
        
        if 'region' in data:
            current_user.region = data['region']
        
        if 'notification_preferences' in data:
            current_user.notification_preferences = json.dumps(data['notification_preferences'])
        
        db.session.commit()
        
        if cache:
            cache.delete(f"user_profile:{current_user.id}")
            cache.delete(f"user_profile_v2:{current_user.id}")
        
        return jsonify({'message': 'Profile updated successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update profile'}), 500

@users_bp.route('/api/user/avatar', methods=['POST'])
@require_auth
def upload_avatar(current_user):
    try:
        if 'avatar' not in request.files:
            return jsonify({'error': 'No avatar file provided'}), 400
        
        avatar_file = request.files['avatar']
        
        if avatar_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
        if not ('.' in avatar_file.filename and avatar_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type'}), 400
        
        if avatar_file.content_length > 5 * 1024 * 1024:
            return jsonify({'error': 'File too large (max 5MB)'}), 400
        
        avatar_url = AvatarService.upload_avatar(current_user.id, avatar_file)
        
        current_user.avatar_url = avatar_url
        db.session.commit()
        
        if cache:
            cache.delete(f"user_profile:{current_user.id}")
        
        return jsonify({
            'message': 'Avatar uploaded successfully',
            'avatar_url': avatar_url
        }), 200
        
    except Exception as e:
        logger.error(f"Error uploading avatar: {e}")
        return jsonify({'error': 'Failed to upload avatar'}), 500

@users_bp.route('/api/user/interactions', methods=['POST'])
@require_auth
def record_interaction(current_user):
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'interaction_type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if data['interaction_type'] == 'remove_watchlist':
            interaction = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=data['content_id'],
                interaction_type='watchlist'
            ).first()
            
            if interaction:
                db.session.delete(interaction)
                db.session.commit()
                return jsonify({'message': 'Removed from watchlist'}), 200
            else:
                return jsonify({'message': 'Content not in watchlist'}), 404
        
        if data['interaction_type'] == 'watchlist':
            existing = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=data['content_id'],
                interaction_type='watchlist'
            ).first()
            
            if existing:
                return jsonify({'message': 'Already in watchlist'}), 200
        
        interaction_metadata = {
            'from_recommendation': data.get('from_recommendation', False),
            'recommendation_score': data.get('recommendation_score'),
            'recommendation_method': data.get('recommendation_method'),
            'device_type': data.get('device_type'),
            'session_length': data.get('session_length'),
            'user_agent': request.headers.get('User-Agent', ''),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=data['content_id'],
            interaction_type=data['interaction_type'],
            rating=data.get('rating'),
            interaction_metadata=json.dumps(interaction_metadata)
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        if cache:
            cache.delete(f"user_profile:{current_user.id}")
            cache.delete(f"user_profile_v2:{current_user.id}")
        
        achievements = AchievementSystem.check_achievements(current_user.id)
        if achievements['new_achievements']:
            for achievement_id in achievements['new_achievements']:
                achievement = AchievementSystem.ACHIEVEMENTS[achievement_id]
                NotificationService.create_notification(
                    current_user.id,
                    'achievement',
                    f'Achievement Unlocked: {achievement["name"]}',
                    achievement['description'],
                    {'achievement_id': achievement_id, 'points': achievement['points']}
                )
        
        return jsonify({
            'message': 'Interaction recorded successfully',
            'new_achievements': achievements['new_achievements']
        }), 201
        
    except Exception as e:
        logger.error(f"Interaction recording error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record interaction'}), 500

@users_bp.route('/api/user/watchlist', methods=['GET'])
@require_auth
def get_watchlist(current_user):
    try:
        sort_by = request.args.get('sort_by', 'added_date')
        order = request.args.get('order', 'desc')
        content_type_filter = request.args.get('type')
        
        watchlist_query = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='watchlist'
        )
        
        if sort_by == 'added_date':
            if order == 'desc':
                watchlist_query = watchlist_query.order_by(UserInteraction.timestamp.desc())
            else:
                watchlist_query = watchlist_query.order_by(UserInteraction.timestamp.asc())
        
        watchlist_interactions = watchlist_query.all()
        content_ids = [interaction.content_id for interaction in watchlist_interactions]
        
        if not content_ids:
            return jsonify({'watchlist': [], 'count': 0}), 200
        
        contents_query = Content.query.filter(Content.id.in_(content_ids))
        
        if content_type_filter:
            contents_query = contents_query.filter(Content.content_type == content_type_filter)
        
        if sort_by == 'title':
            if order == 'desc':
                contents_query = contents_query.order_by(Content.title.desc())
            else:
                contents_query = contents_query.order_by(Content.title.asc())
        elif sort_by == 'rating':
            if order == 'desc':
                contents_query = contents_query.order_by(Content.rating.desc())
            else:
                contents_query = contents_query.order_by(Content.rating.asc())
        elif sort_by == 'release_date':
            if order == 'desc':
                contents_query = contents_query.order_by(Content.release_date.desc())
            else:
                contents_query = contents_query.order_by(Content.release_date.asc())
        
        contents = contents_query.all()
        
        interaction_map = {i.content_id: i for i in watchlist_interactions}
        
        result = []
        for content in contents:
            interaction = interaction_map.get(content.id)
            
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            content_data = {
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
                'youtube_trailer': youtube_url,
                'added_to_watchlist': interaction.timestamp.isoformat() if interaction else None,
                'priority': 'high' if content.is_trending else 'normal'
            }
            
            result.append(content_data)
        
        return jsonify({
            'watchlist': result,
            'count': len(result),
            'sort_by': sort_by,
            'order': order,
            'filter_applied': content_type_filter
        }), 200
        
    except Exception as e:
        logger.error(f"Watchlist error: {e}")
        return jsonify({'error': 'Failed to get watchlist'}), 500

@users_bp.route('/api/user/watchlist/bulk-action', methods=['POST'])
@require_auth
def bulk_watchlist_action(current_user):
    try:
        data = request.get_json()
        
        if 'action' not in data or 'content_ids' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        action = data['action']
        content_ids = data['content_ids']
        
        if action == 'remove':
            UserInteraction.query.filter(
                UserInteraction.user_id == current_user.id,
                UserInteraction.content_id.in_(content_ids),
                UserInteraction.interaction_type == 'watchlist'
            ).delete(synchronize_session=False)
            
            db.session.commit()
            
            return jsonify({
                'message': f'Removed {len(content_ids)} items from watchlist'
            }), 200
        
        elif action == 'mark_watched':
            for content_id in content_ids:
                existing_view = UserInteraction.query.filter_by(
                    user_id=current_user.id,
                    content_id=content_id,
                    interaction_type='view'
                ).first()
                
                if not existing_view:
                    view_interaction = UserInteraction(
                        user_id=current_user.id,
                        content_id=content_id,
                        interaction_type='view',
                        interaction_metadata=json.dumps({'marked_from_watchlist': True})
                    )
                    db.session.add(view_interaction)
            
            db.session.commit()
            
            return jsonify({
                'message': f'Marked {len(content_ids)} items as watched'
            }), 200
        
        else:
            return jsonify({'error': 'Invalid action'}), 400
            
    except Exception as e:
        logger.error(f"Bulk watchlist action error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to perform bulk action'}), 500

@users_bp.route('/api/user/favorites', methods=['GET'])
@require_auth
def get_favorites(current_user):
    try:
        content_type_filter = request.args.get('type')
        sort_by = request.args.get('sort_by', 'added_date')
        
        favorites_query = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='favorite'
        ).order_by(UserInteraction.timestamp.desc())
        
        favorite_interactions = favorites_query.all()
        content_ids = [interaction.content_id for interaction in favorite_interactions]
        
        if not content_ids:
            return jsonify({'favorites': [], 'count': 0}), 200
        
        contents_query = Content.query.filter(Content.id.in_(content_ids))
        
        if content_type_filter:
            contents_query = contents_query.filter(Content.content_type == content_type_filter)
        
        if sort_by == 'rating':
            contents_query = contents_query.order_by(Content.rating.desc())
        elif sort_by == 'title':
            contents_query = contents_query.order_by(Content.title.asc())
        
        contents = contents_query.all()
        
        grouped_favorites = {
            'movies': [],
            'tv_shows': [],
            'anime': []
        }
        
        all_favorites = []
        
        for content in contents:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            content_data = {
                'id': content.id,
                'slug': content.slug,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'languages': json.loads(content.languages or '[]'),
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:200] + '...' if content.overview and len(content.overview) > 200 else content.overview,
                'youtube_trailer': youtube_url,
                'personal_rating': None
            }
            
            user_rating = UserInteraction.query.filter_by(
                user_id=current_user.id,
                content_id=content.id,
                interaction_type='rating'
            ).first()
            
            if user_rating and user_rating.rating:
                content_data['personal_rating'] = user_rating.rating
            
            all_favorites.append(content_data)
            
            if content.content_type == 'movie':
                grouped_favorites['movies'].append(content_data)
            elif content.content_type == 'tv':
                grouped_favorites['tv_shows'].append(content_data)
            elif content.content_type == 'anime':
                grouped_favorites['anime'].append(content_data)
        
        response = {
            'favorites': all_favorites,
            'grouped_favorites': grouped_favorites,
            'count': len(all_favorites),
            'count_by_type': {
                'movies': len(grouped_favorites['movies']),
                'tv_shows': len(grouped_favorites['tv_shows']),
                'anime': len(grouped_favorites['anime'])
            }
        }
        
        if content_type_filter:
            type_map = {'movie': 'movies', 'tv': 'tv_shows', 'anime': 'anime'}
            filtered_type = type_map.get(content_type_filter)
            if filtered_type:
                response['favorites'] = grouped_favorites[filtered_type]
                response['count'] = len(grouped_favorites[filtered_type])
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Favorites error: {e}")
        return jsonify({'error': 'Failed to get favorites'}), 500

@users_bp.route('/api/user/viewing-history', methods=['GET'])
@require_auth
def get_viewing_history(current_user):
    try:
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 50)
        content_type_filter = request.args.get('type')
        
        history_query = UserInteraction.query.filter(
            UserInteraction.user_id == current_user.id,
            UserInteraction.interaction_type.in_(['view', 'rewatch'])
        )
        
        if content_type_filter:
            content_ids = UserInteraction.query.filter(
                UserInteraction.user_id == current_user.id,
                UserInteraction.interaction_type.in_(['view', 'rewatch'])
            ).with_entities(UserInteraction.content_id).distinct().all()
            
            content_ids = [cid[0] for cid in content_ids]
            
            filtered_content_ids = Content.query.filter(
                Content.id.in_(content_ids),
                Content.content_type == content_type_filter
            ).with_entities(Content.id).all()
            
            filtered_content_ids = [cid[0] for cid in filtered_content_ids]
            
            history_query = history_query.filter(
                UserInteraction.content_id.in_(filtered_content_ids)
            )
        
        history_query = history_query.order_by(UserInteraction.timestamp.desc())
        
        offset = (page - 1) * per_page
        history_interactions = history_query.offset(offset).limit(per_page).all()
        total_count = history_query.count()
        
        content_ids = list(set([interaction.content_id for interaction in history_interactions]))
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        content_map = {c.id: c for c in contents}
        
        viewing_history = []
        for interaction in history_interactions:
            content = content_map.get(interaction.content_id)
            if content:
                youtube_url = None
                if content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                history_item = {
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'youtube_trailer': youtube_url,
                    'watched_at': interaction.timestamp.isoformat(),
                    'interaction_type': interaction.interaction_type,
                    'can_rewatch': interaction.interaction_type != 'rewatch'
                }
                
                viewing_history.append(history_item)
        
        return jsonify({
            'viewing_history': viewing_history,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total_count,
                'pages': (total_count + per_page - 1) // per_page
            },
            'filter_applied': content_type_filter
        }), 200
        
    except Exception as e:
        logger.error(f"Viewing history error: {e}")
        return jsonify({'error': 'Failed to get viewing history'}), 500

@users_bp.route('/api/user/viewing-history/clear', methods=['POST'])
@require_auth
def clear_viewing_history(current_user):
    try:
        data = request.get_json()
        content_type = data.get('content_type') if data else None
        
        query = UserInteraction.query.filter(
            UserInteraction.user_id == current_user.id,
            UserInteraction.interaction_type.in_(['view', 'rewatch'])
        )
        
        if content_type:
            content_ids = Content.query.filter_by(content_type=content_type).with_entities(Content.id).all()
            content_ids = [cid[0] for cid in content_ids]
            query = query.filter(UserInteraction.content_id.in_(content_ids))
        
        deleted_count = query.count()
        query.delete(synchronize_session=False)
        
        db.session.commit()
        
        if cache:
            cache.delete(f"user_profile:{current_user.id}")
            cache.delete(f"user_profile_v2:{current_user.id}")
        
        return jsonify({
            'message': f'Cleared {deleted_count} viewing history items',
            'deleted_count': deleted_count
        }), 200
        
    except Exception as e:
        logger.error(f"Clear viewing history error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to clear viewing history'}), 500

@users_bp.route('/api/user/achievements', methods=['GET'])
@require_auth
def get_user_achievements(current_user):
    try:
        achievements = AchievementSystem.check_achievements(current_user.id)
        
        all_achievements = []
        for achievement_id, achievement in AchievementSystem.ACHIEVEMENTS.items():
            achievement_data = {
                'id': achievement_id,
                'name': achievement['name'],
                'description': achievement['description'],
                'icon': achievement['icon'],
                'points': achievement['points'],
                'unlocked': achievement_id in achievements['achievements'].get('unlocked', {}),
                'unlocked_at': None
            }
            
            if achievement_data['unlocked']:
                unlocked_data = achievements['achievements']['unlocked'][achievement_id]
                achievement_data['unlocked_at'] = unlocked_data['unlocked_at']
            
            all_achievements.append(achievement_data)
        
        return jsonify({
            'achievements': all_achievements,
            'summary': achievements['achievements'],
            'recent_unlocked': sorted(
                [a for a in all_achievements if a['unlocked']],
                key=lambda x: x['unlocked_at'] or '',
                reverse=True
            )[:5]
        }), 200
        
    except Exception as e:
        logger.error(f"Achievements error: {e}")
        return jsonify({'error': 'Failed to get achievements'}), 500

@users_bp.route('/api/user/notifications', methods=['GET'])
@require_auth
def get_notifications(current_user):
    try:
        notifications = json.loads(getattr(current_user, 'notifications', None) or '[]')
        
        unread_count = sum(1 for n in notifications if not n.get('read', False))
        
        return jsonify({
            'notifications': notifications,
            'unread_count': unread_count,
            'total_count': len(notifications)
        }), 200
        
    except Exception as e:
        logger.error(f"Notifications error: {e}")
        return jsonify({'error': 'Failed to get notifications'}), 500

@users_bp.route('/api/user/notifications/mark-read', methods=['POST'])
@require_auth
def mark_notifications_read(current_user):
    try:
        data = request.get_json()
        notification_ids = data.get('notification_ids', [])
        
        notifications = json.loads(getattr(current_user, 'notifications', None) or '[]')
        
        if notification_ids:
            for i, notification in enumerate(notifications):
                if i in notification_ids:
                    notification['read'] = True
        else:
            for notification in notifications:
                notification['read'] = True
        
        if hasattr(current_user, 'notifications'):
            current_user.notifications = json.dumps(notifications)
            db.session.commit()
        
        return jsonify({'message': 'Notifications marked as read'}), 200
        
    except Exception as e:
        logger.error(f"Mark notifications read error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to mark notifications as read'}), 500

@users_bp.route('/api/user/settings', methods=['GET'])
@require_auth
def get_user_settings(current_user):
    try:
        settings = {
            'account': {
                'username': current_user.username,
                'email': current_user.email,
                'two_factor_enabled': getattr(current_user, 'two_factor_enabled', False),
                'email_verified': getattr(current_user, 'email_verified', True)
            },
            'preferences': {
                'theme': getattr(current_user, 'theme_preference', 'dark'),
                'ui_language': getattr(current_user, 'ui_language', 'en'),
                'region': getattr(current_user, 'region', 'IN'),
                'default_homepage': getattr(current_user, 'default_homepage', 'trending'),
                'autoplay_trailers': getattr(current_user, 'autoplay_trailers', True),
                'mature_content': getattr(current_user, 'mature_content_enabled', False)
            },
            'notifications': json.loads(getattr(current_user, 'notification_preferences', None) or '{}'),
            'privacy': {
                'profile_visibility': getattr(current_user, 'profile_visibility', 'public'),
                'activity_visibility': getattr(current_user, 'activity_visibility', 'friends'),
                'recommendation_sharing': getattr(current_user, 'recommendation_sharing', True)
            }
        }
        
        if not settings['notifications']:
            settings['notifications'] = {
                'email_notifications': True,
                'new_releases': True,
                'trending_updates': True,
                'personalized_recommendations': True,
                'achievement_unlocks': True,
                'weekly_digest': True
            }
        
        return jsonify(settings), 200
        
    except Exception as e:
        logger.error(f"Get user settings error: {e}")
        return jsonify({'error': 'Failed to get settings'}), 500

@users_bp.route('/api/user/settings', methods=['PUT'])
@require_auth
def update_user_settings(current_user):
    try:
        data = request.get_json()
        
        if 'preferences' in data:
            prefs = data['preferences']
            if 'theme' in prefs:
                current_user.theme_preference = prefs['theme']
            if 'ui_language' in prefs:
                current_user.ui_language = prefs['ui_language']
            if 'region' in prefs:
                current_user.region = prefs['region']
            if 'default_homepage' in prefs:
                current_user.default_homepage = prefs['default_homepage']
            if 'autoplay_trailers' in prefs:
                current_user.autoplay_trailers = prefs['autoplay_trailers']
            if 'mature_content' in prefs:
                current_user.mature_content_enabled = prefs['mature_content']
        
        if 'notifications' in data:
            current_user.notification_preferences = json.dumps(data['notifications'])
        
        if 'privacy' in data:
            privacy = data['privacy']
            if 'profile_visibility' in privacy:
                current_user.profile_visibility = privacy['profile_visibility']
            if 'activity_visibility' in privacy:
                current_user.activity_visibility = privacy['activity_visibility']
            if 'recommendation_sharing' in privacy:
                current_user.recommendation_sharing = privacy['recommendation_sharing']
        
        db.session.commit()
        
        return jsonify({'message': 'Settings updated successfully'}), 200
        
    except Exception as e:
        logger.error(f"Update user settings error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update settings'}), 500

@users_bp.route('/api/user/security/2fa/setup', methods=['POST'])
@require_auth
def setup_2fa(current_user):
    try:
        if getattr(current_user, 'two_factor_enabled', False):
            return jsonify({'error': '2FA is already enabled'}), 400
        
        secret_data = SecurityService.generate_2fa_secret(current_user.id)
        
        if hasattr(current_user, 'two_factor_secret'):
            current_user.two_factor_secret = secret_data['secret']
            db.session.commit()
        
        return jsonify({
            'qr_code': secret_data['qr_code'],
            'manual_entry_key': secret_data['manual_entry_key'],
            'message': 'Scan the QR code with your authenticator app'
        }), 200
        
    except Exception as e:
        logger.error(f"2FA setup error: {e}")
        return jsonify({'error': 'Failed to setup 2FA'}), 500

@users_bp.route('/api/user/security/2fa/verify', methods=['POST'])
@require_auth
def verify_2fa_setup(current_user):
    try:
        data = request.get_json()
        token = data.get('token')
        
        if not token:
            return jsonify({'error': 'Token required'}), 400
        
        secret = getattr(current_user, 'two_factor_secret', '')
        
        if SecurityService.verify_2fa_token(secret, token):
            current_user.two_factor_enabled = True
            db.session.commit()
            
            backup_codes = [SecurityService.generate_session_token()[:8] for _ in range(10)]
            current_user.backup_codes = json.dumps(backup_codes)
            db.session.commit()
            
            return jsonify({
                'message': '2FA enabled successfully',
                'backup_codes': backup_codes
            }), 200
        else:
            return jsonify({'error': 'Invalid token'}), 400
            
    except Exception as e:
        logger.error(f"2FA verification error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to verify 2FA'}), 500

@users_bp.route('/api/user/security/2fa/disable', methods=['POST'])
@require_auth
def disable_2fa(current_user):
    try:
        data = request.get_json()
        token = data.get('token')
        password = data.get('password')
        
        if not token or not password:
            return jsonify({'error': 'Token and password required'}), 400
        
        if not check_password_hash(current_user.password_hash, password):
            return jsonify({'error': 'Invalid password'}), 400
        
        secret = getattr(current_user, 'two_factor_secret', '')
        
        if SecurityService.verify_2fa_token(secret, token):
            current_user.two_factor_enabled = False
            current_user.two_factor_secret = None
            current_user.backup_codes = None
            db.session.commit()
            
            return jsonify({'message': '2FA disabled successfully'}), 200
        else:
            return jsonify({'error': 'Invalid token'}), 400
            
    except Exception as e:
        logger.error(f"2FA disable error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to disable 2FA'}), 500

@users_bp.route('/api/user/security/sessions', methods=['GET'])
@require_auth
def get_user_sessions(current_user):
    try:
        current_session = {
            'id': 'current',
            'device': request.headers.get('User-Agent', 'Unknown'),
            'ip_address': request.remote_addr,
            'last_active': datetime.utcnow().isoformat(),
            'is_current': True
        }
        
        return jsonify({
            'sessions': [current_session],
            'total_sessions': 1
        }), 200
        
    except Exception as e:
        logger.error(f"Get sessions error: {e}")
        return jsonify({'error': 'Failed to get sessions'}), 500

@users_bp.route('/api/user/security/password', methods=['PUT'])
@require_auth
def change_password(current_user):
    try:
        data = request.get_json()
        
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not current_password or not new_password:
            return jsonify({'error': 'Current and new password required'}), 400
        
        if not check_password_hash(current_user.password_hash, current_password):
            return jsonify({'error': 'Current password is incorrect'}), 400
        
        if len(new_password) < 8:
            return jsonify({'error': 'New password must be at least 8 characters'}), 400
        
        current_user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        
        return jsonify({'message': 'Password changed successfully'}), 200
        
    except Exception as e:
        logger.error(f"Change password error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to change password'}), 500

@users_bp.route('/api/user/account/delete', methods=['POST'])
@require_auth
def delete_account(current_user):
    try:
        data = request.get_json()
        password = data.get('password')
        confirmation = data.get('confirmation')
        
        if not password or confirmation != 'DELETE':
            return jsonify({'error': 'Password and confirmation required'}), 400
        
        if not check_password_hash(current_user.password_hash, password):
            return jsonify({'error': 'Invalid password'}), 400
        
        UserInteraction.query.filter_by(user_id=current_user.id).delete()
        
        if Review:
            Review.query.filter_by(user_id=current_user.id).delete()
        
        db.session.delete(current_user)
        db.session.commit()
        
        return jsonify({'message': 'Account deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Delete account error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to delete account'}), 500

@users_bp.route('/api/user/analytics', methods=['GET'])
@require_auth
def get_user_analytics(current_user):
    try:
        stats = UserAnalytics.get_comprehensive_user_stats(current_user.id)
        achievements = AchievementSystem.check_achievements(current_user.id)
        
        interactions = UserInteraction.query.filter_by(user_id=current_user.id).all()
        
        hourly_activity = defaultdict(int)
        daily_activity = defaultdict(int)
        monthly_activity = defaultdict(int)
        
        for interaction in interactions:
            if interaction.interaction_type in ['view', 'rating', 'favorite']:
                hour = interaction.timestamp.hour
                day = interaction.timestamp.strftime('%A')
                month = interaction.timestamp.strftime('%Y-%m')
                
                hourly_activity[hour] += 1
                daily_activity[day] += 1
                monthly_activity[month] += 1
        
        analytics_data = {
            'user_stats': stats,
            'achievements_summary': achievements['achievements'],
            'viewing_patterns': {
                'hourly_activity': dict(hourly_activity),
                'daily_activity': dict(daily_activity),
                'monthly_activity': dict(monthly_activity),
                'most_active_hour': max(hourly_activity, key=hourly_activity.get) if hourly_activity else None,
                'most_active_day': max(daily_activity, key=daily_activity.get) if daily_activity else None
            },
            'engagement_metrics': {
                'profile_completion': _calculate_profile_completion(current_user),
                'activity_score': min(stats['total_interactions'] / 10, 100),
                'diversity_score': round(stats['discovery_score'] * 100, 1),
                'consistency_score': min(stats['viewing_streak'] * 3, 100)
            },
            'recommendations': {
                'improve_profile': _get_profile_improvement_suggestions(current_user, stats),
                'explore_content': _get_content_exploration_suggestions(stats)
            }
        }
        
        return jsonify(analytics_data), 200
        
    except Exception as e:
        logger.error(f"Error getting user analytics: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500

def _get_profile_improvement_suggestions(user, stats):
    suggestions = []
    
    completion = _calculate_profile_completion(user)
    if completion < 80:
        suggestions.append("Complete your profile for better recommendations")
    
    if stats['ratings_given'] < 10:
        suggestions.append("Rate more content to improve recommendation accuracy")
    
    if stats['discovery_score'] < 0.3:
        suggestions.append("Try exploring different genres and languages")
    
    if stats['viewing_streak'] < 7:
        suggestions.append("Build a viewing streak by watching content regularly")
    
    return suggestions[:3]

def _get_content_exploration_suggestions(stats):
    suggestions = []
    
    content_types = stats.get('content_type_distribution', {})
    
    if 'anime' not in content_types or content_types.get('anime', 0) < 5:
        suggestions.append("Explore anime for unique storytelling experiences")
    
    if 'tv' not in content_types or content_types.get('tv', 0) < 10:
        suggestions.append("Try TV series for deeper character development")
    
    languages = stats.get('language_distribution', {})
    if 'hindi' not in languages:
        suggestions.append("Discover Bollywood movies for great Indian cinema")
    
    if 'telugu' not in languages:
        suggestions.append("Check out Telugu films for regional masterpieces")
    
    return suggestions[:3]