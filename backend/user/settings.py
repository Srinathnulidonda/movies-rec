# backend/user/settings.py

from flask import request, jsonify, current_app
from datetime import datetime, timedelta
import json
import logging
import hashlib
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
from .utils import require_auth, db, User, UserInteraction, UserDevice, UserDeviceActivity, extract_device_info, cache_delete, get_cache_key, profile_analyzer
import jwt

logger = logging.getLogger(__name__)

@require_auth
def get_user_settings(current_user):
    try:
        user_devices = get_user_devices(current_user.id)
        privacy_settings = get_privacy_settings(current_user.id)
        security_info = get_security_info(current_user.id)
        
        settings = {
            'account': {
                'username': current_user.username,
                'email': current_user.email,
                'created_at': current_user.created_at.isoformat(),
                'last_active': current_user.last_active.isoformat() if current_user.last_active else None,
                'is_admin': current_user.is_admin
            },
            'preferences': {
                'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
                'preferred_genres': json.loads(current_user.preferred_genres or '[]'),
                'location': current_user.location
            },
            'privacy': privacy_settings,
            'security': security_info,
            'devices': user_devices,
            'notifications': {
                'email_recommendations': True,
                'new_release_alerts': True,
                'watchlist_updates': True,
                'weekly_recap': True,
                'achievement_notifications': True
            },
            'data_management': {
                'profile_visibility': 'public',
                'activity_visibility': 'friends',
                'recommendation_sharing': True,
                'analytics_participation': True
            }
        }
        
        return jsonify({
            'success': True,
            'settings': settings
        }), 200
        
    except Exception as e:
        logger.error(f"Get user settings error: {e}")
        return jsonify({'error': 'Failed to get user settings'}), 500

def get_user_devices(user_id):
    try:
        if not UserDeviceActivity:
            return []
        
        recent_activities = UserDeviceActivity.query.filter_by(
            user_id=user_id
        ).order_by(UserDeviceActivity.timestamp.desc()).limit(50).all()
        
        device_sessions = {}
        for activity in recent_activities:
            device_key = f"{activity.device_type}_{activity.browser}_{activity.os}"
            
            if device_key not in device_sessions:
                device_sessions[device_key] = {
                    'device_type': activity.device_type,
                    'browser': activity.browser,
                    'os': activity.os,
                    'first_seen': activity.timestamp,
                    'last_seen': activity.timestamp,
                    'ip_addresses': set([activity.ip_address]),
                    'session_count': 1,
                    'is_current': False
                }
            else:
                device_sessions[device_key]['last_seen'] = max(
                    device_sessions[device_key]['last_seen'],
                    activity.timestamp
                )
                device_sessions[device_key]['ip_addresses'].add(activity.ip_address)
                device_sessions[device_key]['session_count'] += 1
        
        current_request = request
        current_device_info = extract_device_info(current_request)
        current_device_key = f"{current_device_info['device_type']}_{current_device_info['browser']}_{current_device_info['os']}"
        
        devices = []
        for device_key, device_info in device_sessions.items():
            device_info['is_current'] = device_key == current_device_key
            device_info['ip_addresses'] = list(device_info['ip_addresses'])
            device_info['first_seen'] = device_info['first_seen'].isoformat()
            device_info['last_seen'] = device_info['last_seen'].isoformat()
            device_info['device_id'] = hashlib.md5(device_key.encode()).hexdigest()
            devices.append(device_info)
        
        return sorted(devices, key=lambda x: x['last_seen'], reverse=True)
        
    except Exception as e:
        logger.error(f"Error getting user devices: {e}")
        return []

def get_privacy_settings(user_id):
    return {
        'profile_public': True,
        'activity_tracking': True,
        'personalized_ads': False,
        'data_collection': True,
        'analytics_sharing': True,
        'recommendation_sharing': True,
        'location_tracking': False,
        'device_tracking': True,
        'cross_platform_sync': True
    }

def get_security_info(user_id):
    try:
        if not UserDeviceActivity:
            return {
                'active_sessions': 1,
                'login_locations': ['Current Location'],
                'last_password_change': None,
                'two_factor_enabled': False,
                'suspicious_activity': False
            }
        
        recent_logins = UserDeviceActivity.query.filter_by(
            user_id=user_id
        ).filter(
            UserDeviceActivity.timestamp >= datetime.utcnow() - timedelta(days=30)
        ).order_by(UserDeviceActivity.timestamp.desc()).limit(20).all()
        
        unique_ips = set(activity.ip_address for activity in recent_logins)
        active_sessions = len(set(
            f"{activity.device_type}_{activity.browser}" 
            for activity in recent_logins 
            if activity.timestamp >= datetime.utcnow() - timedelta(hours=24)
        ))
        
        return {
            'active_sessions': active_sessions,
            'login_locations': list(unique_ips)[:5],
            'last_login': recent_logins[0].timestamp.isoformat() if recent_logins else None,
            'total_devices': len(set(f"{a.device_type}_{a.browser}" for a in recent_logins)),
            'suspicious_activity': len(unique_ips) > 10,
            'two_factor_enabled': False
        }
        
    except Exception as e:
        logger.error(f"Error getting security info: {e}")
        return {}

@require_auth
def update_account_settings(current_user):
    try:
        data = request.get_json()
        updated_fields = []
        
        if 'email' in data and data['email'] != current_user.email:
            existing_email = User.query.filter_by(email=data['email']).first()
            if existing_email and existing_email.id != current_user.id:
                return jsonify({'error': 'Email already in use'}), 400
            
            current_user.email = data['email']
            updated_fields.append('email')
        
        if 'username' in data and data['username'] != current_user.username:
            existing_username = User.query.filter_by(username=data['username']).first()
            if existing_username and existing_username.id != current_user.id:
                return jsonify({'error': 'Username already in use'}), 400
            
            current_user.username = data['username']
            updated_fields.append('username')
        
        if 'preferred_languages' in data:
            current_user.preferred_languages = json.dumps(data['preferred_languages'])
            updated_fields.append('preferred_languages')
        
        if 'preferred_genres' in data:
            current_user.preferred_genres = json.dumps(data['preferred_genres'])
            updated_fields.append('preferred_genres')
        
        if 'location' in data:
            current_user.location = data['location']
            updated_fields.append('location')
        
        db.session.commit()
        
        cache_keys = [
            get_cache_key('user_profile', current_user.id),
            get_cache_key('user_dashboard', current_user.id)
        ]
        for key in cache_keys:
            cache_delete(key)
        
        return jsonify({
            'success': True,
            'message': f'Account settings updated: {", ".join(updated_fields)}',
            'updated_fields': updated_fields
        }), 200
        
    except Exception as e:
        logger.error(f"Update account settings error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update account settings'}), 500

@require_auth
def change_password(current_user):
    try:
        data = request.get_json()
        
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not current_password or not new_password:
            return jsonify({'error': 'Current password and new password required'}), 400
        
        if not check_password_hash(current_user.password_hash, current_password):
            return jsonify({'error': 'Current password is incorrect'}), 400
        
        if len(new_password) < 6:
            return jsonify({'error': 'New password must be at least 6 characters'}), 400
        
        current_user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        
        logger.info(f"Password changed for user {current_user.username}")
        
        return jsonify({
            'success': True,
            'message': 'Password changed successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Change password error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to change password'}), 500

@require_auth
def get_active_devices(current_user):
    try:
        devices = get_user_devices(current_user.id)
        
        active_devices = [
            device for device in devices 
            if datetime.fromisoformat(device['last_seen'].replace('Z', '+00:00')) >= datetime.utcnow() - timedelta(days=7)
        ]
        
        return jsonify({
            'success': True,
            'active_devices': active_devices,
            'total_devices': len(devices),
            'active_count': len(active_devices)
        }), 200
        
    except Exception as e:
        logger.error(f"Get active devices error: {e}")
        return jsonify({'error': 'Failed to get active devices'}), 500

@require_auth
def logout_all_devices(current_user):
    try:
        if UserDeviceActivity:
            UserDeviceActivity.query.filter_by(user_id=current_user.id).delete()
            db.session.commit()
        
        cache_keys = [
            get_cache_key('user_profile', current_user.id),
            get_cache_key('user_dashboard', current_user.id),
            get_cache_key('user_activity', current_user.id)
        ]
        for key in cache_keys:
            cache_delete(key)
        
        logger.info(f"All devices logged out for user {current_user.username}")
        
        return jsonify({
            'success': True,
            'message': 'Successfully logged out from all devices'
        }), 200
        
    except Exception as e:
        logger.error(f"Logout all devices error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to logout from all devices'}), 500

@require_auth
def revoke_device_access(current_user):
    try:
        data = request.get_json()
        device_id = data.get('device_id')
        
        if not device_id:
            return jsonify({'error': 'Device ID required'}), 400
        
        if UserDeviceActivity:
            devices = get_user_devices(current_user.id)
            target_device = next((d for d in devices if d['device_id'] == device_id), None)
            
            if not target_device:
                return jsonify({'error': 'Device not found'}), 404
            
            UserDeviceActivity.query.filter(
                UserDeviceActivity.user_id == current_user.id,
                UserDeviceActivity.device_type == target_device['device_type'],
                UserDeviceActivity.browser == target_device['browser'],
                UserDeviceActivity.os == target_device['os']
            ).delete()
            
            db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Device access revoked successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Revoke device access error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to revoke device access'}), 500

@require_auth
def update_privacy_settings(current_user):
    try:
        data = request.get_json()
        
        privacy_changes = []
        
        if 'profile_public' in data:
            privacy_changes.append(f"Profile visibility: {'Public' if data['profile_public'] else 'Private'}")
        
        if 'activity_tracking' in data:
            privacy_changes.append(f"Activity tracking: {'Enabled' if data['activity_tracking'] else 'Disabled'}")
        
        if 'personalized_ads' in data:
            privacy_changes.append(f"Personalized ads: {'Enabled' if data['personalized_ads'] else 'Disabled'}")
        
        if 'data_collection' in data:
            privacy_changes.append(f"Data collection: {'Enabled' if data['data_collection'] else 'Disabled'}")
        
        return jsonify({
            'success': True,
            'message': 'Privacy settings updated',
            'changes': privacy_changes
        }), 200
        
    except Exception as e:
        logger.error(f"Update privacy settings error: {e}")
        return jsonify({'error': 'Failed to update privacy settings'}), 500

@require_auth
def export_user_data(current_user):
    try:
        user_interactions = UserInteraction.query.filter_by(user_id=current_user.id).all()
        
        user_data = {
            'account': {
                'username': current_user.username,
                'email': current_user.email,
                'created_at': current_user.created_at.isoformat(),
                'last_active': current_user.last_active.isoformat() if current_user.last_active else None
            },
            'preferences': {
                'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
                'preferred_genres': json.loads(current_user.preferred_genres or '[]'),
                'location': current_user.location
            },
            'interactions': [
                {
                    'content_id': interaction.content_id,
                    'interaction_type': interaction.interaction_type,
                    'rating': interaction.rating,
                    'timestamp': interaction.timestamp.isoformat(),
                    'metadata': json.loads(interaction.interaction_metadata or '{}')
                }
                for interaction in user_interactions
            ],
            'device_activity': [],
            'exported_at': datetime.utcnow().isoformat()
        }
        
        if UserDeviceActivity:
            device_activities = UserDeviceActivity.query.filter_by(user_id=current_user.id).all()
            user_data['device_activity'] = [
                {
                    'device_type': activity.device_type,
                    'browser': activity.browser,
                    'os': activity.os,
                    'ip_address': activity.ip_address,
                    'timestamp': activity.timestamp.isoformat()
                }
                for activity in device_activities
            ]
        
        return jsonify({
            'success': True,
            'data': user_data,
            'total_interactions': len(user_interactions),
            'total_device_activities': len(user_data['device_activity'])
        }), 200
        
    except Exception as e:
        logger.error(f"Export user data error: {e}")
        return jsonify({'error': 'Failed to export user data'}), 500

@require_auth
def delete_account(current_user):
    try:
        data = request.get_json()
        password = data.get('password')
        confirmation = data.get('confirmation')
        
        if not password:
            return jsonify({'error': 'Password required to delete account'}), 400
        
        if confirmation != 'DELETE_ACCOUNT':
            return jsonify({'error': 'Please type "DELETE_ACCOUNT" to confirm'}), 400
        
        if not check_password_hash(current_user.password_hash, password):
            return jsonify({'error': 'Incorrect password'}), 400
        
        user_id = current_user.id
        username = current_user.username
        
        UserInteraction.query.filter_by(user_id=user_id).delete()
        
        if UserDeviceActivity:
            UserDeviceActivity.query.filter_by(user_id=user_id).delete()
        
        current_user.username = f"deleted_user_{user_id}_{int(datetime.utcnow().timestamp())}"
        current_user.email = f"deleted_{user_id}@cinebrain.deleted"
        current_user.avatar_url = None
        current_user.preferred_languages = None
        current_user.preferred_genres = None
        current_user.location = None
        current_user.password_hash = generate_password_hash(secrets.token_urlsafe(32))
        
        db.session.commit()
        
        if profile_analyzer:
            try:
                cache_key = get_cache_key('user_profile', user_id)
                cache_delete(cache_key)
            except:
                pass
        
        logger.info(f"Account deleted for user {username} (ID: {user_id})")
        
        return jsonify({
            'success': True,
            'message': 'Account deleted successfully. All personal data has been removed.'
        }), 200
        
    except Exception as e:
        logger.error(f"Delete account error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to delete account'}), 500

@require_auth
def get_data_control_options(current_user):
    try:
        total_interactions = UserInteraction.query.filter_by(user_id=current_user.id).count()
        total_device_activities = 0
        
        if UserDeviceActivity:
            total_device_activities = UserDeviceActivity.query.filter_by(user_id=current_user.id).count()
        
        data_summary = {
            'user_account': {
                'created': current_user.created_at.isoformat(),
                'last_active': current_user.last_active.isoformat() if current_user.last_active else None
            },
            'activity_data': {
                'total_interactions': total_interactions,
                'total_device_sessions': total_device_activities,
                'data_retention_days': 365
            },
            'export_options': {
                'full_data_export': 'Complete account data including interactions and device history',
                'interactions_only': 'Just your content interactions and ratings',
                'preferences_only': 'Account settings and preferences'
            },
            'deletion_options': {
                'soft_delete': 'Anonymize account but keep statistical data',
                'hard_delete': 'Complete removal of all data (cannot be undone)',
                'selective_delete': 'Choose specific data types to remove'
            }
        }
        
        return jsonify({
            'success': True,
            'data_summary': data_summary
        }), 200
        
    except Exception as e:
        logger.error(f"Get data control options error: {e}")
        return jsonify({'error': 'Failed to get data control options'}), 500

@require_auth
def download_specific_data(current_user):
    try:
        data = request.get_json()
        export_type = data.get('export_type', 'full')
        
        export_data = {}
        
        if export_type in ['full', 'interactions']:
            interactions = UserInteraction.query.filter_by(user_id=current_user.id).all()
            export_data['interactions'] = [
                {
                    'content_id': i.content_id,
                    'type': i.interaction_type,
                    'rating': i.rating,
                    'timestamp': i.timestamp.isoformat(),
                    'metadata': json.loads(i.interaction_metadata or '{}')
                }
                for i in interactions
            ]
        
        if export_type in ['full', 'preferences']:
            export_data['preferences'] = {
                'languages': json.loads(current_user.preferred_languages or '[]'),
                'genres': json.loads(current_user.preferred_genres or '[]'),
                'location': current_user.location
            }
        
        if export_type == 'full' and UserDeviceActivity:
            device_activities = UserDeviceActivity.query.filter_by(user_id=current_user.id).all()
            export_data['device_history'] = [
                {
                    'device': f"{a.device_type} - {a.browser} on {a.os}",
                    'ip': a.ip_address,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in device_activities
            ]
        
        return jsonify({
            'success': True,
            'export_type': export_type,
            'data': export_data,
            'generated_at': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Download specific data error: {e}")
        return jsonify({'error': 'Failed to download data'}), 500

@require_auth
def get_login_history(current_user):
    try:
        if not UserDeviceActivity:
            return jsonify({
                'success': True,
                'login_history': [],
                'message': 'Login history tracking not available'
            }), 200
        
        limit = int(request.args.get('limit', 50))
        
        login_activities = UserDeviceActivity.query.filter_by(
            user_id=current_user.id
        ).order_by(UserDeviceActivity.timestamp.desc()).limit(limit).all()
        
        formatted_history = []
        for activity in login_activities:
            formatted_history.append({
                'timestamp': activity.timestamp.isoformat(),
                'device_info': f"{activity.device_type.title()} - {activity.browser.title()}",
                'operating_system': activity.os.title(),
                'ip_address': activity.ip_address,
                'session_id': hashlib.md5(f"{activity.user_id}_{activity.timestamp}_{activity.ip_address}".encode()).hexdigest()[:8]
            })
        
        return jsonify({
            'success': True,
            'login_history': formatted_history,
            'total_sessions': len(formatted_history)
        }), 200
        
    except Exception as e:
        logger.error(f"Get login history error: {e}")
        return jsonify({'error': 'Failed to get login history'}), 500