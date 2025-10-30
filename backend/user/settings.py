# user/settings.py
from flask import request, jsonify
from datetime import datetime
import json
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from .utils import require_auth, db

logger = logging.getLogger(__name__)

@require_auth
def get_user_settings(current_user):
    """Get user account settings"""
    try:
        settings = {
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
            'privacy': {
                'profile_visibility': 'public',  # This could be expanded
                'activity_visibility': 'public'
            },
            'notifications': {
                'email_recommendations': True,  # These could be database fields
                'new_release_alerts': True,
                'watchlist_updates': True
            }
        }
        
        return jsonify({
            'success': True,
            'settings': settings
        }), 200
        
    except Exception as e:
        logger.error(f"Get user settings error: {e}")
        return jsonify({'error': 'Failed to get user settings'}), 500

@require_auth
def update_account_settings(current_user):
    """Update account settings"""
    try:
        data = request.get_json()
        updated_fields = []
        
        # Update email if provided
        if 'email' in data and data['email'] != current_user.email:
            # Check if email is already taken
            from .utils import User
            existing_email = User.query.filter_by(email=data['email']).first()
            if existing_email and existing_email.id != current_user.id:
                return jsonify({'error': 'Email already in use'}), 400
            
            current_user.email = data['email']
            updated_fields.append('email')
        
        # Update username if provided
        if 'username' in data and data['username'] != current_user.username:
            # Check if username is already taken
            from .utils import User
            existing_username = User.query.filter_by(username=data['username']).first()
            if existing_username and existing_username.id != current_user.id:
                return jsonify({'error': 'Username already in use'}), 400
            
            current_user.username = data['username']
            updated_fields.append('username')
        
        db.session.commit()
        
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
    """Change user password"""
    try:
        data = request.get_json()
        
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not current_password or not new_password:
            return jsonify({'error': 'Current password and new password required'}), 400
        
        # Verify current password
        if not check_password_hash(current_user.password_hash, current_password):
            return jsonify({'error': 'Current password is incorrect'}), 400
        
        # Validate new password
        if len(new_password) < 6:
            return jsonify({'error': 'New password must be at least 6 characters'}), 400
        
        # Update password
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
def delete_account(current_user):
    """Delete user account (soft delete for now)"""
    try:
        data = request.get_json()
        password = data.get('password')
        
        if not password:
            return jsonify({'error': 'Password required to delete account'}), 400
        
        # Verify password
        if not check_password_hash(current_user.password_hash, password):
            return jsonify({'error': 'Incorrect password'}), 400
        
        # For now, just mark as inactive instead of actual deletion
        # In production, you might want to anonymize data instead
        current_user.username = f"deleted_user_{current_user.id}"
        current_user.email = f"deleted_{current_user.id}@cinebrain.deleted"
        current_user.avatar_url = None
        current_user.preferred_languages = None
        current_user.preferred_genres = None
        current_user.location = None
        
        db.session.commit()
        
        logger.info(f"Account deleted for user ID {current_user.id}")
        
        return jsonify({
            'success': True,
            'message': 'Account deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Delete account error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to delete account'}), 500

@require_auth
def export_user_data(current_user):
    """Export user data"""
    try:
        from .utils import UserInteraction
        
        # Get user interactions
        interactions = UserInteraction.query.filter_by(user_id=current_user.id).all()
        
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
                for interaction in interactions
            ],
            'exported_at': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'success': True,
            'data': user_data
        }), 200
        
    except Exception as e:
        logger.error(f"Export user data error: {e}")
        return jsonify({'error': 'Failed to export user data'}), 500