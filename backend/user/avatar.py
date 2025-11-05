# backend/user/avatar.py

from flask import request, jsonify
from datetime import datetime
import logging
import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
import base64
import io
from PIL import Image
import re
from .utils import require_auth, db, User, cache_delete, get_cache_key

logger = logging.getLogger(__name__)

cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET'),
    secure=True
)

def validate_avatar_upload(file_data):
    try:
        if file_data.startswith('data:image/'):
            header, encoded = file_data.split(',', 1)
            file_data = base64.b64decode(encoded)
        
        image = Image.open(io.BytesIO(file_data))
        
        if image.format not in ['JPEG', 'PNG', 'WEBP']:
            return False, "Invalid image format. Only JPEG, PNG, and WEBP are allowed."
        
        if len(file_data) > 5 * 1024 * 1024:
            return False, "Image size too large. Maximum 5MB allowed."
        
        width, height = image.size
        if width > 2048 or height > 2048:
            return False, "Image dimensions too large. Maximum 2048x2048 pixels."
        
        if width < 50 or height < 50:
            return False, "Image too small. Minimum 50x50 pixels required."
        
        return True, "Valid image"
        
    except Exception as e:
        return False, f"Invalid image data: {str(e)}"

def process_avatar_image(file_data):
    try:
        if isinstance(file_data, str) and file_data.startswith('data:image/'):
            header, encoded = file_data.split(',', 1)
            file_data = base64.b64decode(encoded)
        
        image = Image.open(io.BytesIO(file_data))
        
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        
        image = image.resize((300, 300), Image.Resampling.LANCZOS)
        
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=85, optimize=True)
        output.seek(0)
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error processing avatar image: {e}")
        return None

def clean_username_for_storage(username):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', username.lower())

def delete_old_avatar(current_user):
    if not current_user.avatar_url:
        return
        
    if 'cloudinary' in current_user.avatar_url:
        try:
            clean_username = clean_username_for_storage(current_user.username)
            old_public_id = f"cinebrain/avatars/cinebrain_avatar_{clean_username}"
            cloudinary.uploader.destroy(old_public_id)
            
            url_parts = current_user.avatar_url.split('/')
            if len(url_parts) > 2:
                public_id = url_parts[-1].split('.')[0]
                if public_id.startswith('cinebrain_avatar_'):
                    cloudinary.uploader.destroy(f"cinebrain/avatars/{public_id}")
        except Exception as e:
            logger.warning(f"Failed to delete old avatar: {e}")

def upload_avatar_to_cloudinary(processed_image, username):
    clean_username = clean_username_for_storage(username)
    
    upload_result = cloudinary.uploader.upload(
        processed_image,
        folder="cinebrain/avatars",
        public_id=f"cinebrain_avatar_{clean_username}",
        transformation=[
            {'width': 300, 'height': 300, 'crop': 'fill', 'gravity': 'face'},
            {'quality': 'auto:good'},
            {'format': 'jpg'}
        ],
        tags=['cinebrain', 'avatar', f'username_{clean_username}'],
        overwrite=True,
        resource_type="image"
    )
    
    return upload_result

def is_cloudinary_configured():
    return all([
        os.environ.get('CLOUDINARY_CLOUD_NAME'),
        os.environ.get('CLOUDINARY_API_KEY'),
        os.environ.get('CLOUDINARY_API_SECRET')
    ])

@require_auth
def upload_avatar(current_user):
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        if not is_cloudinary_configured():
            return jsonify({'error': 'CineBrain avatar upload not configured'}), 503
        
        image_data = data['image']
        
        is_valid, message = validate_avatar_upload(image_data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        processed_image = process_avatar_image(image_data)
        if not processed_image:
            return jsonify({'error': 'Failed to process image'}), 400
        
        delete_old_avatar(current_user)
        
        try:
            upload_result = upload_avatar_to_cloudinary(processed_image, current_user.username)
            
            current_user.avatar_url = upload_result['secure_url']
            db.session.commit()
            
            cache_key = get_cache_key('user_profile', current_user.id)
            cache_delete(cache_key)
            
            logger.info(f"CineBrain: Avatar uploaded successfully for user {current_user.username} (ID: {current_user.id})")
            
            return jsonify({
                'success': True,
                'message': 'Avatar uploaded successfully',
                'avatar_url': upload_result['secure_url'],
                'username': current_user.username,
                'cloudinary_data': {
                    'public_id': upload_result['public_id'],
                    'version': upload_result['version'],
                    'width': upload_result['width'],
                    'height': upload_result['height'],
                    'format': upload_result['format'],
                    'bytes': upload_result['bytes'],
                    'stored_as': f"cinebrain_avatar_{clean_username_for_storage(current_user.username)}"
                }
            }), 200
            
        except cloudinary.exceptions.Error as e:
            logger.error(f"Cloudinary upload error for user {current_user.username}: {e}")
            return jsonify({'error': 'Failed to upload image to cloud storage'}), 500
            
    except Exception as e:
        logger.error(f"CineBrain avatar upload error for user {current_user.username}: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to upload avatar'}), 500

@require_auth
def delete_avatar(current_user):
    try:
        if not current_user.avatar_url:
            return jsonify({'error': 'No avatar to delete'}), 400
        
        if 'cloudinary' in current_user.avatar_url:
            delete_old_avatar(current_user)
        
        current_user.avatar_url = None
        db.session.commit()
        
        cache_key = get_cache_key('user_profile', current_user.id)
        cache_delete(cache_key)
        
        logger.info(f"CineBrain: Avatar deleted for user {current_user.username} (ID: {current_user.id})")
        
        return jsonify({
            'success': True,
            'message': 'Avatar deleted successfully',
            'username': current_user.username
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain avatar deletion error for user {current_user.username}: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to delete avatar'}), 500

@require_auth
def get_avatar_url(current_user):
    try:
        return jsonify({
            'success': True,
            'avatar_url': current_user.avatar_url,
            'has_avatar': bool(current_user.avatar_url),
            'username': current_user.username
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain get avatar URL error: {e}")
        return jsonify({'error': 'Failed to get avatar URL'}), 500