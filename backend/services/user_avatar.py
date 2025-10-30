#backend/services/user_avatar.py
from flask import Blueprint, request, jsonify
from datetime import datetime
import logging
import jwt
from functools import wraps
import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
import base64
import io
from PIL import Image
import re

avatar_bp = Blueprint('user_avatar', __name__)

logger = logging.getLogger(__name__)

db = None
User = None
app = None

cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET'),
    secure=True
)

def init_user_avatar(flask_app, database, models):
    global db, User, app
    
    app = flask_app
    db = database
    User = models['User']
    
    logger.info("CineBrain avatar service initialized successfully")

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return '', 200
            
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No CineBrain token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': 'Invalid CineBrain token'}), 401
            
            current_user.last_active = datetime.utcnow()
            try:
                db.session.commit()
            except Exception as e:
                logger.warning(f"Failed to update CineBrain user last_active: {e}")
                db.session.rollback()
                
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'CineBrain token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid CineBrain token'}), 401
        except Exception as e:
            logger.error(f"CineBrain authentication error: {e}")
            return jsonify({'error': 'CineBrain authentication failed'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

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

def delete_avatar_from_cloudinary(username, avatar_url):
    try:
        clean_username = clean_username_for_storage(username)
        
        username_public_id = f"cinebrain/avatars/cinebrain_avatar_{clean_username}"
        result = cloudinary.uploader.destroy(username_public_id)
        logger.info(f"Cloudinary deletion result for {username}: {result}")
        
        if result.get('result') != 'ok':
            url_parts = avatar_url.split('/')
            if len(url_parts) > 2:
                public_id_with_ext = url_parts[-1]
                public_id = public_id_with_ext.split('.')[0]
                
                folder_index = -1
                for i, part in enumerate(url_parts):
                    if part == 'cinebrain':
                        folder_index = i
                        break
                
                if folder_index >= 0:
                    folder_path = '/'.join(url_parts[folder_index:-1])
                    full_public_id = f"{folder_path}/{public_id}"
                    
                    result = cloudinary.uploader.destroy(full_public_id)
                    logger.info(f"Cloudinary fallback deletion result: {result}")
                    
    except Exception as e:
        logger.warning(f"Failed to delete avatar from Cloudinary for {username}: {e}")

def is_cloudinary_configured():
    return all([
        os.environ.get('CLOUDINARY_CLOUD_NAME'),
        os.environ.get('CLOUDINARY_API_KEY'),
        os.environ.get('CLOUDINARY_API_SECRET')
    ])

@avatar_bp.route('/api/users/avatar/upload', methods=['POST', 'OPTIONS'])
@require_auth
def upload_avatar(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
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

@avatar_bp.route('/api/users/avatar/delete', methods=['DELETE', 'OPTIONS'])
@require_auth
def delete_avatar(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        if not current_user.avatar_url:
            return jsonify({'error': 'No avatar to delete'}), 400
        
        if 'cloudinary' in current_user.avatar_url:
            delete_avatar_from_cloudinary(current_user.username, current_user.avatar_url)
        
        current_user.avatar_url = None
        db.session.commit()
        
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

@avatar_bp.route('/api/users/avatar/url', methods=['GET', 'OPTIONS'])
@require_auth
def get_avatar_url(current_user):
    if request.method == 'OPTIONS':
        return '', 200
    
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

@avatar_bp.route('/api/users/avatar/health', methods=['GET'])
def avatar_health():
    try:
        health_info = {
            'status': 'healthy',
            'service': 'cinebrain_avatar',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0'
        }
        
        cloudinary_configured = is_cloudinary_configured()
        health_info['cloudinary'] = 'configured' if cloudinary_configured else 'not_configured'
        
        if cloudinary_configured:
            try:
                cloudinary.api.ping()
                health_info['cloudinary_connection'] = 'connected'
            except Exception as e:
                health_info['cloudinary_connection'] = f'error: {str(e)}'
                health_info['status'] = 'degraded'
        
        try:
            if User:
                users_with_avatars = User.query.filter(User.avatar_url.isnot(None)).count()
                total_users = User.query.count()
                
                health_info['avatar_metrics'] = {
                    'users_with_avatars': users_with_avatars,
                    'total_users': total_users,
                    'adoption_rate': round((users_with_avatars / total_users * 100), 1) if total_users > 0 else 0
                }
        except Exception as e:
            health_info['avatar_metrics'] = {'error': str(e)}
        
        health_info['features'] = {
            'image_validation': True,
            'image_processing': True,
            'automatic_resizing': True,
            'format_conversion': True,
            'username_based_storage': True,
            'old_avatar_cleanup': True,
            'cloudinary_integration': cloudinary_configured
        }
        
        health_info['supported_formats'] = ['JPEG', 'PNG', 'WEBP']
        health_info['max_file_size'] = '5MB'
        health_info['output_size'] = '300x300'
        
        return jsonify(health_info), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'cinebrain_avatar',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@avatar_bp.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    allowed_origins = [
        'https://cinebrain.vercel.app',
        'http://127.0.0.1:5500',
        'http://127.0.0.1:5501',
        'http://localhost:3000',
        'http://localhost:5173'
    ]
    
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response

__all__ = ['avatar_bp', 'init_user_avatar']