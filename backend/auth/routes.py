# auth/routes.py

from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta
import jwt
import logging

from .service import (
    check_rate_limit, generate_reset_token, verify_reset_token, 
    validate_password, get_request_info, 
    redis_client, app, db, FRONTEND_URL, EMAIL_REGEX
)
from .mail_templates import get_professional_template

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__)

def get_user_model():
    """Get User model dynamically to avoid import timing issues"""
    from .service import User
    if User is None:
        try:
            from app import User as AppUser
            return AppUser
        except ImportError:
            logger.error("Could not import User model")
            return None
    return User

def get_email_service():
    """Get email service dynamically to avoid import timing issues"""
    from .service import email_service
    if email_service is None:
        logger.error("Email service not initialized")
        return None
    return email_service

@auth_bp.route('/api/auth/forgot-password', methods=['POST', 'OPTIONS'])
def forgot_password():
    """Request password reset"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        if not email or not EMAIL_REGEX.match(email):
            return jsonify({'error': 'Please provide a valid email address'}), 400
        
        if not check_rate_limit(f"forgot_password:{email}", max_requests=3, window=600):
            return jsonify({
                'error': 'Too many password reset requests. Please try again in 10 minutes.'
            }), 429
        
        # Get User model dynamically
        User = get_user_model()
        if not User:
            return jsonify({'error': 'Service temporarily unavailable'}), 503
        
        # Get email service dynamically
        email_service = get_email_service()
        if not email_service:
            return jsonify({'error': 'Email service temporarily unavailable'}), 503
        
        user = User.query.filter_by(email=email).first()
        
        if user:
            token = generate_reset_token(email)
            reset_url = f"{FRONTEND_URL}/auth/reset-password.html?token={token}"
            
            html_content, text_content = get_professional_template(
                'password_reset',
                reset_url=reset_url,
                user_name=user.username,
                user_email=email
            )
            
            email_service.queue_email(
                to=email,
                subject="Reset your password - CineBrain",
                html=html_content,
                text=text_content,
                priority='high',
                reset_token=token,
                to_name=user.username
            )
            
            logger.info(f"Password reset requested for {email}")
            
            if not email_service.email_enabled:
                return jsonify({
                    'success': True,
                    'message': 'Password reset link generated. Check logs for the link.',
                    'fallback_mode': True,
                    'reset_url': reset_url
                }), 200
        
        return jsonify({
            'success': True,
            'message': 'If an account exists with this email, you will receive password reset instructions shortly.'
        }), 200
        
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        return jsonify({'error': 'Failed to process password reset request'}), 500

@auth_bp.route('/api/auth/reset-password', methods=['POST', 'OPTIONS'])
def reset_password():
    """Reset password with token"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        token = data.get('token', '').strip()
        new_password = data.get('password', '')
        confirm_password = data.get('confirmPassword', '')
        
        if not token:
            return jsonify({'error': 'Reset token is required'}), 400
        
        if not new_password or not confirm_password:
            return jsonify({'error': 'Password and confirmation are required'}), 400
        
        if new_password != confirm_password:
            return jsonify({'error': 'Passwords do not match'}), 400
        
        is_valid, message = validate_password(new_password)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        email = verify_reset_token(token)
        if not email:
            return jsonify({'error': 'Invalid or expired reset token'}), 400
        
        # Get User model dynamically
        User = get_user_model()
        if not User:
            return jsonify({'error': 'Service temporarily unavailable'}), 503
        
        # Get email service dynamically
        email_service = get_email_service()
        if not email_service:
            return jsonify({'error': 'Email service temporarily unavailable'}), 503
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        user.password_hash = generate_password_hash(new_password)
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        # Clean up token
        if redis_client:
            try:
                redis_client.delete(f"reset_token:{token[:20]}")
            except:
                pass
        
        ip_address, location, device = get_request_info(request)
        
        # Send confirmation email
        html_content, text_content = get_professional_template(
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
            priority='high',
            to_name=user.username
        )
        
        # Generate new auth token
        auth_token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30),
            'iat': datetime.utcnow()
        }, app.secret_key, algorithm='HS256')
        
        logger.info(f"Password reset successful for {email}")
        
        return jsonify({
            'success': True,
            'message': 'Password has been reset successfully',
            'token': auth_token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Reset password error: {e}")
        return jsonify({'error': 'Failed to reset password'}), 500

@auth_bp.route('/api/auth/verify-reset-token', methods=['POST', 'OPTIONS'])
def verify_token():
    """Verify reset token validity"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        token = data.get('token', '').strip()
        
        if not token:
            return jsonify({'valid': False, 'error': 'No token provided'}), 400
        
        email = verify_reset_token(token)
        if email:
            # Get User model dynamically
            User = get_user_model()
            if not User:
                return jsonify({'valid': False, 'error': 'Service temporarily unavailable'}), 503
            
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

@auth_bp.route('/api/auth/fallback-emails', methods=['GET'])
def get_fallback_emails():
    """Admin endpoint to retrieve fallback emails when API fails"""
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Admin authentication required'}), 401
        
        # Get email service dynamically
        email_service = get_email_service()
        if not email_service:
            return jsonify({'error': 'Email service not available'}), 503
        
        emails = email_service.get_fallback_emails(limit=50)
        
        return jsonify({
            'success': True,
            'fallback_emails': emails,
            'email_service_enabled': email_service.email_enabled,
            'count': len(emails)
        }), 200
        
    except Exception as e:
        logger.error(f"Get fallback emails error: {e}")
        return jsonify({'error': 'Failed to retrieve fallback emails'}), 500

@auth_bp.route('/api/auth/email-stats', methods=['GET'])
def get_email_stats():
    """Get email queue statistics"""
    try:
        # Get email service dynamically
        email_service = get_email_service()
        if not email_service:
            return jsonify({'error': 'Email service not available'}), 503
        
        stats = email_service.get_queue_stats()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'email_service_enabled': email_service.email_enabled,
            'service_type': 'brevo',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Get email stats error: {e}")
        return jsonify({'error': 'Failed to retrieve email statistics'}), 500

@auth_bp.route('/api/auth/health', methods=['GET'])
def auth_health():
    """Authentication service health check"""
    try:
        # Test database connection
        User = get_user_model()
        if User:
            User.query.limit(1).first()
        
        # Get email service dynamically
        email_service = get_email_service()
        
        # Check Redis status
        redis_status = 'not_configured'
        redis_stats = {}
        if redis_client:
            try:
                redis_client.ping()
                redis_status = 'connected'
                
                info = redis_client.info()
                redis_stats = {
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory_human': info.get('used_memory_human', 'N/A'),
                    'total_connections_received': info.get('total_connections_received', 0)
                }
            except:
                redis_status = 'disconnected'
        
        # Check email service status
        email_configured = email_service is not None and email_service.is_configured
        email_enabled = email_service.email_enabled if email_service else False
        smtp_enabled = email_service.smtp_enabled if email_service else False
        api_enabled = email_service.api_enabled if email_service else False
        
        # Get queue sizes
        queue_stats = email_service.get_queue_stats() if email_service else {}
        
        return jsonify({
            'status': 'healthy',
            'service': 'authentication',
            'email_service': 'Brevo',
            'email_method': 'SMTP' if smtp_enabled else 'API' if api_enabled else 'None',
            'email_configured': email_configured,
            'email_enabled': email_enabled,
            'smtp_enabled': smtp_enabled,
            'api_enabled': api_enabled,
            'email_queue_size': queue_stats.get('queue_size', 0),
            'fallback_queue_size': queue_stats.get('fallback_queue_size', 0),
            'redis_status': redis_status,
            'redis_stats': redis_stats,
            'frontend_url': FRONTEND_URL,
            'fallback_mode': not email_enabled,
            'user_model_available': User is not None,
            'email_service_available': email_service is not None,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'authentication',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@auth_bp.after_request
def after_request(response):
    """Add CORS headers"""
    origin = request.headers.get('Origin')
    allowed_origins = [FRONTEND_URL, 'http://127.0.0.1:5500', 'http://127.0.0.1:5501']
    
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response