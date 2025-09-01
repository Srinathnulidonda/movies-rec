from flask import Blueprint, request, jsonify
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import jwt
import os
import logging
from functools import wraps
import re

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
auth_bp = Blueprint('auth', __name__)

# Frontend URL configuration
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')
BACKEND_URL = os.environ.get('BACKEND_URL', 'https://backend-app-970m.onrender.com')

# Email validation regex
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# These will be initialized by init_auth function
app = None
db = None
User = None
mail = None
serializer = None

# Password reset token salt
PASSWORD_RESET_SALT = 'password-reset-salt-cinebrain-2025'

def init_auth(flask_app, database, user_model):
    """Initialize auth module with Flask app and models"""
    global app, db, User, mail, serializer
    
    app = flask_app
    db = database
    User = user_model
    
    # Configure Flask-Mail
    app.config['MAIL_SERVER'] = 'smtp.gmail.com'
    app.config['MAIL_PORT'] = 587
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USE_SSL'] = False
    app.config['MAIL_USERNAME'] = os.environ.get('GMAIL_USERNAME', 'projects.srinath@gmail.com')
    app.config['MAIL_PASSWORD'] = os.environ.get('GMAIL_APP_PASSWORD', 'wuus nsow nbee xewv')
    app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('GMAIL_USERNAME', 'mail@cinebrain.com')
    
    # Initialize Flask-Mail
    mail = Mail(app)
    
    # Initialize token serializer
    serializer = URLSafeTimedSerializer(app.secret_key)

# Helper functions
def generate_reset_token(email):
    """Generate a secure password reset token"""
    return serializer.dumps(email, salt=PASSWORD_RESET_SALT)

def verify_reset_token(token, expiration=3600):
    """Verify password reset token (expires in 1 hour by default)"""
    try:
        email = serializer.loads(token, salt=PASSWORD_RESET_SALT, max_age=expiration)
        return email
    except SignatureExpired:
        return None  # Token expired
    except BadTimeSignature:
        return None  # Invalid token

def send_reset_email(user_email, reset_token):
    """Send password reset email with frontend URL"""
    try:
        # Create reset URL pointing to frontend
        reset_url = f"{FRONTEND_URL}/auth/reset.html?token={reset_token}"
        
        msg = Message(
            subject='Password Reset Request - CineBrain',
            recipients=[user_email]
        )
        
        # HTML email body with matching styling
        msg.html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Inter:wght@300;400;500;600;700&display=swap');
                
                body {{
                    font-family: 'Inter', sans-serif;
                    line-height: 1.6;
                    color: #ffffff;
                    margin: 0;
                    padding: 0;
                    background: #000000;
                }}
                
                .wrapper {{
                    background: #000000;
                    padding: 40px 20px;
                }}
                
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    background: #1a1a1a;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 16px;
                    overflow: hidden;
                    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
                }}
                
                .header {{
                    background: linear-gradient(135deg, #E50914 0%, #8B5CF6 100%);
                    padding: 40px 30px;
                    text-align: center;
                }}
                
                .brand-name {{
                    font-family: 'Bangers', cursive;
                    font-size: 48px;
                    color: white;
                    margin: 0;
                    letter-spacing: 2px;
                    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
                }}
                
                .tagline {{
                    font-size: 14px;
                    color: rgba(255, 255, 255, 0.9);
                    margin-top: 8px;
                }}
                
                .content {{
                    padding: 40px 30px;
                    background: #1a1a1a;
                }}
                
                .greeting {{
                    font-size: 24px;
                    font-weight: 600;
                    margin-bottom: 20px;
                    color: #ffffff;
                }}
                
                .message {{
                    color: #b3b3b3;
                    font-size: 16px;
                    line-height: 1.6;
                    margin-bottom: 30px;
                }}
                
                .button-container {{
                    text-align: center;
                    margin: 30px 0;
                }}
                
                .reset-button {{
                    display: inline-block;
                    padding: 14px 32px;
                    background: linear-gradient(135deg, #E50914, #8B5CF6);
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 16px;
                    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3);
                    transition: all 0.3s ease;
                }}
                
                .reset-button:hover {{
                    box-shadow: 0 6px 30px rgba(139, 92, 246, 0.5);
                    transform: translateY(-2px);
                }}
                
                .divider {{
                    height: 1px;
                    background: rgba(255, 255, 255, 0.1);
                    margin: 30px 0;
                }}
                
                .info-box {{
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                    padding: 16px;
                    margin: 20px 0;
                }}
                
                .info-box p {{
                    margin: 0;
                    color: #b3b3b3;
                    font-size: 14px;
                }}
                
                .link-text {{
                    word-break: break-all;
                    color: #8B5CF6;
                    font-size: 12px;
                    margin-top: 8px;
                }}
                
                .security-note {{
                    background: rgba(239, 68, 68, 0.1);
                    border: 1px solid rgba(239, 68, 68, 0.2);
                    border-radius: 8px;
                    padding: 16px;
                    margin: 20px 0;
                    color: #b3b3b3;
                    font-size: 14px;
                }}
                
                .footer {{
                    background: #121212;
                    padding: 30px;
                    text-align: center;
                    color: #666;
                    font-size: 14px;
                    border-top: 1px solid rgba(255, 255, 255, 0.1);
                }}
                
                .footer a {{
                    color: #8B5CF6;
                    text-decoration: none;
                }}
                
                .icon {{
                    display: inline-block;
                    margin-right: 8px;
                }}
            </style>
        </head>
        <body>
            <div class="wrapper">
                <div class="container">
                    <div class="header">
                        <h1 class="brand-name">CineBrain</h1>
                        <p class="tagline">Your Personal Entertainment Companion</p>
                    </div>
                    <div class="content">
                        <h2 class="greeting">Password Reset Request 🔐</h2>
                        <p class="message">
                            We received a request to reset the password for your CineBrain account. 
                            If you made this request, click the button below to create a new password:
                        </p>
                        
                        <div class="button-container">
                            <a href="{reset_url}" class="reset-button">Reset My Password</a>
                        </div>
                        
                        <div class="divider"></div>
                        
                        <div class="info-box">
                            <p><strong>Can't click the button?</strong> Copy and paste this link into your browser:</p>
                            <p class="link-text">{reset_url}</p>
                        </div>
                        
                        <div class="security-note">
                            <p><span class="icon">⏰</span><strong>Important:</strong> This link will expire in 1 hour for security reasons.</p>
                            <p><span class="icon">🔒</span><strong>Security tip:</strong> If you didn't request this password reset, you can safely ignore this email. Your password won't be changed.</p>
                        </div>
                    </div>
                    <div class="footer">
                        <p>This email was sent by CineBrain - Your AI-powered movie recommendation platform</p>
                        <p>Need help? <a href="{FRONTEND_URL}/support">Contact our support team</a></p>
                        <p>&copy; 2025 CineBrain. All rights reserved.</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Plain text fallback
        msg.body = f"""
Hello!

We received a request to reset the password for your CineBrain account.

To reset your password, please click on the following link:
{reset_url}

This link will expire in 1 hour for security reasons.

If you didn't request this password reset, you can safely ignore this email. Your password won't be changed.

Best regards,
The CineBrain Team

---
CineBrain - Your AI-powered movie recommendation platform
© 2025 CineBrain. All rights reserved.
        """
        
        mail.send(msg)
        logger.info(f"Password reset email sent to {user_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send reset email to {user_email}: {e}")
        return False

def send_password_changed_email(user_email, user_name=None):
    """Send password changed confirmation email"""
    try:
        msg = Message(
            subject='Password Changed Successfully - CineBrain',
            recipients=[user_email]
        )
        
        # HTML email body with matching styling
        msg.html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Inter:wght@300;400;500;600;700&display=swap');
                
                body {{
                    font-family: 'Inter', sans-serif;
                    line-height: 1.6;
                    color: #ffffff;
                    margin: 0;
                    padding: 0;
                    background: #000000;
                }}
                
                .wrapper {{
                    background: #000000;
                    padding: 40px 20px;
                }}
                
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    background: #1a1a1a;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 16px;
                    overflow: hidden;
                    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
                }}
                
                .header {{
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    padding: 40px 30px;
                    text-align: center;
                }}
                
                .success-icon {{
                    font-size: 64px;
                    margin-bottom: 16px;
                }}
                
                .header-title {{
                    font-size: 28px;
                    font-weight: 700;
                    color: white;
                    margin: 0;
                }}
                
                .content {{
                    padding: 40px 30px;
                    background: #1a1a1a;
                }}
                
                .greeting {{
                    font-size: 20px;
                    font-weight: 600;
                    margin-bottom: 20px;
                    color: #ffffff;
                }}
                
                .message {{
                    color: #b3b3b3;
                    font-size: 16px;
                    line-height: 1.6;
                    margin-bottom: 30px;
                }}
                
                .button-container {{
                    text-align: center;
                    margin: 30px 0;
                }}
                
                .login-button {{
                    display: inline-block;
                    padding: 14px 32px;
                    background: linear-gradient(135deg, #E50914, #8B5CF6);
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 16px;
                    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3);
                }}
                
                .security-tips {{
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                }}
                
                .security-tips h3 {{
                    color: #ffffff;
                    font-size: 18px;
                    margin-bottom: 12px;
                }}
                
                .tip {{
                    color: #b3b3b3;
                    font-size: 14px;
                    margin: 8px 0;
                    padding-left: 20px;
                    position: relative;
                }}
                
                .tip:before {{
                    content: "✓";
                    position: absolute;
                    left: 0;
                    color: #10b981;
                }}
                
                .warning {{
                    background: rgba(239, 68, 68, 0.1);
                    border: 1px solid rgba(239, 68, 68, 0.2);
                    border-radius: 8px;
                    padding: 16px;
                    margin: 20px 0;
                    color: #f87171;
                    font-size: 14px;
                }}
                
                .footer {{
                    background: #121212;
                    padding: 30px;
                    text-align: center;
                    color: #666;
                    font-size: 14px;
                    border-top: 1px solid rgba(255, 255, 255, 0.1);
                }}
            </style>
        </head>
        <body>
            <div class="wrapper">
                <div class="container">
                    <div class="header">
                        <div class="success-icon">✅</div>
                        <h1 class="header-title">Password Changed Successfully!</h1>
                    </div>
                    <div class="content">
                        <h2 class="greeting">Hi{' ' + user_name if user_name else ''}! 👋</h2>
                        <p class="message">
                            Your CineBrain account password has been successfully changed. 
                            You can now log in with your new password.
                        </p>
                        
                        <div class="button-container">
                            <a href="{FRONTEND_URL}" class="login-button">Go to CineBrain</a>
                        </div>
                        
                        <div class="warning">
                            <strong>⚠️ Didn't make this change?</strong><br>
                            If you didn't change your password, your account may be compromised. 
                            Please contact our support team immediately and change your password.
                        </div>
                        
                        <div class="security-tips">
                            <h3>🔒 Security Tips:</h3>
                            <div class="tip">Use a unique password for CineBrain</div>
                            <div class="tip">Enable two-factor authentication when available</div>
                            <div class="tip">Never share your password with anyone</div>
                            <div class="tip">Use a password manager for better security</div>
                            <div class="tip">Regularly update your password</div>
                        </div>
                    </div>
                    <div class="footer">
                        <p>Thank you for using CineBrain!</p>
                        <p>&copy; 2025 CineBrain. All rights reserved.</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        mail.send(msg)
        return True
    except Exception as e:
        logger.error(f"Failed to send password changed email: {e}")
        return False

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    return True, "Valid password"

# Authentication routes
@auth_bp.route('/api/auth/forgot-password', methods=['POST', 'OPTIONS'])
def forgot_password():
    """Request password reset"""
    # Handle preflight request
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        # Validate email format
        if not email or not EMAIL_REGEX.match(email):
            return jsonify({'error': 'Please provide a valid email address'}), 400
        
        # Check if user exists
        user = User.query.filter_by(email=email).first()
        
        # Always return success to prevent email enumeration
        # But only send email if user exists
        if user:
            # Generate reset token
            token = generate_reset_token(email)
            
            # Send email
            email_sent = send_reset_email(email, token)
            
            if not email_sent:
                logger.error(f"Failed to send reset email to {email}")
                # Still return success to prevent enumeration
        
        return jsonify({
            'success': True,
            'message': 'If an account exists with this email, you will receive password reset instructions.'
        }), 200
        
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        return jsonify({'error': 'Failed to process password reset request'}), 500

@auth_bp.route('/api/auth/reset-password', methods=['POST', 'OPTIONS'])
def reset_password():
    """Reset password with token"""
    # Handle preflight request
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        token = data.get('token', '').strip()
        new_password = data.get('password', '')
        confirm_password = data.get('confirmPassword', '')  # Match frontend field name
        
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
        
        # Send confirmation email
        send_password_changed_email(email, user.username)
        
        # Generate a new auth token for immediate login
        auth_token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30),
            'iat': datetime.utcnow()
        }, app.secret_key, algorithm='HS256')
        
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
    """Verify if a reset token is valid"""
    # Handle preflight request
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        token = data.get('token', '').strip()
        
        if not token:
            return jsonify({'valid': False, 'error': 'No token provided'}), 400
        
        email = verify_reset_token(token)
        if email:
            # Also check if user still exists
            user = User.query.filter_by(email=email).first()
            if user:
                return jsonify({
                    'valid': True, 
                    'email': email,
                    'masked_email': email[:3] + '***' + email[email.index('@'):]  # Mask email for privacy
                }), 200
            else:
                return jsonify({'valid': False, 'error': 'User not found'}), 400
        else:
            return jsonify({'valid': False, 'error': 'Invalid or expired token'}), 400
            
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return jsonify({'valid': False, 'error': 'Failed to verify token'}), 500

# Add CORS headers to all auth responses
@auth_bp.after_request
def after_request(response):
    """Add CORS headers to responses"""
    origin = request.headers.get('Origin')
    if origin in [FRONTEND_URL, 'http://localhost:3000', 'http://localhost:5173']:
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Authentication decorator
def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Handle preflight
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
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

# Health check for auth service
@auth_bp.route('/api/auth/health', methods=['GET'])
def auth_health():
    """Check auth service health"""
    try:
        # Test database connection
        User.query.limit(1).first()
        
        # Test email configuration
        mail_configured = bool(app.config.get('MAIL_USERNAME'))
        
        return jsonify({
            'status': 'healthy',
            'service': 'authentication',
            'mail_configured': mail_configured,
            'frontend_url': FRONTEND_URL,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'authentication',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500