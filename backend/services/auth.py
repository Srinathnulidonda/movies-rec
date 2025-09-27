import random
import string
import secrets
from datetime import datetime, timedelta

@auth_bp.route('/api/auth/forgot-password-direct', methods=['POST', 'OPTIONS'])
def forgot_password_direct():
    """Direct password reset - returns code immediately"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        # Validate email
        if not email or not EMAIL_REGEX.match(email):
            return jsonify({'error': 'Please provide a valid email address'}), 400
        
        # Rate limiting
        if not check_rate_limit(f"forgot_password_direct:{email}", max_requests=5, window=600):
            return jsonify({
                'error': 'Too many reset attempts. Please try again in 10 minutes.'
            }), 429
        
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Generate 6-digit reset code
            reset_code = ''.join(random.choices(string.digits, k=6))
            
            # Also generate a backup token (for URL method)
            reset_token = secrets.token_urlsafe(32)
            
            # Store both in Redis with 15 minute expiry
            if redis_client:
                # Store code -> email mapping
                redis_client.setex(
                    f"reset_code:{reset_code}",
                    900,  # 15 minutes
                    json.dumps({
                        'email': email,
                        'user_id': user.id,
                        'created_at': datetime.utcnow().isoformat()
                    })
                )
                
                # Store token -> email mapping
                redis_client.setex(
                    f"reset_token_direct:{reset_token}",
                    900,  # 15 minutes
                    json.dumps({
                        'email': email,
                        'user_id': user.id,
                        'created_at': datetime.utcnow().isoformat()
                    })
                )
                
                logger.info(f"Reset code generated for {email}: {reset_code}")
            
            # Generate reset URL
            reset_url = f"{FRONTEND_URL}/reset-password.html?token={reset_token}"
            
            return jsonify({
                'success': True,
                'resetCode': reset_code,
                'resetToken': reset_token,
                'resetUrl': reset_url,
                'expiresIn': 15,  # minutes
                'message': 'Reset code generated successfully'
            }), 200
        
        # Don't reveal if user doesn't exist
        return jsonify({
            'success': False,
            'message': 'If an account exists with this email, a reset code has been generated.'
        }), 200
        
    except Exception as e:
        logger.error(f"Direct password reset error: {e}")
        return jsonify({'error': 'Failed to process reset request'}), 500

@auth_bp.route('/api/auth/reset-by-code', methods=['POST', 'OPTIONS'])
def reset_by_code():
    """Reset password using the 6-digit code"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        code = data.get('code', '').strip()
        new_password = data.get('newPassword', '')
        confirm_password = data.get('confirmPassword', '')
        
        # Validate inputs
        if not code or len(code) != 6:
            return jsonify({'error': 'Invalid reset code'}), 400
        
        if not new_password or not confirm_password:
            return jsonify({'error': 'Password and confirmation are required'}), 400
        
        if new_password != confirm_password:
            return jsonify({'error': 'Passwords do not match'}), 400
        
        # Validate password strength
        is_valid, message = validate_password(new_password)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Check code in Redis
        if redis_client:
            code_data = redis_client.get(f"reset_code:{code}")
            
            if not code_data:
                return jsonify({'error': 'Invalid or expired reset code'}), 400
            
            code_info = json.loads(code_data)
            email = code_info['email']
            user_id = code_info['user_id']
            
            # Get user and update password
            user = User.query.get(user_id)
            if user and user.email == email:
                user.password_hash = generate_password_hash(new_password)
                user.last_active = datetime.utcnow()
                db.session.commit()
                
                # Delete the used code
                redis_client.delete(f"reset_code:{code}")
                
                # Generate auth token for auto-login
                auth_token = jwt.encode({
                    'user_id': user.id,
                    'exp': datetime.utcnow() + timedelta(days=30),
                    'iat': datetime.utcnow()
                }, app.secret_key, algorithm='HS256')
                
                logger.info(f"Password reset successful for {email} using code")
                
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
            
        return jsonify({'error': 'Invalid or expired reset code'}), 400
        
    except Exception as e:
        logger.error(f"Reset by code error: {e}")
        return jsonify({'error': 'Failed to reset password'}), 500

@auth_bp.route('/api/auth/verify-reset-code', methods=['POST', 'OPTIONS'])
def verify_reset_code():
    """Verify if a reset code is valid"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        code = data.get('code', '').strip()
        
        if not code or len(code) != 6:
            return jsonify({'valid': False, 'error': 'Invalid code format'}), 400
        
        if redis_client:
            code_data = redis_client.get(f"reset_code:{code}")
            
            if code_data:
                code_info = json.loads(code_data)
                email = code_info['email']
                
                # Mask email for privacy
                masked_email = email[:3] + '***' + email[email.index('@'):]
                
                return jsonify({
                    'valid': True,
                    'maskedEmail': masked_email,
                    'message': 'Code is valid'
                }), 200
        
        return jsonify({'valid': False, 'error': 'Invalid or expired code'}), 400
        
    except Exception as e:
        logger.error(f"Code verification error: {e}")
        return jsonify({'valid': False, 'error': 'Verification failed'}), 500