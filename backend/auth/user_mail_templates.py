# auth/user_mail_templates.py

from datetime import datetime
import os

FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')

def get_professional_template(content_type: str, **kwargs) -> tuple:
    """Get professional email templates matching CineBrain frontend branding"""
    
    # CineBrain Brand Colors - Matching frontend auth pages
    cinebrain_colors = {
        'primary': '#113CCF',
        'primary_light': '#1E4FE5',
        'primary_dark': '#0A2A9F',
        'accent': '#1E4FE5',
        'dark': '#0E0E0E',
        'darker': '#000000',
        'gray_dark': '#1A1D29',
        'gray_medium': '#2A2D3A',
        'text_secondary': '#9CA3AF',
        'success': '#10B981',
        'success_dark': '#059669',
        'error': '#EF4444',
        'error_dark': '#DC2626',
        'warning': '#F59E0B',
        'gold': '#FFD700'
    }
    
    # Base CSS matching CineBrain frontend design
    base_css = f"""
    <style type="text/css">
        @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', Arial, sans-serif;
            font-size: 16px;
            line-height: 1.6;
            color: #333333;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
        }}
        
        .email-wrapper {{
            width: 100%;
            background: linear-gradient(135deg,
                {cinebrain_colors['primary']} 0%,
                {cinebrain_colors['primary_light']} 50%,
                {cinebrain_colors['accent']} 100%);
            padding: 40px 20px;
            min-height: 100vh;
        }}
        
        .email-container {{
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
        }}
        
        .brand-header {{
            background: linear-gradient(135deg,
                {cinebrain_colors['darker']} 0%,
                {cinebrain_colors['dark']} 50%,
                {cinebrain_colors['gray_dark']} 100%);
            padding: 40px 30px;
            text-align: center;
            position: relative;
        }}
        
        .brand-header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg,
                {cinebrain_colors['primary']} 0%,
                {cinebrain_colors['primary_light']} 50%,
                {cinebrain_colors['accent']} 100%);
            opacity: 0.05;
        }}
        
        .brand-name {{
            font-family: 'Bangers', cursive;
            font-size: 48px;
            background: linear-gradient(135deg,
                {cinebrain_colors['primary']} 0%,
                {cinebrain_colors['primary_light']} 50%,
                {cinebrain_colors['accent']} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: 2px;
            margin: 0 0 8px 0;
            position: relative;
            z-index: 1;
            text-shadow: 0 2px 10px rgba(17, 60, 207, 0.3);
        }}
        
        .brand-tagline {{
            font-size: 14px;
            color: {cinebrain_colors['text_secondary']};
            font-weight: 400;
            letter-spacing: 0.5px;
            position: relative;
            z-index: 1;
            margin: 0;
        }}
        
        .content-section {{
            padding: 40px 30px;
            background-color: #ffffff;
        }}
        
        .content-title {{
            font-size: 28px;
            font-weight: 600;
            color: #333333;
            margin: 0 0 20px;
            text-align: center;
        }}
        
        .content-description {{
            font-size: 16px;
            color: #555555;
            margin-bottom: 30px;
            text-align: center;
            line-height: 1.6;
        }}
        
        .content-body {{
            font-size: 16px;
            line-height: 1.6;
            color: #555555;
            margin-bottom: 30px;
        }}
        
        .cta-button {{
            display: inline-block;
            font-size: 16px;
            font-weight: 600;
            text-decoration: none;
            text-align: center;
            padding: 14px 30px;
            border-radius: 12px;
            background: linear-gradient(135deg,
                {cinebrain_colors['primary']} 0%,
                {cinebrain_colors['primary_light']} 100%);
            color: #ffffff !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(17, 60, 207, 0.3);
        }}
        
        .cta-button:hover {{
            background: linear-gradient(135deg,
                {cinebrain_colors['primary_light']} 0%,
                {cinebrain_colors['accent']} 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(17, 60, 207, 0.4);
            color: #ffffff !important;
            text-decoration: none;
        }}
        
        .cta-container {{
            text-align: center;
            margin: 30px 0;
        }}
        
        .info-card {{
            background: linear-gradient(135deg,
                rgba(17, 60, 207, 0.05) 0%,
                rgba(30, 79, 229, 0.05) 100%);
            border: 1px solid rgba(17, 60, 207, 0.1);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .info-card h3 {{
            font-size: 16px;
            font-weight: 600;
            color: {cinebrain_colors['primary']};
            margin: 0 0 8px 0;
        }}
        
        .info-card p {{
            font-size: 14px;
            color: #666666;
            margin: 0;
            line-height: 1.5;
        }}
        
        .security-notice {{
            background: linear-gradient(135deg,
                rgba(16, 185, 129, 0.05) 0%,
                rgba(5, 150, 105, 0.05) 100%);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 12px;
            padding: 16px 20px;
            margin: 20px 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .security-icon {{
            width: 20px;
            height: 20px;
            color: {cinebrain_colors['success']};
        }}
        
        .security-text {{
            font-size: 14px;
            color: #666666;
            margin: 0;
        }}
        
        .footer-section {{
            background: linear-gradient(135deg,
                {cinebrain_colors['gray_dark']} 0%,
                {cinebrain_colors['gray_medium']} 100%);
            padding: 30px;
            text-align: center;
            border-top: 1px solid rgba(17, 60, 207, 0.1);
        }}
        
        .footer-text {{
            font-size: 13px;
            color: {cinebrain_colors['text_secondary']};
            margin: 5px 0;
            line-height: 1.4;
        }}
        
        .footer-links {{
            margin-top: 20px;
            text-align: center;
        }}
        
        .footer-link {{
            color: {cinebrain_colors['primary']};
            text-decoration: none;
            font-size: 13px;
            font-weight: 500;
            margin: 0 15px;
            transition: color 0.3s ease;
        }}
        
        .footer-link:hover {{
            color: {cinebrain_colors['primary_light']};
            text-decoration: underline;
        }}
        
        .fallback-url {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
            word-break: break-all;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            color: #666666;
        }}
        
        .fallback-label {{
            font-size: 12px;
            font-weight: 600;
            color: #666666;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .success-indicator {{
            background: linear-gradient(135deg,
                {cinebrain_colors['success']} 0%,
                {cinebrain_colors['success_dark']} 100%);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            text-align: center;
            margin: 20px 0;
            font-weight: 500;
        }}
        
        .warning-indicator {{
            background: linear-gradient(135deg,
                {cinebrain_colors['warning']} 0%,
                #d97706 100%);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            text-align: center;
            margin: 20px 0;
            font-weight: 500;
        }}
        
        .divider {{
            height: 1px;
            background: linear-gradient(90deg,
                transparent 0%,
                rgba(17, 60, 207, 0.2) 50%,
                transparent 100%);
            margin: 30px 0;
        }}
        
        @media only screen and (max-width: 600px) {{
            .email-wrapper {{
                padding: 20px 10px;
            }}
            
            .content-section {{
                padding: 30px 20px;
            }}
            
            .brand-name {{
                font-size: 36px;
            }}
            
            .content-title {{
                font-size: 24px;
            }}
            
            .cta-button {{
                display: block;
                margin: 0 auto;
                max-width: 280px;
            }}
            
            .footer-section {{
                padding: 25px 20px;
            }}
        }}
        
        @media (prefers-color-scheme: dark) {{
            .email-container {{
                background-color: #ffffff;
            }}
        }}
    </style>
    """
    
    if content_type == 'password_reset':
        return _get_password_reset_template(base_css, cinebrain_colors, **kwargs)
    elif content_type == 'password_changed':
        return _get_password_changed_template(base_css, cinebrain_colors, **kwargs)
    else:
        return _get_generic_template(base_css, cinebrain_colors, **kwargs)

def _get_password_reset_template(base_css: str, colors: dict, **kwargs) -> tuple:
    """Generate password reset email template with CineBrain branding"""
    reset_url = kwargs.get('reset_url', '')
    user_name = kwargs.get('user_name', 'there')
    user_email = kwargs.get('user_email', '')
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <title>Reset your CineBrain password</title>
        {base_css}
    </head>
    <body>
        <div class="email-wrapper">
            <div class="email-container">
                <div class="brand-header">
                    <h1 class="brand-name">CineBrain</h1>
                    <p class="brand-tagline">The Mind Behind Your Next Favorite</p>
                </div>
                
                <div class="content-section">
                    <h2 class="content-title">Reset Your Password</h2>
                    <p class="content-description">
                        Hi {user_name}, you recently requested to reset your password for your CineBrain account.
                    </p>
                    
                    <div class="content-body">
                        <p>Click the button below to create a new password for your CineBrain account:</p>
                    </div>
                    
                    <div class="cta-container">
                        <a href="{reset_url}" class="cta-button">Reset Password</a>
                    </div>
                    
                    <div class="info-card">
                        <h3>⏰ Important Security Notice</h3>
                        <p>This password reset link will expire in 1 hour for your security.</p>
                    </div>
                    
                    <div class="security-notice">
                        <svg class="security-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
                        </svg>
                        <p class="security-text">
                            If you didn't request this password reset, please ignore this email or contact our support team.
                        </p>
                    </div>
                    
                    <div class="divider"></div>
                    
                    <div class="fallback-label">If the button doesn't work, copy this link:</div>
                    <div class="fallback-url">{reset_url}</div>
                </div>
                
                <div class="footer-section">
                    <p class="footer-text">
                        This email was sent to <strong>{user_email}</strong> because a password reset was requested for this CineBrain account.
                    </p>
                    <p class="footer-text">
                        © {datetime.now().year} CineBrain. All rights reserved.
                    </p>
                    <div class="footer-links">
                        <a href="{FRONTEND_URL}" class="footer-link">Visit CineBrain</a>
                        <a href="{FRONTEND_URL}/privacy" class="footer-link">Privacy Policy</a>
                        <a href="{FRONTEND_URL}/terms" class="footer-link">Terms of Service</a>
                        <a href="{FRONTEND_URL}/support" class="footer-link">Get Support</a>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text = f"""
Reset your CineBrain password

Hi {user_name},

You recently requested to reset your password for your CineBrain account.

To reset your password, visit the following link:
{reset_url}

⏰ This password reset link will expire in 1 hour for your security.

🔒 If you didn't request this password reset, please ignore this email or contact our support team.

Best regards,
The CineBrain Team
The Mind Behind Your Next Favorite

© {datetime.now().year} CineBrain. All rights reserved.

This email was sent to {user_email} because a password reset was requested for this CineBrain account.

Visit CineBrain: {FRONTEND_URL}
Get Support: {FRONTEND_URL}/support
    """
    
    return html, text

def _get_password_changed_template(base_css: str, colors: dict, **kwargs) -> tuple:
    """Generate password changed confirmation template with CineBrain branding"""
    user_name = kwargs.get('user_name', 'there')
    user_email = kwargs.get('user_email', '')
    ip_address = kwargs.get('ip_address', 'Unknown')
    device = kwargs.get('device', 'Unknown device')
    location = kwargs.get('location', 'Unknown location')
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <title>Your CineBrain password was changed</title>
        {base_css}
    </head>
    <body>
        <div class="email-wrapper">
            <div class="email-container">
                <div class="brand-header">
                    <h1 class="brand-name">CineBrain</h1>
                    <p class="brand-tagline">The Mind Behind Your Next Favorite</p>
                </div>
                
                <div class="content-section">
                    <h2 class="content-title">Password Successfully Changed</h2>
                    <p class="content-description">
                        Hi {user_name}, your CineBrain account password was successfully updated.
                    </p>
                    
                    <div class="success-indicator">
                        ✅ Your CineBrain account is secure with your new password
                    </div>
                    
                    <div class="info-card">
                        <h3>🔐 Change Details</h3>
                        <p><strong>Time:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')}</p>
                        <p><strong>Device:</strong> {device}</p>
                        <p><strong>IP Address:</strong> {ip_address}</p>
                        <p><strong>Location:</strong> {location}</p>
                    </div>
                    
                    <div class="content-body">
                        <p>If you made this change, no further action is needed. You can continue enjoying CineBrain with your new password.</p>
                    </div>
                    
                    <div class="warning-indicator">
                        ⚠️ If you didn't make this change, please reset your password immediately and contact our support team.
                    </div>
                    
                    <div class="cta-container">
                        <a href="{FRONTEND_URL}/auth/login.html" class="cta-button">Sign in to CineBrain</a>
                    </div>
                    
                    <div class="security-notice">
                        <svg class="security-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path>
                        </svg>
                        <p class="security-text">
                            Your CineBrain account security is our top priority. We use industry-standard encryption.
                        </p>
                    </div>
                </div>
                
                <div class="footer-section">
                    <p class="footer-text">
                        This is a security notification for <strong>{user_email}</strong>
                    </p>
                    <p class="footer-text">
                        © {datetime.now().year} CineBrain. All rights reserved.
                    </p>
                    <div class="footer-links">
                        <a href="{FRONTEND_URL}" class="footer-link">Visit CineBrain</a>
                        <a href="{FRONTEND_URL}/support" class="footer-link">Get Support</a>
                        <a href="{FRONTEND_URL}/security" class="footer-link">Security Info</a>
                        <a href="{FRONTEND_URL}/auth/login.html" class="footer-link">Sign In</a>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text = f"""
Your CineBrain password was changed

Hi {user_name},

Your CineBrain account password was successfully updated.

✅ Your CineBrain account is secure with your new password

🔐 Change Details:
- Time: {datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')}
- Device: {device}
- IP Address: {ip_address}
- Location: {location}

If you made this change, no further action is needed. You can continue enjoying CineBrain with your new password.

⚠️ If you didn't make this change, please reset your password immediately and contact our support team.

Sign in to CineBrain: {FRONTEND_URL}/auth/login.html
Get Support: {FRONTEND_URL}/support

Best regards,
The CineBrain Team
The Mind Behind Your Next Favorite

© {datetime.now().year} CineBrain. All rights reserved.

This is a security notification for {user_email}
    """
    
    return html, text

def _get_generic_template(base_css: str, colors: dict, **kwargs) -> tuple:
    """Generate generic email template with CineBrain branding"""
    subject = kwargs.get('subject', 'CineBrain Notification')
    content = kwargs.get('content', '')
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{subject}</title>
        {base_css}
    </head>
    <body>
        <div class="email-wrapper">
            <div class="email-container">
                <div class="brand-header">
                    <h1 class="brand-name">CineBrain</h1>
                    <p class="brand-tagline">The Mind Behind Your Next Favorite</p>
                </div>
                
                <div class="content-section">
                    <div class="content-body">{content}</div>
                    
                    <div class="cta-container">
                        <a href="{FRONTEND_URL}" class="cta-button">Visit CineBrain</a>
                    </div>
                </div>
                
                <div class="footer-section">
                    <p class="footer-text">© {datetime.now().year} CineBrain. All rights reserved.</p>
                    <div class="footer-links">
                        <a href="{FRONTEND_URL}" class="footer-link">Visit CineBrain</a>
                        <a href="{FRONTEND_URL}/support" class="footer-link">Support</a>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text = f"{subject}\n\n{content}\n\nBest regards,\nThe CineBrain Team\nThe Mind Behind Your Next Favorite\n\n© {datetime.now().year} CineBrain. All rights reserved."
    
    return html, text