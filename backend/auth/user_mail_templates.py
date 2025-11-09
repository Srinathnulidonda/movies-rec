# auth/mail_templates.py

from datetime import datetime
import os

FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')

def get_professional_template(content_type: str, **kwargs) -> tuple:
    """Get professional email templates"""
    base_css = """
    <style type="text/css">
        @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Inter:wght@300;400;500;600;700&display=swap');
        
        body {
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            font-size: 16px;
            line-height: 1.6;
            color: #1a1a1a;
            background: #f8f9fa;
        }
        
        .email-wrapper {
            width: 100%;
            background: #f8f9fa;
            padding: 32px 16px;
        }
        
        .email-container {
            max-width: 600px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(17,60,207,0.2);
            overflow: hidden;
            border: 1px solid #e8eaed;
        }
        
        .header {
            background: linear-gradient(135deg, #113CCF 0%, #1E4FE5 50%, #1E4FE5 100%);
            padding: 48px 32px;
            text-align: center;
        }
        
        .brand-logo {
            font-family: 'Bangers', cursive;
            font-size: 42px;
            font-weight: 400;
            letter-spacing: 1px;
            color: #ffffff;
            margin: 0;
        }
        
        .brand-tagline {
            font-size: 14px;
            color: rgba(255,255,255,0.95);
            margin: 8px 0 0;
        }
        
        .content {
            padding: 48px 32px;
            background: #ffffff;
        }
        
        .content-title {
            font-size: 32px;
            font-weight: 600;
            color: #1a1a1a;
            margin: 0 0 16px;
            text-align: center;
        }
        
        .content-body {
            font-size: 16px;
            line-height: 1.7;
            color: #1a1a1a;
            margin-bottom: 24px;
        }
        
        .btn {
            display: inline-block;
            font-size: 16px;
            font-weight: 600;
            text-decoration: none;
            text-align: center;
            padding: 16px 32px;
            border-radius: 50px;
            background: linear-gradient(135deg, #113CCF 0%, #1E4FE5 100%);
            color: #ffffff !important;
            min-width: 200px;
        }
        
        .btn-container {
            text-align: center;
            margin: 32px 0;
        }
        
        .alert {
            padding: 16px 24px;
            border-radius: 12px;
            margin: 24px 0;
            background: rgba(245,158,11,0.1);
            border-left: 4px solid #f59e0b;
            color: #d97706;
        }
        
        .footer {
            background: #f8f9fa;
            padding: 32px;
            text-align: center;
            border-top: 1px solid #e8eaed;
        }
        
        .footer-text {
            font-size: 12px;
            color: #999999;
            margin: 8px 0;
        }
    </style>
    """
    
    if content_type == 'password_reset':
        return _get_password_reset_template(base_css, **kwargs)
    elif content_type == 'password_changed':
        return _get_password_changed_template(base_css, **kwargs)
    else:
        return _get_generic_template(base_css, **kwargs)

def _get_password_reset_template(base_css: str, **kwargs) -> tuple:
    """Generate password reset email template"""
    reset_url = kwargs.get('reset_url', '')
    user_name = kwargs.get('user_name', 'there')
    user_email = kwargs.get('user_email', '')
    
    html = f"""
    <!DOCTYPE html>
    <html>
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
                    <div class="brand-logo">CineBrain</div>
                    <div class="brand-tagline">The Mind Behind Your Next Favorite</div>
                </div>
                
                <div class="content">
                    <h1 class="content-title">Reset your password</h1>
                    
                    <div class="content-body">
                        <p>Hi {user_name},</p>
                        <p>We received a request to reset your CineBrain account password. Click the button below to create a new password:</p>
                    </div>
                    
                    <div class="btn-container">
                        <a href="{reset_url}" class="btn">Reset Password</a>
                    </div>
                    
                    <div class="alert">
                        <strong>⏰ This link expires in 1 hour</strong><br>
                        For security reasons, this password reset link will expire soon.
                    </div>
                    
                    <div style="margin-top: 24px; padding: 16px; background: #f8f9fa; border-radius: 8px;">
                        <p style="margin: 0; font-size: 13px; color: #666;">
                            Can't click the button? Copy this link:<br>
                            <code style="word-break: break-all;">{reset_url}</code>
                        </p>
                    </div>
                </div>
                
                <div class="footer">
                    <p class="footer-text">
                        If you didn't request this, you can safely ignore this email.
                    </p>
                    <p class="footer-text">
                        © {datetime.now().year} CineBrain, Inc.
                    </p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text = f"""
Reset your password - CineBrain

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

def _get_password_changed_template(base_css: str, **kwargs) -> tuple:
    """Generate password changed confirmation template"""
    user_name = kwargs.get('user_name', 'there')
    user_email = kwargs.get('user_email', '')
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Password changed - CineBrain</title>
        {base_css}
    </head>
    <body>
        <div class="email-wrapper">
            <div class="email-container">
                <div class="header" style="background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);">
                    <div class="brand-logo">✅ Password Changed</div>
                    <div class="brand-tagline">Your account is now secured</div>
                </div>
                
                <div class="content">
                    <h1 class="content-title">Password successfully changed</h1>
                    
                    <div class="content-body">
                        <p>Hi {user_name},</p>
                        <p>Your CineBrain account password was successfully changed.</p>
                    </div>
                    
                    <div class="btn-container">
                        <a href="{FRONTEND_URL}/login" class="btn">Sign in to CineBrain</a>
                    </div>
                </div>
                
                <div class="footer">
                    <p class="footer-text">
                        If you didn't make this change, please contact support immediately.
                    </p>
                    <p class="footer-text">
                        © {datetime.now().year} CineBrain, Inc.
                    </p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text = f"""
Password Changed Successfully - CineBrain

Hi {user_name},

Your CineBrain account password was successfully changed.

If you didn't make this change, please contact support immediately.

© {datetime.now().year} CineBrain, Inc.
    """
    
    return html, text

def _get_generic_template(base_css: str, **kwargs) -> tuple:
    """Generate generic email template"""
    subject = kwargs.get('subject', 'CineBrain')
    content = kwargs.get('content', '')
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{subject}</title>
        {base_css}
    </head>
    <body>
        <div class="email-wrapper">
            <div class="email-container">
                <div class="header">
                    <div class="brand-logo">CineBrain</div>
                </div>
                <div class="content">
                    <div class="content-body">{content}</div>
                </div>
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