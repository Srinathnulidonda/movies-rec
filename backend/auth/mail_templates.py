# auth/mail_templates.py

from datetime import datetime
import os

FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')

def get_professional_template(content_type: str, **kwargs) -> tuple:
    """Get professional email templates with anti-spam optimizations"""
    base_css = """
    <style type="text/css">
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        
        body {
            margin: 0;
            padding: 0;
            font-family: 'Inter', Arial, sans-serif;
            font-size: 16px;
            line-height: 1.6;
            color: #333333;
            background-color: #f4f4f4;
        }
        
        .email-wrapper {
            width: 100%;
            background-color: #f4f4f4;
            padding: 20px 0;
        }
        
        .email-container {
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .header {
            background-color: #113CCF;
            padding: 40px 20px;
            text-align: center;
        }
        
        .brand-logo {
            font-size: 32px;
            font-weight: 600;
            color: #ffffff;
            margin: 0;
            text-decoration: none;
        }
        
        .content {
            padding: 40px 30px;
            background-color: #ffffff;
        }
        
        .content-title {
            font-size: 24px;
            font-weight: 600;
            color: #333333;
            margin: 0 0 20px;
        }
        
        .content-body {
            font-size: 16px;
            line-height: 1.6;
            color: #555555;
            margin-bottom: 30px;
        }
        
        .btn {
            display: inline-block;
            font-size: 16px;
            font-weight: 500;
            text-decoration: none;
            text-align: center;
            padding: 14px 30px;
            border-radius: 5px;
            background-color: #113CCF;
            color: #ffffff !important;
        }
        
        .btn-container {
            text-align: center;
            margin: 30px 0;
        }
        
        .footer {
            background-color: #f8f9fa;
            padding: 30px;
            text-align: center;
            border-top: 1px solid #e0e0e0;
        }
        
        .footer-text {
            font-size: 13px;
            color: #666666;
            margin: 5px 0;
        }
        
        .footer-links {
            margin-top: 15px;
        }
        
        .footer-links a {
            color: #113CCF;
            text-decoration: none;
            margin: 0 10px;
            font-size: 13px;
        }
        
        @media only screen and (max-width: 600px) {
            .content {
                padding: 30px 20px;
            }
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
                <div class="header">
                    <h1 class="brand-logo">CineBrain</h1>
                </div>
                
                <div class="content">
                    <h2 class="content-title">Reset your password</h2>
                    
                    <div class="content-body">
                        <p>Hi {user_name},</p>
                        <p>You recently requested to reset your password for your CineBrain account. Click the button below to reset it:</p>
                    </div>
                    
                    <div class="btn-container">
                        <a href="{reset_url}" class="btn" style="color: #ffffff;">Reset Password</a>
                    </div>
                    
                    <div class="content-body">
                        <p><strong>This password reset link will expire in 1 hour.</strong></p>
                        <p>If you did not request a password reset, please ignore this email or contact support if you have concerns.</p>
                        <p style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #e0e0e0; font-size: 14px; color: #666666;">
                            If the button above doesn't work, copy and paste this link into your browser:<br>
                            <span style="color: #113CCF; word-break: break-all;">{reset_url}</span>
                        </p>
                    </div>
                </div>
                
                <div class="footer">
                    <p class="footer-text">
                        This email was sent to {user_email} because a password reset was requested for this account.
                    </p>
                    <p class="footer-text">
                        © {datetime.now().year} CineBrain. All rights reserved.
                    </p>
                    <div class="footer-links">
                        <a href="{FRONTEND_URL}">Visit CineBrain</a>
                        <a href="{FRONTEND_URL}/privacy">Privacy Policy</a>
                        <a href="{FRONTEND_URL}/terms">Terms of Service</a>
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

This password reset link will expire in 1 hour.

If you did not request a password reset, please ignore this email or contact support if you have concerns.

Best regards,
The CineBrain Team

© {datetime.now().year} CineBrain. All rights reserved.

This email was sent to {user_email} because a password reset was requested for this account.
    """
    
    return html, text

def _get_password_changed_template(base_css: str, **kwargs) -> tuple:
    """Generate password changed confirmation template"""
    user_name = kwargs.get('user_name', 'there')
    user_email = kwargs.get('user_email', '')
    ip_address = kwargs.get('ip_address', 'Unknown')
    device = kwargs.get('device', 'Unknown device')
    
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
                <div class="header" style="background-color: #22c55e;">
                    <h1 class="brand-logo">CineBrain</h1>
                </div>
                
                <div class="content">
                    <h2 class="content-title">Password successfully changed</h2>
                    
                    <div class="content-body">
                        <p>Hi {user_name},</p>
                        <p>Your CineBrain account password was successfully changed.</p>
                        
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                            <p style="margin: 0; font-size: 14px;"><strong>Details:</strong></p>
                            <p style="margin: 5px 0; font-size: 14px;">Time: {datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')}</p>
                            <p style="margin: 5px 0; font-size: 14px;">Device: {device}</p>
                            <p style="margin: 5px 0; font-size: 14px;">IP Address: {ip_address}</p>
                        </div>
                        
                        <p>If you made this change, no further action is needed.</p>
                        <p><strong>If you did not make this change, please reset your password immediately and contact our support team.</strong></p>
                    </div>
                    
                    <div class="btn-container">
                        <a href="{FRONTEND_URL}/login" class="btn" style="color: #ffffff;">Sign in to CineBrain</a>
                    </div>
                </div>
                
                <div class="footer">
                    <p class="footer-text">
                        This is a security notification for {user_email}
                    </p>
                    <p class="footer-text">
                        © {datetime.now().year} CineBrain. All rights reserved.
                    </p>
                    <div class="footer-links">
                        <a href="{FRONTEND_URL}">Visit CineBrain</a>
                        <a href="{FRONTEND_URL}/support">Get Support</a>
                        <a href="{FRONTEND_URL}/security">Security Info</a>
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

Your CineBrain account password was successfully changed.

Details:
- Time: {datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')}
- Device: {device}
- IP Address: {ip_address}

If you made this change, no further action is needed.

If you did not make this change, please reset your password immediately and contact our support team.

Sign in to CineBrain: {FRONTEND_URL}/login

Best regards,
The CineBrain Team

© {datetime.now().year} CineBrain. All rights reserved.

This is a security notification for {user_email}
    """
    
    return html, text

def _get_generic_template(base_css: str, **kwargs) -> tuple:
    """Generate generic email template"""
    subject = kwargs.get('subject', 'CineBrain')
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
                <div class="header">
                    <h1 class="brand-logo">CineBrain</h1>
                </div>
                <div class="content">
                    <div class="content-body">{content}</div>
                </div>
                <div class="footer">
                    <p class="footer-text">© {datetime.now().year} CineBrain. All rights reserved.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text = f"{subject}\n\n{content}\n\n© {datetime.now().year} CineBrain. All rights reserved."
    
    return html, text