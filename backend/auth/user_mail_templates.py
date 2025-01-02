# auth/user_mail_templates.py

from datetime import datetime
import os

FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')

def get_professional_template(content_type: str, **kwargs) -> tuple:
    """Get professional email templates with enterprise-level design"""
    base_css = """
    <style type="text/css">
        @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* Reset and base styles */
        body, table, td, th, p, a {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            margin: 0;
            padding: 0;
            border: 0;
            font-size: 100%;
            vertical-align: baseline;
        }
        
        body {
            -webkit-text-size-adjust: 100%;
            -ms-text-size-adjust: 100%;
            margin: 0 !important;
            padding: 0 !important;
            background-color: #ffffff;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: #1f2937;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        table {
            border-collapse: collapse;
            mso-table-lspace: 0pt;
            mso-table-rspace: 0pt;
        }
        
        img {
            border: 0;
            outline: none;
            text-decoration: none;
            -ms-interpolation-mode: bicubic;
            max-width: 100%;
            height: auto;
            display: block;
        }
        
        /* Main container */
        .email-wrapper {
            width: 100% !important;
            background-color: #ffffff;
            padding: 20px 0;
            min-height: auto;
        }
        
        .email-container {
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1), 0 4px 15px rgba(0, 0, 0, 0.08);
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid #e5e7eb;
        }
        
        /* Header section */
        .header {
            background-color: #1A1D29;
            padding: 20px 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .brand-logo {
            font-family: 'Bangers', cursive;
            font-size: clamp(36px, 8vw, 48px);
            font-weight: 400;
            background: linear-gradient(135deg, #113CCF 0%, #1E4FE5 50%, #2563EB 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            text-decoration: none;
            letter-spacing: 2px;
            position: relative;
            z-index: 2;
            display: inline-block;
            line-height: 1.2;
            text-shadow: 0 2px 10px rgba(17, 60, 207, 0.3);
        }
        
        .brand-subtitle {
            font-size: 12px;
            color: #9CA3AF;
            font-weight: 400;
            margin: 4px 0 0 0;
            letter-spacing: 0.5px;
            position: relative;
            z-index: 2;
        }
        
        /* Content section */
        .content {
            padding: 30px 40px;
            background-color: #ffffff;
        }
        
        .content-header {
            text-align: center;
            margin-bottom: 24px;
        }
        
        .content-title {
            font-family: 'Bangers', cursive;
            font-size: clamp(24px, 6vw, 36px);
            font-weight: 400;
            background: linear-gradient(135deg, #113CCF 0%, #1E4FE5 50%, #2563EB 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 0 8px 0;
            letter-spacing: 2px;
            line-height: 1.2;
            text-shadow: 0 2px 10px rgba(17, 60, 207, 0.3);
        }
        
        .content-subtitle {
            font-size: 16px;
            color: #6b7280;
            font-weight: 400;
            margin: 0;
            line-height: 1.5;
        }
        
        .content-body {
            margin-bottom: 20px;
        }
        
        .greeting {
            font-size: 18px;
            font-weight: 600;
            color: #111827;
            margin: 0 0 20px 0;
            line-height: 1.4;
        }
        
        .message-text {
            font-size: 16px;
            color: #374151;
            line-height: 1.65;
            margin: 0 0 16px 0;
            font-weight: 400;
        }
        
        .message-text.bold {
            font-weight: 600;
            color: #111827;
        }
        
        .message-text.highlight {
            background: linear-gradient(135deg, rgba(17, 60, 207, 0.08) 0%, rgba(30, 79, 229, 0.06) 100%);
            padding: 16px 20px;
            border-radius: 12px;
            border-left: 4px solid #113CCF;
            margin: 20px 0;
        }
        
        /* Button section */
        .btn-container {
            text-align: center;
            margin: 30px 0;
        }
        
        .btn-primary {
            display: inline-block;
            background: linear-gradient(135deg, #113CCF 0%, #1E4FE5 100%);
            color: #ffffff !important;
            text-decoration: none;
            font-size: 16px;
            font-weight: 600;
            padding: 14px 30px;
            border-radius: 12px;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 12px rgba(17, 60, 207, 0.3), 0 2px 6px rgba(17, 60, 207, 0.2);
            transition: all 0.3s ease;
            border: 2px solid #113CCF;
            line-height: 1.4;
        }
        
        .btn-primary:hover {
            background: linear-gradient(135deg, #0A2A9F 0%, #113CCF 100%);
            box-shadow: 0 6px 20px rgba(17, 60, 207, 0.4), 0 3px 8px rgba(17, 60, 207, 0.3);
            transform: translateY(-1px);
        }
        
        /* Info box */
        .info-box {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            margin: 24px 0;
            border-left: 4px solid #113CCF;
        }
        
        .info-title {
            font-size: 13px;
            font-weight: 700;
            color: #374151;
            margin: 0 0 12px 0;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }
        
        .info-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 8px 0;
            font-size: 14px;
        }
        
        .info-label {
            color: #6b7280;
            font-weight: 500;
        }
        
        .info-value {
            color: #111827;
            font-weight: 600;
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        }
        
        /* Security notice */
        .security-notice {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.06) 100%);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 12px;
            padding: 16px;
            margin: 20px 0;
            border-left: 4px solid #10b981;
        }
        
        .security-notice.warning {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(220, 38, 38, 0.06) 100%);
            border-color: rgba(239, 68, 68, 0.2);
            border-left-color: #ef4444;
        }
        
        .security-icon {
            font-size: 18px;
            margin-bottom: 6px;
            display: block;
        }
        
        .security-text {
            font-size: 14px;
            line-height: 1.5;
            color: #374151;
            margin: 0;
            font-weight: 500;
        }
        
        /* Footer */
        .footer {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            padding: 20px 40px;
            border-top: 1px solid #e5e7eb;
        }
        
        .footer-content {
            text-align: center;
        }
        
        .footer-text {
            font-size: 12px;
            color: #6b7280;
            margin: 6px 0;
            line-height: 1.5;
            font-weight: 400;
        }
        
        .footer-brand {
            font-size: 13px;
            color: #374151;
            margin: 0 0 12px 0;
            font-weight: 600;
        }
        
        .footer-links {
            margin: 12px 0 0 0;
        }
        
        .footer-link {
            color: #113CCF;
            text-decoration: none;
            margin: 0 12px;
            font-size: 12px;
            font-weight: 500;
            transition: color 0.3s ease;
        }
        
        .footer-link:hover {
            color: #0A2A9F;
            text-decoration: underline;
        }
        
        .footer-link:last-child {
            margin-right: 0;
        }
        
        /* Mobile optimizations */
        @media only screen and (max-width: 600px) {
            .email-wrapper {
                padding: 15px 0;
            }
            
            .email-container {
                margin: 0 10px;
                border-radius: 12px;
            }
            
            .header {
                padding: 16px 20px;
            }
            
            .brand-logo {
                font-size: 32px;
                letter-spacing: 2px;
            }
            
            .brand-subtitle {
                font-size: 11px;
            }
            
            .content {
                padding: 24px 20px;
            }
            
            .content-title {
                font-size: 24px;
                letter-spacing: 1.5px;
            }
            
            .content-subtitle {
                font-size: 14px;
            }
            
            .btn-primary {
                display: block;
                width: 100%;
                box-sizing: border-box;
                text-align: center;
                padding: 16px 20px;
            }
            
            .footer {
                padding: 16px 20px;
            }
            
            .footer-link {
                display: block;
                margin: 6px 0;
            }
            
            .info-item {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .info-value {
                margin-top: 4px;
            }
        }
        
        /* Fallback for email clients that don't support background-clip */
        @supports not (-webkit-background-clip: text) {
            .brand-logo,
            .content-title {
                background: none !important;
                -webkit-background-clip: inherit !important;
                -webkit-text-fill-color: #113CCF !important;
                background-clip: inherit !important;
                color: #113CCF !important;
            }
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            .email-wrapper {
                background-color: #ffffff !important;
            }
            
            .email-container {
                background-color: #ffffff !important;
                border-color: #e5e7eb !important;
            }
            
            .content {
                background-color: #ffffff !important;
            }
        }
        
        /* High contrast mode */
        @media (prefers-contrast: high) {
            .btn-primary {
                border-width: 3px;
            }
            
            .info-box,
            .security-notice {
                border-width: 2px;
            }
            
            .brand-logo,
            .content-title {
                background: none !important;
                -webkit-background-clip: inherit !important;
                -webkit-text-fill-color: #113CCF !important;
                background-clip: inherit !important;
                color: #113CCF !important;
            }
        }
        
        /* Print styles */
        @media print {
            .email-wrapper {
                background: white !important;
                padding: 0 !important;
            }
            
            .email-container {
                box-shadow: none !important;
                border: 1px solid #000 !important;
            }
            
            .header {
                background: #1A1D29 !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }
            
            .btn-primary {
                background: #113CCF !important;
                border: 2px solid #113CCF !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }
            
            .brand-logo,
            .content-title {
                background: none !important;
                -webkit-background-clip: inherit !important;
                -webkit-text-fill-color: #113CCF !important;
                background-clip: inherit !important;
                color: #113CCF !important;
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
    """Generate enterprise-level password reset email template"""
    reset_url = kwargs.get('reset_url', '')
    user_name = kwargs.get('user_name', 'there')
    user_email = kwargs.get('user_email', '')
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en" xmlns="http://www.w3.org/1999/xhtml" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="x-apple-disable-message-reformatting">
        <meta name="format-detection" content="telephone=no,address=no,email=no,date=no,url=no">
        <meta name="color-scheme" content="light dark">
        <meta name="supported-color-schemes" content="light dark">
        <title>Reset your password - CineBrain</title>
        <!--[if mso]>
        <noscript>
            <xml>
                <o:OfficeDocumentSettings>
                    <o:AllowPNG/>
                    <o:PixelsPerInch>96</o:PixelsPerInch>
                </o:OfficeDocumentSettings>
            </xml>
        </noscript>
        <![endif]-->
        {base_css}
    </head>
    <body>
        <div role="article" aria-roledescription="email" aria-label="Password Reset Email" lang="en">
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header">
                        <h1 class="brand-logo">CineBrain</h1>
                        <p class="brand-subtitle">The Mind Behind Your Next Favorite</p>
                    </div>
                    
                    <div class="content">
                        <div class="content-header">
                            <h2 class="content-title">Reset Your Password</h2>
                            <p class="content-subtitle">Secure access to your CineBrain account</p>
                        </div>
                        
                        <div class="content-body">
                            <p class="greeting">Hello {user_name},</p>
                            
                            <p class="message-text">
                                We received a request to reset the password for your CineBrain account. To proceed with resetting your password, please click the secure button below:
                            </p>
                            
                            <div class="btn-container">
                                <a href="{reset_url}" class="btn-primary" role="button" aria-label="Reset your password">
                                    Reset My Password
                                </a>
                            </div>
                            
                            <div class="security-notice">
                                <span class="security-icon">🔒</span>
                                <p class="security-text">
                                    <strong>Security Notice:</strong> This password reset link will expire in 1 hour for your protection. If you didn't request this reset, please ignore this email or contact our support team.
                                </p>
                            </div>
                            
                            <p class="message-text">
                                For your security, this link can only be used once and will expire automatically. If you continue to have trouble accessing your account, our support team is here to help.
                            </p>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <div class="footer-content">
                            <p class="footer-brand">© {datetime.now().year} CineBrain</p>
                            <p class="footer-text">
                                This security email was sent to {user_email} because a password reset was requested for your CineBrain account.
                            </p>
                            <p class="footer-text">
                                CineBrain is committed to protecting your account security and privacy.
                            </p>
                            <div class="footer-links">
                                <a href="{FRONTEND_URL}" class="footer-link">Visit CineBrain</a>
                                <a href="{FRONTEND_URL}/privacy" class="footer-link">Privacy Policy</a>
                                <a href="{FRONTEND_URL}/terms" class="footer-link">Terms of Service</a>
                                <a href="{FRONTEND_URL}/support" class="footer-link">Support</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text = f"""
CineBrain - Reset Your Password

Hello {user_name},

We received a request to reset the password for your CineBrain account.

To reset your password, visit the following secure link:
{reset_url}

SECURITY NOTICE: This password reset link will expire in 1 hour for your protection.

If you didn't request this password reset, please ignore this email or contact our support team if you have concerns.

For your security, this link can only be used once and will expire automatically.

---

© {datetime.now().year} CineBrain
The Mind Behind Your Next Favorite

This security email was sent to {user_email} because a password reset was requested for your CineBrain account.

Visit CineBrain: {FRONTEND_URL}
Privacy Policy: {FRONTEND_URL}/privacy
Terms of Service: {FRONTEND_URL}/terms
Support: {FRONTEND_URL}/support

CineBrain is committed to protecting your account security and privacy.
    """
    
    return html, text

def _get_password_changed_template(base_css: str, **kwargs) -> tuple:
    """Generate enterprise-level password changed confirmation template"""
    user_name = kwargs.get('user_name', 'there')
    user_email = kwargs.get('user_email', '')
    ip_address = kwargs.get('ip_address', 'Unknown')
    location = kwargs.get('location', 'Unknown location')
    device = kwargs.get('device', 'Unknown device')
    
    # Format timestamp
    timestamp = datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en" xmlns="http://www.w3.org/1999/xhtml" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="x-apple-disable-message-reformatting">
        <meta name="format-detection" content="telephone=no,address=no,email=no,date=no,url=no">
        <meta name="color-scheme" content="light dark">
        <meta name="supported-color-schemes" content="light dark">
        <title>Password changed successfully - CineBrain</title>
        {base_css}
    </head>
    <body>
        <div role="article" aria-roledescription="email" aria-label="Password Changed Confirmation Email" lang="en">
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header">
                        <h1 class="brand-logo">CineBrain</h1>
                        <p class="brand-subtitle">The Mind Behind Your Next Favorite</p>
                    </div>
                    
                    <div class="content">
                        <div class="content-header">
                            <h2 class="content-title" style="background: linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Password Updated Successfully</h2>
                            <p class="content-subtitle">Your CineBrain account is now more secure</p>
                        </div>
                        
                        <div class="content-body">
                            <p class="greeting">Hello {user_name},</p>
                            
                            <p class="message-text">
                                Great news! Your CineBrain account password has been successfully updated. Your account security has been enhanced and you can continue enjoying our personalized entertainment recommendations.
                            </p>
                            
                            <div class="info-box">
                                <p class="info-title">Security Details</p>
                                <div class="info-item">
                                    <span class="info-label">Changed on:</span>
                                    <span class="info-value">{timestamp}</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-label">Device:</span>
                                    <span class="info-value">{device}</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-label">Location:</span>
                                    <span class="info-value">{location}</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-label">IP Address:</span>
                                    <span class="info-value">{ip_address}</span>
                                </div>
                            </div>
                            
                            <div class="security-notice">
                                <span class="security-icon">✅</span>
                                <p class="security-text">
                                    <strong>Account Secured:</strong> If you made this change, no further action is needed. Your account is now protected with your new password.
                                </p>
                            </div>
                            
                            <div class="security-notice warning">
                                <span class="security-icon">⚠️</span>
                                <p class="security-text">
                                    <strong>Didn't make this change?</strong> If you didn't update your password, please contact our support team immediately and reset your password to secure your account.
                                </p>
                            </div>
                            
                            <div class="btn-container">
                                <a href="{FRONTEND_URL}/login" class="btn-primary" role="button" aria-label="Sign in to CineBrain">
                                    Sign In to CineBrain
                                </a>
                            </div>
                            
                            <p class="message-text">
                                Thank you for keeping your CineBrain account secure. We're committed to protecting your privacy and providing you with the best entertainment discovery experience.
                            </p>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <div class="footer-content">
                            <p class="footer-brand">© {datetime.now().year} CineBrain</p>
                            <p class="footer-text">
                                This security notification was sent to {user_email} to confirm your password change.
                            </p>
                            <p class="footer-text">
                                CineBrain is committed to protecting your account security and privacy.
                            </p>
                            <div class="footer-links">
                                <a href="{FRONTEND_URL}" class="footer-link">Visit CineBrain</a>
                                <a href="{FRONTEND_URL}/privacy" class="footer-link">Privacy Policy</a>
                                <a href="{FRONTEND_URL}/terms" class="footer-link">Terms of Service</a>
                                <a href="{FRONTEND_URL}/support" class="footer-link">Support</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text = f"""
CineBrain - Password Updated Successfully

Hello {user_name},

Great news! Your CineBrain account password has been successfully updated.

SECURITY DETAILS:
- Changed on: {timestamp}
- Device: {device}  
- Location: {location}
- IP Address: {ip_address}

✅ ACCOUNT SECURED: If you made this change, no further action is needed. Your account is now protected with your new password.

⚠️ DIDN'T MAKE THIS CHANGE? If you didn't update your password, please contact our support team immediately and reset your password to secure your account.

Sign in to CineBrain: {FRONTEND_URL}/login

Thank you for keeping your CineBrain account secure. We're committed to protecting your privacy and providing you with the best entertainment discovery experience.

---

© {datetime.now().year} CineBrain
The Mind Behind Your Next Favorite

This security notification was sent to {user_email} to confirm your password change.

Visit CineBrain: {FRONTEND_URL}
Privacy Policy: {FRONTEND_URL}/privacy
Terms of Service: {FRONTEND_URL}/terms
Support: {FRONTEND_URL}/support

CineBrain is committed to protecting your account security and privacy.
    """
    
    return html, text

def _get_generic_template(base_css: str, **kwargs) -> tuple:
    """Generate enterprise-level generic email template"""
    subject = kwargs.get('subject', 'CineBrain')
    content = kwargs.get('content', '')
    user_name = kwargs.get('user_name', 'there')
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en" xmlns="http://www.w3.org/1999/xhtml" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="x-apple-disable-message-reformatting">
        <meta name="format-detection" content="telephone=no,address=no,email=no,date=no,url=no">
        <meta name="color-scheme" content="light dark">
        <meta name="supported-color-schemes" content="light dark">
        <title>{subject}</title>
        {base_css}
    </head>
    <body>
        <div role="article" aria-roledescription="email" aria-label="CineBrain Email" lang="en">
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header">
                        <h1 class="brand-logo">CineBrain</h1>
                        <p class="brand-subtitle">The Mind Behind Your Next Favorite</p>
                    </div>
                    
                    <div class="content">
                        <div class="content-header">
                            <h2 class="content-title">{subject}</h2>
                        </div>
                        
                        <div class="content-body">
                            <p class="greeting">Hello {user_name},</p>
                            <div class="message-text">{content}</div>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <div class="footer-content">
                            <p class="footer-brand">© {datetime.now().year} CineBrain</p>
                            <p class="footer-text">
                                The Mind Behind Your Next Favorite - Discover personalized entertainment recommendations.
                            </p>
                            <div class="footer-links">
                                <a href="{FRONTEND_URL}" class="footer-link">Visit CineBrain</a>
                                <a href="{FRONTEND_URL}/privacy" class="footer-link">Privacy Policy</a>
                                <a href="{FRONTEND_URL}/terms" class="footer-link">Terms of Service</a>
                                <a href="{FRONTEND_URL}/support" class="footer-link">Support</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    text = f"""
CineBrain - {subject}

Hello {user_name},

{content}

---

© {datetime.now().year} CineBrain
The Mind Behind Your Next Favorite

Visit CineBrain: {FRONTEND_URL}
Privacy Policy: {FRONTEND_URL}/privacy
Terms of Service: {FRONTEND_URL}/terms
Support: {FRONTEND_URL}/support

Discover personalized entertainment recommendations at CineBrain.
    """
    
    return html, text