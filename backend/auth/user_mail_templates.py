# auth/user_mail_templates.py

from datetime import datetime
import os

FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')

def get_professional_template(content_type: str, **kwargs) -> tuple:
    """Get professional email templates with enterprise-level design"""
    base_css = """
    <style type="text/css">
        @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* CSS Variables matching login page */
        :root {
            --cinebrain-primary: #113CCF;
            --cinebrain-primary-light: #1E4FE5;
            --cinebrain-accent: #1E4FE5;
            --text-secondary: #9CA3AF;
        }
        
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
        
        /* Main container - Fully responsive */
        .email-wrapper {
            width: 100% !important;
            background-color: #ffffff;
            padding: clamp(15px, 4vw, 20px) 0;
            min-height: auto;
        }
        
        .email-container {
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1), 0 4px 15px rgba(0, 0, 0, 0.08);
            border-radius: clamp(12px, 3vw, 16px);
            overflow: hidden;
            border: 1px solid #e5e7eb;
            width: calc(100% - clamp(20px, 5vw, 40px));
        }
        
        /* Header section - Mobile responsive */
        .header {
            background-color: #ffffff;
            padding: clamp(16px, 4vw, 20px) clamp(20px, 6vw, 40px);
            text-align: center;
            position: relative;
            overflow: hidden;
            border-bottom: 1px solid #f1f5f9;
        }
        
        .brand-logo {
            max-width: clamp(140px, 35vw, 200px);
            height: auto;
            margin: 0 auto;
            display: block;
        }
        
        .brand-subtitle {
            font-size: clamp(10px, 2.5vw, 14px);
            color: var(--text-secondary);
            font-weight: 400;
            margin: clamp(4px, 2vw, 8px) 0 0 0;
            letter-spacing: clamp(0.3px, 0.1vw, 0.5px);
            position: relative;
            z-index: 2;
            line-height: 1.4;
        }
        
        /* Content section - Fully responsive */
        .content {
            padding: clamp(20px, 6vw, 30px) clamp(20px, 6vw, 40px);
            background-color: #ffffff;
        }
        
        .content-header {
            text-align: center;
            margin-bottom: clamp(20px, 5vw, 24px);
        }
        
        .content-title {
            font-family: 'Inter', sans-serif;
            font-size: clamp(20px, 5vw, 28px);
            font-weight: 600;
            color: #111827;
            margin: 0 0 clamp(6px, 2vw, 8px) 0;
            letter-spacing: clamp(0.3px, 0.1vw, 0.5px);
            line-height: 1.3;
        }
        
        .content-subtitle {
            font-size: clamp(14px, 3.5vw, 16px);
            color: #6b7280;
            font-weight: 400;
            margin: 0;
            line-height: 1.5;
        }
        
        .content-body {
            margin-bottom: clamp(16px, 4vw, 20px);
        }
        
        .greeting {
            font-size: clamp(16px, 4vw, 18px);
            font-weight: 600;
            color: #111827;
            margin: 0 0 clamp(16px, 4vw, 20px) 0;
            line-height: 1.4;
        }
        
        .message-text {
            font-size: clamp(14px, 3.5vw, 16px);
            color: #374151;
            line-height: 1.65;
            margin: 0 0 clamp(14px, 3.5vw, 16px) 0;
            font-weight: 400;
        }
        
        .message-text.bold {
            font-weight: 600;
            color: #111827;
        }
        
        .message-text.highlight {
            background: linear-gradient(135deg, rgba(17, 60, 207, 0.08) 0%, rgba(30, 79, 229, 0.06) 100%);
            padding: clamp(12px, 4vw, 16px) clamp(16px, 4vw, 20px);
            border-radius: clamp(8px, 2vw, 12px);
            border-left: 4px solid var(--cinebrain-primary);
            margin: clamp(16px, 4vw, 20px) 0;
        }
        
        /* Button section - Mobile responsive */
        .btn-container {
            text-align: center;
            margin: clamp(24px, 6vw, 30px) 0;
        }
        
        .btn-primary {
            display: inline-block;
            background: linear-gradient(135deg, var(--cinebrain-primary) 0%, var(--cinebrain-primary-light) 100%);
            color: #ffffff !important;
            text-decoration: none;
            font-size: clamp(14px, 3.5vw, 16px);
            font-weight: 600;
            padding: clamp(14px, 3.5vw, 16px) clamp(24px, 6vw, 32px);
            border-radius: clamp(8px, 2vw, 12px);
            letter-spacing: clamp(0.3px, 0.1vw, 0.5px);
            box-shadow: 0 10px 30px rgba(17, 60, 207, 0.3);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: none;
            line-height: 1.4;
            position: relative;
            overflow: hidden;
            min-height: clamp(44px, 12vw, 54px);
            width: auto;
            max-width: 100%;
            box-sizing: border-box;
        }
        
        .btn-primary::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, var(--cinebrain-primary-light) 0%, var(--cinebrain-accent) 100%);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .btn-primary:hover::before {
            opacity: 1;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 40px rgba(17, 60, 207, 0.4);
            color: #ffffff !important;
        }
        
        .btn-primary span {
            position: relative;
            z-index: 1;
        }
        
        /* Info box - Mobile responsive */
        .info-box {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 1px solid #e2e8f0;
            border-radius: clamp(8px, 2vw, 12px);
            padding: clamp(16px, 4vw, 20px);
            margin: clamp(20px, 5vw, 24px) 0;
            border-left: 4px solid var(--cinebrain-primary);
        }
        
        .info-title {
            font-size: clamp(11px, 2.8vw, 13px);
            font-weight: 700;
            color: #374151;
            margin: 0 0 clamp(10px, 2.5vw, 12px) 0;
            text-transform: uppercase;
            letter-spacing: clamp(0.6px, 0.15vw, 0.8px);
        }
        
        .info-item {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin: clamp(6px, 2vw, 8px) 0;
            font-size: clamp(12px, 3vw, 14px);
            flex-wrap: wrap;
            gap: clamp(4px, 1vw, 8px);
        }
        
        .info-label {
            color: #6b7280;
            font-weight: 500;
            flex: 0 0 auto;
            min-width: clamp(80px, 20vw, 120px);
        }
        
        .info-value {
            color: #111827;
            font-weight: 600;
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            word-break: break-word;
            text-align: right;
            flex: 1;
        }
        
        /* Security notice - Mobile responsive */
        .security-notice {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.06) 100%);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: clamp(8px, 2vw, 12px);
            padding: clamp(12px, 3vw, 16px);
            margin: clamp(16px, 4vw, 20px) 0;
            border-left: 4px solid #10b981;
        }
        
        .security-notice.warning {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(220, 38, 38, 0.06) 100%);
            border-color: rgba(239, 68, 68, 0.2);
            border-left-color: #ef4444;
        }
        
        .security-icon {
            font-size: clamp(16px, 4vw, 18px);
            margin-bottom: clamp(4px, 1.5vw, 6px);
            display: block;
            line-height: 1;
        }
        
        .security-text {
            font-size: clamp(12px, 3vw, 14px);
            line-height: 1.5;
            color: #374151;
            margin: 0;
            font-weight: 500;
        }
        
        /* Footer - Mobile responsive and compact */
        .footer {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            padding: clamp(12px, 3vw, 16px) clamp(20px, 5vw, 40px);
            border-top: 1px solid #e5e7eb;
        }
        
        .footer-content {
            text-align: center;
        }
        
        .footer-text {
            font-size: clamp(10px, 2.5vw, 12px);
            color: #6b7280;
            margin: 0 0 clamp(6px, 2vw, 8px) 0;
            line-height: 1.4;
            font-weight: 400;
        }
        
        .footer-brand {
            font-size: clamp(11px, 2.8vw, 13px);
            color: #374151;
            margin: 0 0 clamp(6px, 2vw, 8px) 0;
            font-weight: 600;
        }
        
        .footer-links {
            margin: 0;
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: clamp(6px, 2vw, 8px);
        }
        
        .footer-link {
            color: var(--cinebrain-primary);
            text-decoration: none;
            font-size: clamp(10px, 2.5vw, 12px);
            font-weight: 500;
            transition: color 0.3s ease;
            display: inline-block;
            padding: clamp(2px, 0.5vw, 4px) 0;
            white-space: nowrap;
        }
        
        .footer-link:hover {
            color: #0A2A9F;
            text-decoration: underline;
        }
        
        /* Mobile specific adjustments */
        @media only screen and (max-width: 480px) {
            .email-container {
                width: calc(100% - 20px);
                margin: 0 10px;
                border-radius: 12px;
            }
            
            .btn-primary {
                width: 100%;
                padding: 16px 20px;
                min-height: 48px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .info-item {
                flex-direction: column;
                align-items: flex-start;
                gap: 2px;
            }
            
            .info-label {
                min-width: auto;
            }
            
            .info-value {
                text-align: left;
                margin-top: 2px;
            }
            
            .footer-links {
                flex-direction: column;
                gap: 4px;
            }
            
            .footer-link {
                display: block;
                margin: 2px 0;
            }
        }
        
        /* Small mobile devices */
        @media only screen and (max-width: 360px) {
            .email-container {
                width: calc(100% - 16px);
                margin: 0 8px;
            }
            
            .content-title {
                font-size: 20px;
                letter-spacing: 0.3px;
            }
            
            .btn-primary {
                padding: 14px 16px;
                min-height: 44px;
                font-size: 14px;
            }
        }
        
        /* Large screens - maintain desktop design */
        @media only screen and (min-width: 769px) {
            .email-wrapper {
                padding: 20px 0;
            }
            
            .email-container {
                width: 600px;
            }
            
            .header {
                padding: 20px 40px;
            }
            
            .content {
                padding: 30px 40px;
            }
            
            .footer {
                padding: 16px 40px;
            }
            
            .btn-primary {
                width: auto;
                display: inline-block;
            }
            
            .footer-links {
                flex-direction: row;
            }
            
            .footer-link {
                display: inline;
                margin: 0 8px;
            }
            
            .info-item {
                flex-direction: row;
                align-items: center;
            }
            
            .info-value {
                text-align: right;
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
            
            .header {
                background-color: #ffffff !important;
            }
        }
        
        /* High contrast mode */
        @media (prefers-contrast: high) {
            .btn-primary {
                border: 3px solid var(--cinebrain-primary);
            }
            
            .info-box,
            .security-notice {
                border-width: 2px;
            }
            
            .content-title {
                color: var(--cinebrain-primary) !important;
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
                background: #ffffff !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }
            
            .btn-primary {
                background: var(--cinebrain-primary) !important;
                border: 2px solid var(--cinebrain-primary) !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }
            
            .content-title {
                color: #111827 !important;
            }
        }
        
        /* Accessibility improvements */
        @media (prefers-reduced-motion: reduce) {
            .btn-primary,
            .btn-primary::before {
                transition: none;
            }
            
            .btn-primary:hover {
                transform: none;
            }
        }
        
        /* Ultra-wide screens */
        @media only screen and (min-width: 1200px) {
            .email-container {
                max-width: 640px;
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
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes, minimum-scale=1.0, maximum-scale=5.0">
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
                        <img src="https://i.postimg.cc/2S7RfBJR/cinebrain-brand-4k.png" alt="CineBrain" class="brand-logo">
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
                                    <span>Reset My Password</span>
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
                            <p class="footer-text">CineBrain is committed to protecting your account security and privacy.</p>
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
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes, minimum-scale=1.0, maximum-scale=5.0">
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
                        <img src="https://i.postimg.cc/2S7RfBJR/cinebrain-brand-4k.png" alt="CineBrain" class="brand-logo">
                        <p class="brand-subtitle">The Mind Behind Your Next Favorite</p>
                    </div>
                    
                    <div class="content">
                        <div class="content-header">
                            <h2 class="content-title" style="color: #10b981;">Password Updated Successfully</h2>
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
                                    <span>Sign In to CineBrain</span>
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
                            <p class="footer-text">CineBrain is committed to protecting your account security and privacy.</p>
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
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes, minimum-scale=1.0, maximum-scale=5.0">
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
                        <img src="https://i.postimg.cc/2S7RfBJR/cinebrain-brand-4k.png" alt="CineBrain" class="brand-logo">
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
                            <p class="footer-text">The Mind Behind Your Next Favorite - Discover personalized entertainment recommendations.</p>
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