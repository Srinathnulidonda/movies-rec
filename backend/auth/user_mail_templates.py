# auth/user_mail_templates.py

from datetime import datetime
import os

FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')
BRAND_IMAGE_URL = 'https://i.postimg.cc/W3H6FXDR/cinebrain-brand-4k.png'

def get_professional_template(content_type: str, **kwargs) -> tuple:
    """Get world-class email templates inspired by top tech companies"""
    base_css = """
    <style type="text/css">
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Reset */
        body, table, td, a { -webkit-text-size-adjust: 100%; -ms-text-size-adjust: 100%; }
        table, td { mso-table-lspace: 0pt; mso-table-rspace: 0pt; }
        img { -ms-interpolation-mode: bicubic; border: 0; outline: none; text-decoration: none; display: block; }
        body { margin: 0; padding: 0; width: 100%; height: 100%; }
        
        /* Prevent auto-scaling */
        .ReadMsgBody { width: 100%; }
        .ExternalClass { width: 100%; }
        .ExternalClass, .ExternalClass p, .ExternalClass span, .ExternalClass font, .ExternalClass td, .ExternalClass div { line-height: 100%; }
        
        /* Remove spacing */
        table { border-collapse: collapse; }
        
        /* Mobile Styles */
        @media only screen and (max-width: 600px) {
            .wrapper { width: 100% !important; }
            .container { width: 100% !important; min-width: 100% !important; }
            .content-cell { padding: 24px 20px !important; }
            .header-cell { padding: 24px 20px !important; }
            .footer-cell { padding: 20px !important; }
            .brand-img { width: 180px !important; height: auto !important; }
            .h1 { font-size: 20px !important; line-height: 26px !important; }
            .h2 { font-size: 16px !important; line-height: 22px !important; }
            .text { font-size: 14px !important; line-height: 22px !important; }
            .btn { padding: 12px 24px !important; font-size: 14px !important; }
            .info-table { padding: 16px !important; }
            .footer-link { display: inline-block !important; margin: 0 6px !important; font-size: 12px !important; }
            .footer-divider { margin: 0 2px !important; }
            .copyright { font-size: 11px !important; line-height: 18px !important; }
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
    """Generate mobile-optimized password reset email template"""
    reset_url = kwargs.get('reset_url', '')
    user_name = kwargs.get('user_name', 'there')
    
    html = f"""<!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="format-detection" content="telephone=no, date=no, address=no, email=no">
    <meta name="x-apple-disable-message-reformatting">
    <title>Reset your password</title>
    {base_css}
</head>
<body style="margin: 0; padding: 0; background-color: #f6f9fc; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;">
    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #f6f9fc; margin: 0; padding: 0;">
        <tr>
            <td align="center" style="padding: 20px 0;">
                <table role="presentation" cellspacing="0" cellpadding="0" border="0" class="container" style="width: 100%; max-width: 600px; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    
                    <!-- Header -->
                    <tr>
                        <td class="header-cell" align="center" style="background-color: #000000; padding: 32px 24px;">
                            <img src="{BRAND_IMAGE_URL}" alt="CineBrain" class="brand-img" width="220" height="auto" style="display: block; margin: 0 auto; max-width: 100%;">
                        </td>
                    </tr>
                    
                    <!-- Content -->
                    <tr>
                        <td class="content-cell" style="padding: 36px 40px 24px;">
                            <h1 class="h1" style="font-size: 24px; font-weight: 700; color: #1a1a1a; margin: 0 0 8px 0; line-height: 1.3; letter-spacing: -0.02em;">Reset your password</h1>
                            <p class="h2" style="font-size: 15px; color: #6b7280; margin: 0 0 24px 0; line-height: 1.5;">We've received a request to reset your password</p>
                            
                            <p class="text" style="font-size: 15px; color: #1a1a1a; margin: 0 0 16px 0; font-weight: 500;">Hi {user_name} 👋</p>
                            
                            <p class="text" style="font-size: 15px; color: #4b5563; line-height: 1.6; margin: 0 0 24px 0;">Someone requested a password reset for your CineBrain account. If this was you, click the button below to set a new password.</p>
                            
                            <!-- Button -->
                            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                <tr>
                                    <td align="center" style="padding: 8px 0 24px 0;">
                                        <!--[if mso]>
                                        <v:roundrect xmlns:v="urn:schemas-microsoft-com:vml" xmlns:w="urn:schemas-microsoft-com:office:word" href="{reset_url}" style="height:44px;v-text-anchor:middle;width:180px;" arcsize="18%" stroke="f" fillcolor="#113CCF">
                                            <w:anchorlock/>
                                            <center style="color:#ffffff;font-family:sans-serif;font-size:15px;font-weight:600;">Reset Password</center>
                                        </v:roundrect>
                                        <![endif]-->
                                        <!--[if !mso]><!-->
                                        <a href="{reset_url}" class="btn" style="display: inline-block; background: #113CCF; color: #ffffff; text-decoration: none; font-size: 15px; font-weight: 600; padding: 13px 28px; border-radius: 8px; box-shadow: 0 2px 6px rgba(17,60,207,0.24);" target="_blank">Reset Password</a>
                                        <!--<![endif]-->
                                    </td>
                                </tr>
                            </table>
                            
                            <!-- Alert -->
                            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #fef3c7; border: 1px solid #fcd34d; border-radius: 8px; margin: 0 0 20px 0;">
                                <tr>
                                    <td style="padding: 14px 16px;">
                                        <table role="presentation" cellspacing="0" cellpadding="0" border="0">
                                            <tr>
                                                <td style="width: 28px; vertical-align: top; padding-top: 2px;"><span style="font-size: 20px; line-height: 1;">⏰</span></td>
                                                <td style="vertical-align: top;">
                                                    <p style="font-size: 13px; font-weight: 600; color: #1a1a1a; margin: 0 0 3px 0; line-height: 1.3;">This link expires in 1 hour</p>
                                                    <p style="font-size: 13px; color: #4b5563; margin: 0; line-height: 1.4;">For security reasons, this link will expire in 60 minutes.</p>
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
                            </table>
                            
                            <p style="font-size: 13px; color: #9ca3af; margin: 0; line-height: 1.4;">If you didn't request this, you can safely ignore this email.</p>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td class="footer-cell" style="background-color: #f9fafb; padding: 24px 40px; border-top: 1px solid #e5e7eb;">
                            <p class="copyright" align="center" style="font-size: 12px; color: #9ca3af; margin: 0 0 12px 0; line-height: 1.5;">© {datetime.now().year} CineBrain. All rights reserved.<br>AI-powered movie recommendations</p>
                            
                            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                <tr>
                                    <td align="center" style="padding: 0;">
                                        <a href="{FRONTEND_URL}" class="footer-link" style="color: #6b7280; text-decoration: none; font-size: 13px; margin: 0 10px; font-weight: 500;">Visit CineBrain</a><span class="footer-divider" style="color: #d1d5db; margin: 0 3px;">•</span><a href="{FRONTEND_URL}/privacy" class="footer-link" style="color: #6b7280; text-decoration: none; font-size: 13px; margin: 0 10px; font-weight: 500;">Privacy</a><span class="footer-divider" style="color: #d1d5db; margin: 0 3px;">•</span><a href="{FRONTEND_URL}/terms" class="footer-link" style="color: #6b7280; text-decoration: none; font-size: 13px; margin: 0 10px; font-weight: 500;">Terms</a><span class="footer-divider" style="color: #d1d5db; margin: 0 3px;">•</span><a href="{FRONTEND_URL}/support" class="footer-link" style="color: #6b7280; text-decoration: none; font-size: 13px; margin: 0 10px; font-weight: 500;">Get Help</a>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""
    
    text = f"""CineBrain
Reset your password

Hi {user_name} 👋

Someone requested a password reset for your CineBrain account. Click the link below to set a new password:

{reset_url}

⏰ This link expires in 1 hour

If you didn't request this, you can safely ignore this email.

---
© {datetime.now().year} CineBrain. All rights reserved.
AI-powered movie recommendations

Visit CineBrain: {FRONTEND_URL}
Privacy: {FRONTEND_URL}/privacy
Terms: {FRONTEND_URL}/terms
Get Help: {FRONTEND_URL}/support"""
    
    return html, text

def _get_password_changed_template(base_css: str, **kwargs) -> tuple:
    """Generate mobile-optimized password changed template"""
    user_name = kwargs.get('user_name', 'there')
    ip_address = kwargs.get('ip_address', 'Unknown')
    device = kwargs.get('device', 'Unknown device')
    timestamp = datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')
    
    html = f"""<!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="format-detection" content="telephone=no, date=no, address=no, email=no">
    <meta name="x-apple-disable-message-reformatting">
    <title>Password changed successfully</title>
    {base_css}
</head>
<body style="margin: 0; padding: 0; background-color: #f6f9fc; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;">
    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #f6f9fc; margin: 0; padding: 0;">
        <tr>
            <td align="center" style="padding: 20px 0;">
                <table role="presentation" cellspacing="0" cellpadding="0" border="0" class="container" style="width: 100%; max-width: 600px; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    
                    <!-- Header -->
                    <tr>
                        <td class="header-cell" align="center" style="background-color: #000000; padding: 32px 24px;">
                            <img src="{BRAND_IMAGE_URL}" alt="CineBrain" class="brand-img" width="220" height="auto" style="display: block; margin: 0 auto; max-width: 100%;">
                        </td>
                    </tr>
                    
                    <!-- Content -->
                    <tr>
                        <td class="content-cell" style="padding: 36px 40px 24px;">
                            <h1 class="h1" style="font-size: 24px; font-weight: 700; color: #1a1a1a; margin: 0 0 8px 0; line-height: 1.3; letter-spacing: -0.02em;">Password changed ✅</h1>
                            <p class="h2" style="font-size: 15px; color: #6b7280; margin: 0 0 24px 0; line-height: 1.5;">Your account is now secured with a new password</p>
                            
                            <p class="text" style="font-size: 15px; color: #1a1a1a; margin: 0 0 16px 0; font-weight: 500;">Hi {user_name} 👋</p>
                            
                            <p class="text" style="font-size: 15px; color: #4b5563; line-height: 1.6; margin: 0 0 20px 0;">Great news! Your password has been successfully updated. You can now sign in with your new password.</p>
                            
                            <!-- Info Card -->
                            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" class="info-table" style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 10px; border: 1px solid #e5e7eb; margin: 0 0 20px 0;">
                                <tr>
                                    <td style="padding: 18px;">
                                        <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                            <tr>
                                                <td style="padding: 6px 0; border-bottom: 1px solid #e5e7eb;">
                                                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                                        <tr>
                                                            <td style="font-size: 13px; color: #6b7280; font-weight: 500; width: 32%; vertical-align: top;">Changed on</td>
                                                            <td style="font-size: 13px; color: #1a1a1a; font-weight: 600; width: 68%; word-wrap: break-word;">{timestamp}</td>
                                                        </tr>
                                                    </table>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td style="padding: 6px 0; border-bottom: 1px solid #e5e7eb;">
                                                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                                        <tr>
                                                            <td style="font-size: 13px; color: #6b7280; font-weight: 500; width: 32%; vertical-align: top;">Device</td>
                                                            <td style="font-size: 13px; color: #1a1a1a; font-weight: 600; width: 68%; word-wrap: break-word;">{device}</td>
                                                        </tr>
                                                    </table>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td style="padding: 6px 0;">
                                                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                                        <tr>
                                                            <td style="font-size: 13px; color: #6b7280; font-weight: 500; width: 32%; vertical-align: top;">IP Address</td>
                                                            <td style="font-size: 13px; color: #1a1a1a; font-weight: 600; width: 68%; word-wrap: break-word;">{ip_address}</td>
                                                        </tr>
                                                    </table>
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
                            </table>
                            
                            <!-- Success Alert -->
                            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #d1fae5; border: 1px solid #6ee7b7; border-radius: 8px; margin: 0 0 20px 0;">
                                <tr>
                                    <td style="padding: 14px 16px;">
                                        <table role="presentation" cellspacing="0" cellpadding="0" border="0">
                                            <tr>
                                                <td style="width: 28px; vertical-align: top; padding-top: 2px;"><span style="font-size: 20px; line-height: 1;">🛡️</span></td>
                                                <td style="vertical-align: top;">
                                                    <p style="font-size: 13px; font-weight: 600; color: #1a1a1a; margin: 0 0 3px 0; line-height: 1.3;">Was this you?</p>
                                                    <p style="font-size: 13px; color: #4b5563; margin: 0; line-height: 1.4;">If you made this change, no further action is needed.</p>
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
                            </table>
                            
                            <p class="text" style="font-size: 14px; color: #4b5563; line-height: 1.6; margin: 0 0 20px 0;"><strong style="color: #1a1a1a;">Didn't change your password?</strong><br>Please secure your account immediately and contact support.</p>
                            
                            <!-- Button -->
                            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                <tr>
                                    <td align="center" style="padding: 4px 0 20px 0;">
                                        <!--[if mso]>
                                        <v:roundrect xmlns:v="urn:schemas-microsoft-com:vml" xmlns:w="urn:schemas-microsoft-com:office:word" href="{FRONTEND_URL}/login" style="height:44px;v-text-anchor:middle;width:160px;" arcsize="18%" stroke="f" fillcolor="#113CCF">
                                            <w:anchorlock/>
                                            <center style="color:#ffffff;font-family:sans-serif;font-size:15px;font-weight:600;">Sign In</center>
                                        </v:roundrect>
                                        <![endif]-->
                                        <!--[if !mso]><!-->
                                        <a href="{FRONTEND_URL}/login" class="btn" style="display: inline-block; background: #113CCF; color: #ffffff; text-decoration: none; font-size: 15px; font-weight: 600; padding: 13px 28px; border-radius: 8px; box-shadow: 0 2px 6px rgba(17,60,207,0.24);" target="_blank">Sign In to CineBrain</a>
                                        <!--<![endif]-->
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td class="footer-cell" style="background-color: #f9fafb; padding: 24px 40px; border-top: 1px solid #e5e7eb;">
                            <p class="copyright" align="center" style="font-size: 12px; color: #9ca3af; margin: 0 0 12px 0; line-height: 1.5;">© {datetime.now().year} CineBrain. All rights reserved.<br>AI-powered movie recommendations</p>
                            
                            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                <tr>
                                    <td align="center" style="padding: 0;">
                                        <a href="{FRONTEND_URL}" class="footer-link" style="color: #6b7280; text-decoration: none; font-size: 13px; margin: 0 10px; font-weight: 500;">Visit CineBrain</a><span class="footer-divider" style="color: #d1d5db; margin: 0 3px;">•</span><a href="{FRONTEND_URL}/privacy" class="footer-link" style="color: #6b7280; text-decoration: none; font-size: 13px; margin: 0 10px; font-weight: 500;">Privacy</a><span class="footer-divider" style="color: #d1d5db; margin: 0 3px;">•</span><a href="{FRONTEND_URL}/terms" class="footer-link" style="color: #6b7280; text-decoration: none; font-size: 13px; margin: 0 10px; font-weight: 500;">Terms</a><span class="footer-divider" style="color: #d1d5db; margin: 0 3px;">•</span><a href="{FRONTEND_URL}/support" class="footer-link" style="color: #6b7280; text-decoration: none; font-size: 13px; margin: 0 10px; font-weight: 500;">Get Help</a>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""
    
    text = f"""CineBrain
Password changed successfully ✅

Hi {user_name} 👋

Great news! Your password has been successfully updated.

Details:
• Changed on: {timestamp}
• Device: {device}
• IP Address: {ip_address}

🛡️ Was this you?
If you made this change, no further action is needed.

Didn't change your password?
Please secure your account immediately and contact support.

Sign in: {FRONTEND_URL}/login

---
© {datetime.now().year} CineBrain. All rights reserved.
AI-powered movie recommendations

Visit CineBrain: {FRONTEND_URL}
Privacy: {FRONTEND_URL}/privacy
Terms: {FRONTEND_URL}/terms
Get Help: {FRONTEND_URL}/support"""
    
    return html, text

def _get_generic_template(base_css: str, **kwargs) -> tuple:
    """Generate mobile-optimized generic template"""
    subject = kwargs.get('subject', 'CineBrain Update')
    content = kwargs.get('content', '')
    user_name = kwargs.get('user_name', 'there')
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{subject}</title>
    {base_css}
</head>
<body style="margin: 0; padding: 0; background-color: #f6f9fc; font-family: 'Inter', Arial, sans-serif;">
    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #f6f9fc;">
        <tr>
            <td align="center" style="padding: 20px 0;">
                <table role="presentation" cellspacing="0" cellpadding="0" border="0" class="container" style="width: 100%; max-width: 600px; background-color: #ffffff; border-radius: 12px;">
                    <tr>
                        <td class="header-cell" align="center" style="background-color: #000000; padding: 32px 24px;">
                            <img src="{BRAND_IMAGE_URL}" alt="CineBrain" class="brand-img" width="220" height="auto" style="display: block; margin: 0 auto;">
                        </td>
                    </tr>
                    <tr>
                        <td class="content-cell" style="padding: 36px 40px 24px;">
                            <h1 class="h1" style="font-size: 24px; color: #1a1a1a; margin: 0 0 20px 0;">{subject}</h1>
                            <p class="text" style="font-size: 15px; margin: 0 0 16px 0;">Hi {user_name} 👋</p>
                            <div class="text" style="font-size: 15px; color: #4b5563; line-height: 1.6;">{content}</div>
                            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                <tr>
                                    <td align="center" style="padding: 24px 0 20px 0;">
                                        <a href="{FRONTEND_URL}" class="btn" style="display: inline-block; background: #113CCF; color: #ffffff; text-decoration: none; font-size: 15px; font-weight: 600; padding: 13px 28px; border-radius: 8px;">Visit CineBrain</a>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td class="footer-cell" style="background-color: #f9fafb; padding: 24px 40px; border-top: 1px solid #e5e7eb;">
                            <p class="copyright" align="center" style="font-size: 12px; color: #9ca3af; margin: 0 0 12px 0; line-height: 1.5;">© {datetime.now().year} CineBrain. All rights reserved.<br>AI-powered movie recommendations</p>
                            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                <tr>
                                    <td align="center">
                                        <a href="{FRONTEND_URL}" class="footer-link" style="color: #6b7280; text-decoration: none; font-size: 13px; margin: 0 10px;">Visit CineBrain</a><span style="color: #d1d5db;">•</span><a href="{FRONTEND_URL}/privacy" class="footer-link" style="color: #6b7280; text-decoration: none; font-size: 13px; margin: 0 10px;">Privacy</a><span style="color: #d1d5db;">•</span><a href="{FRONTEND_URL}/terms" class="footer-link" style="color: #6b7280; text-decoration: none; font-size: 13px; margin: 0 10px;">Terms</a><span style="color: #d1d5db;">•</span><a href="{FRONTEND_URL}/support" class="footer-link" style="color: #6b7280; text-decoration: none; font-size: 13px; margin: 0 10px;">Get Help</a>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""
    
    text = f"""CineBrain
{subject}

Hi {user_name} 👋

{content}

Visit: {FRONTEND_URL}

---
© {datetime.now().year} CineBrain. All rights reserved.
AI-powered movie recommendations

Visit CineBrain: {FRONTEND_URL}
Privacy: {FRONTEND_URL}/privacy
Terms: {FRONTEND_URL}/terms
Get Help: {FRONTEND_URL}/support"""
    
    return html, text