# auth/admin_mail_templates.py

from datetime import datetime
import os

FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')
BRAND_IMAGE_URL = 'https://i.postimg.cc/W3H6FXDR/cinebrain-brand-4k.png'

def get_admin_template(template_type: str, **kwargs) -> tuple:
    """Get professional admin email templates matching CineBrain design"""
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
            .alert-box { padding: 16px !important; }
            .footer-link { display: inline-block !important; margin: 0 6px !important; font-size: 12px !important; }
            .footer-divider { margin: 0 2px !important; }
            .copyright { font-size: 11px !important; line-height: 18px !important; }
        }
    </style>
    """
    
    if template_type == 'admin_notification':
        return _get_admin_notification_template(base_css, **kwargs)
    elif template_type == 'system_alert':
        return _get_system_alert_template(base_css, **kwargs)
    elif template_type == 'daily_summary':
        return _get_daily_summary_template(base_css, **kwargs)
    elif template_type == 'urgent_notification':
        return _get_urgent_notification_template(base_css, **kwargs)
    else:
        return _get_generic_admin_template(base_css, **kwargs)

def _get_admin_notification_template(base_css: str, **kwargs) -> tuple:
    """General admin notification template"""
    subject = kwargs.get('subject', 'CineBrain Admin Alert')
    content = kwargs.get('content', '')
    is_urgent = kwargs.get('is_urgent', False)
    timestamp = kwargs.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))
    
    # Choose header color based on urgency
    header_gradient = "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)" if is_urgent else "linear-gradient(135deg, #1a73e8 0%, #4285f4 100%)"
    priority_text = "🚨 URGENT ALERT" if is_urgent else "📋 ADMIN NOTIFICATION"
    
    html = f"""<!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="format-detection" content="telephone=no, date=no, address=no, email=no">
    <meta name="x-apple-disable-message-reformatting">
    <title>{subject}</title>
    {base_css}
</head>
<body style="margin: 0; padding: 0; background-color: #f6f9fc; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;">
    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #f6f9fc; margin: 0; padding: 0;">
        <tr>
            <td align="center" style="padding: 20px 0;">
                <table role="presentation" cellspacing="0" cellpadding="0" border="0" class="container" style="width: 100%; max-width: 600px; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    
                    <!-- Header -->
                    <tr>
                        <td class="header-cell" align="center" style="background: {header_gradient}; padding: 40px 24px;">
                            <img src="{BRAND_IMAGE_URL}" alt="CineBrain" class="brand-img" width="220" height="auto" style="display: block; margin: 0 auto 16px auto; max-width: 100%;">
                            <p style="color: #ffffff; font-size: 16px; margin: 0; opacity: 0.9; font-weight: 500;">{priority_text}</p>
                        </td>
                    </tr>
                    
                    <!-- Content -->
                    <tr>
                        <td class="content-cell" style="padding: 36px 40px 24px;">
                            <!-- Alert Header -->
                            <div align="center" style="margin: 0 0 32px 0;">
                                <div style="font-size: 48px; line-height: 1; margin: 0 0 16px 0;">{'🚨' if is_urgent else '🔔'}</div>
                                <h1 class="h1" style="font-size: 26px; font-weight: 700; color: #1a1a1a; margin: 0 0 8px 0; line-height: 1.2; letter-spacing: -0.02em;">{subject}</h1>
                                <p class="h2" style="font-size: 16px; color: #6b7280; margin: 0; line-height: 1.4;">CineBrain Admin Dashboard</p>
                            </div>
                            
                            <p class="text" style="font-size: 15px; color: #1a1a1a; margin: 0 0 20px 0; font-weight: 500;">Hello Admin 👋</p>
                            
                            <p class="text" style="font-size: 15px; color: #4b5563; line-height: 1.6; margin: 0 0 24px 0;">You're receiving this notification because an important event has occurred on the CineBrain platform that requires your attention.</p>
                            
                            <!-- Alert Details -->
                            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" class="alert-box" style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 12px; border: 1px solid #e5e7eb; margin: 0 0 24px 0;">
                                <tr>
                                    <td style="padding: 24px;">
                                        <h3 style="font-size: 16px; font-weight: 700; color: #1a1a1a; margin: 0 0 16px 0;">📋 Alert Details</h3>
                                        
                                        <div style="background: #ffffff; padding: 20px; border-left: 4px solid {'#ef4444' if is_urgent else '#3b82f6'}; border-radius: 8px; font-size: 14px; line-height: 1.6; color: #374151;">
                                            {content}
                                        </div>
                                        
                                        <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="margin-top: 16px;">
                                            <tr>
                                                <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb;">
                                                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                                        <tr>
                                                            <td style="font-size: 13px; color: #6b7280; font-weight: 500; width: 25%; vertical-align: top;">Priority</td>
                                                            <td style="font-size: 12px; color: {'#ef4444' if is_urgent else '#3b82f6'}; font-weight: 700; width: 75%; text-transform: uppercase; background: {'#ef444420' if is_urgent else '#3b82f620'}; padding: 2px 8px; border-radius: 12px; display: inline-block;">{'URGENT' if is_urgent else 'NORMAL'}</td>
                                                        </tr>
                                                    </table>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb;">
                                                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                                        <tr>
                                                            <td style="font-size: 13px; color: #6b7280; font-weight: 500; width: 25%; vertical-align: top;">System</td>
                                                            <td style="font-size: 13px; color: #1a1a1a; font-weight: 600; width: 75%;">CineBrain Admin Dashboard</td>
                                                        </tr>
                                                    </table>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td style="padding: 8px 0;">
                                                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                                        <tr>
                                                            <td style="font-size: 13px; color: #6b7280; font-weight: 500; width: 25%; vertical-align: top;">Timestamp</td>
                                                            <td style="font-size: 13px; color: #1a1a1a; font-weight: 600; width: 75%;">{timestamp}</td>
                                                        </tr>
                                                    </table>
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
                            </table>
                            
                            <!-- Action Required -->
                            {f'''
                            <div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border: 1px solid #fca5a5; border-radius: 10px; padding: 20px; margin: 0 0 24px 0;">
                                <h3 style="font-size: 15px; font-weight: 700; color: #dc2626; margin: 0 0 12px 0;">⚡ Immediate Action Required</h3>
                                <p style="font-size: 13px; color: #7f1d1d; margin: 0; line-height: 1.5;">This is an urgent notification that requires immediate attention. Please log into the admin dashboard as soon as possible to review and take action.</p>
                            </div>
                            ''' if is_urgent else '''
                            <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 1px solid #93c5fd; border-radius: 10px; padding: 20px; margin: 0 0 24px 0;">
                                <h3 style="font-size: 15px; font-weight: 700; color: #1e40af; margin: 0 0 12px 0;">📋 Action Recommended</h3>
                                <p style="font-size: 13px; color: #1e3a8a; margin: 0; line-height: 1.5;">Please review this notification when convenient and take any necessary action through the admin dashboard.</p>
                            </div>
                            '''}
                            
                            <!-- CTA Button -->
                            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                <tr>
                                    <td align="center" style="padding: 8px 0 24px 0;">
                                        <!--[if mso]>
                                        <v:roundrect xmlns:v="urn:schemas-microsoft-com:vml" xmlns:w="urn:schemas-microsoft-com:office:word" href="{FRONTEND_URL}/admin" style="height:48px;v-text-anchor:middle;width:220px;" arcsize="15%" stroke="f" fillcolor="#113CCF">
                                            <w:anchorlock/>
                                            <center style="color:#ffffff;font-family:sans-serif;font-size:16px;font-weight:700;">Open Admin Dashboard</center>
                                        </v:roundrect>
                                        <![endif]-->
                                        <!--[if !mso]><!-->
                                        <a href="{FRONTEND_URL}/admin" class="btn" style="display: inline-block; background: #113CCF; color: #ffffff; text-decoration: none; font-size: 16px; font-weight: 700; padding: 15px 32px; border-radius: 8px; box-shadow: 0 4px 12px rgba(17,60,207,0.3); transition: all 0.2s;" target="_blank">Open Admin Dashboard 🎛️</a>
                                        <!--<![endif]-->
                                    </td>
                                </tr>
                            </table>
                            
                            <p style="font-size: 13px; color: #9ca3af; margin: 0; line-height: 1.4;">This is an automated notification from the CineBrain admin system. If you have questions, please contact the development team.</p>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td class="footer-cell" style="background-color: #f9fafb; padding: 24px 40px; border-top: 1px solid #e5e7eb;">
                            <p class="copyright" align="center" style="font-size: 12px; color: #9ca3af; margin: 0 0 12px 0; line-height: 1.5;">© {datetime.now().year} CineBrain Admin System. All rights reserved.<br>Automated notification • {timestamp}</p>
                            
                            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                <tr>
                                    <td align="center" style="padding: 0;">
                                        <a href="{FRONTEND_URL}/admin" class="footer-link" style="color: #6b7280; text-decoration: none; font-size: 13px; margin: 0 10px; font-weight: 500;">Admin Dashboard</a><span class="footer-divider" style="color: #d1d5db; margin: 0 3px;">•</span><a href="{FRONTEND_URL}/admin/system-health" class="footer-link" style="color: #6b7280; text-decoration: none; font-size: 13px; margin: 0 10px; font-weight: 500;">System Health</a><span class="footer-divider" style="color: #d1d5db; margin: 0 3px;">•</span><a href="{FRONTEND_URL}/admin/support" class="footer-link" style="color: #6b7280; text-decoration: none; font-size: 13px; margin: 0 10px; font-weight: 500;">Support Center</a>
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
    
    text = f"""{'🚨 URGENT ALERT' if is_urgent else '🔔 ADMIN NOTIFICATION'} - CineBrain

{subject}

Hello Admin 👋

{content}

Priority: {'URGENT' if is_urgent else 'NORMAL'}
System: CineBrain Admin Dashboard
Timestamp: {timestamp}

{'⚡ IMMEDIATE ACTION REQUIRED - Please log into the admin dashboard immediately.' if is_urgent else '📋 Action recommended - Please review when convenient.'}

Admin Dashboard: {FRONTEND_URL}/admin

---
© {datetime.now().year} CineBrain Admin System
Automated notification • {timestamp}

Admin Dashboard: {FRONTEND_URL}/admin
System Health: {FRONTEND_URL}/admin/system-health
Support Center: {FRONTEND_URL}/admin/support"""
    
    return html, text

def _get_system_alert_template(base_css: str, **kwargs) -> tuple:
    """System alert notification template"""
    alert_type = kwargs.get('alert_type', 'System Alert')
    details = kwargs.get('details', '')
    severity = kwargs.get('severity', 'medium')  # low, medium, high, critical
    
    severity_colors = {
        'low': '#10b981',
        'medium': '#f59e0b', 
        'high': '#ef4444',
        'critical': '#7c2d12'
    }
    
    severity_color = severity_colors.get(severity, '#f59e0b')
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>System Alert - {alert_type}</title>
    {base_css}
</head>
<body style="margin: 0; padding: 0; background-color: #f6f9fc; font-family: 'Inter', Arial, sans-serif;">
    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #f6f9fc;">
        <tr>
            <td align="center" style="padding: 20px 0;">
                <table role="presentation" cellspacing="0" cellpadding="0" border="0" class="container" style="width: 100%; max-width: 600px; background-color: #ffffff; border-radius: 12px;">
                    <tr>
                        <td class="header-cell" align="center" style="background: linear-gradient(135deg, {severity_color} 0%, {severity_color}dd 100%); padding: 32px 24px;">
                            <img src="{BRAND_IMAGE_URL}" alt="CineBrain" class="brand-img" width="220" height="auto" style="display: block; margin: 0 auto 16px auto;">
                            <p style="color: #ffffff; font-size: 16px; margin: 0; opacity: 0.9; font-weight: 500;">⚠️ System Alert</p>
                        </td>
                    </tr>
                    <tr>
                        <td class="content-cell" style="padding: 36px 40px 24px;">
                            <h1 class="h1" style="font-size: 24px; color: #1a1a1a; margin: 0 0 20px 0;">{alert_type}</h1>
                            <div style="background: {severity_color}20; border-left: 4px solid {severity_color}; padding: 16px; margin: 20px 0; border-radius: 8px;">
                                <p style="margin: 0; font-size: 14px; line-height: 1.6; color: #374151;"><strong>Alert Details:</strong><br>{details}</p>
                            </div>
                            <div style="background: #f8fafc; padding: 16px; border-radius: 8px; margin: 20px 0;">
                                <p style="margin: 0; font-size: 14px;"><strong>Severity:</strong> <span style="color: {severity_color}; font-weight: bold; text-transform: uppercase;">{severity}</span></p>
                                <p style="margin: 8px 0 0 0; font-size: 14px;"><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                                <p style="margin: 8px 0 0 0; font-size: 14px;"><strong>System:</strong> CineBrain Platform</p>
                            </div>
                            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                <tr>
                                    <td align="center" style="padding: 24px 0 20px 0;">
                                        <a href="{FRONTEND_URL}/admin/system-health" class="btn" style="display: inline-block; background: #113CCF; color: #ffffff; text-decoration: none; font-size: 15px; font-weight: 600; padding: 13px 28px; border-radius: 8px;">Check System Status</a>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td class="footer-cell" style="background-color: #f9fafb; padding: 24px 40px; border-top: 1px solid #e5e7eb;">
                            <p class="copyright" align="center" style="font-size: 12px; color: #9ca3af; margin: 0; line-height: 1.5;">© {datetime.now().year} CineBrain System Monitor</p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""
    
    text = f"""⚠️ System Alert - CineBrain

{alert_type}

Alert Details:
{details}

Severity: {severity.upper()}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
System: CineBrain Platform

Check System Status: {FRONTEND_URL}/admin/system-health

---
© {datetime.now().year} CineBrain System Monitor"""
    
    return html, text

def _get_daily_summary_template(base_css: str, **kwargs) -> tuple:
    """Daily summary email template"""
    stats = kwargs.get('stats', {})
    highlights = kwargs.get('highlights', [])
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Summary - CineBrain</title>
    {base_css}
</head>
<body style="margin: 0; padding: 0; background-color: #f6f9fc; font-family: 'Inter', Arial, sans-serif;">
    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #f6f9fc;">
        <tr>
            <td align="center" style="padding: 20px 0;">
                <table role="presentation" cellspacing="0" cellpadding="0" border="0" class="container" style="width: 100%; max-width: 600px; background-color: #ffffff; border-radius: 12px;">
                    <tr>
                        <td class="header-cell" align="center" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 32px 24px;">
                            <img src="{BRAND_IMAGE_URL}" alt="CineBrain" class="brand-img" width="220" height="auto" style="display: block; margin: 0 auto 16px auto;">
                            <p style="color: #ffffff; font-size: 16px; margin: 0; opacity: 0.9; font-weight: 500;">📊 Daily Summary</p>
                        </td>
                    </tr>
                    <tr>
                        <td class="content-cell" style="padding: 36px 40px 24px;">
                            <h1 class="h1" style="font-size: 24px; color: #1a1a1a; margin: 0 0 20px 0;">Daily Platform Summary</h1>
                            <p class="text" style="font-size: 15px; color: #4b5563; margin: 0 0 24px 0;">Here's what happened on CineBrain today:</p>
                            
                            <div style="background: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0;">
                                <h3 style="margin: 0 0 16px 0; color: #1a1a1a;">📈 Today's Statistics</h3>
                                <ul style="margin: 0; padding-left: 20px; color: #4b5563;">
                                    <li>New Users: {stats.get('new_users', 0)}</li>
                                    <li>Total Interactions: {stats.get('interactions', 0)}</li>
                                    <li>Support Tickets: {stats.get('tickets', 0)}</li>
                                    <li>Content Added: {stats.get('new_content', 0)}</li>
                                </ul>
                            </div>
                            
                            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                <tr>
                                    <td align="center" style="padding: 24px 0 20px 0;">
                                        <a href="{FRONTEND_URL}/admin" class="btn" style="display: inline-block; background: #113CCF; color: #ffffff; text-decoration: none; font-size: 15px; font-weight: 600; padding: 13px 28px; border-radius: 8px;">View Full Dashboard</a>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td class="footer-cell" style="background-color: #f9fafb; padding: 24px 40px; border-top: 1px solid #e5e7eb;">
                            <p class="copyright" align="center" style="font-size: 12px; color: #9ca3af; margin: 0; line-height: 1.5;">© {datetime.now().year} CineBrain Daily Reports</p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""
    
    text = f"""📊 Daily Summary - CineBrain

Daily Platform Summary

Today's Statistics:
• New Users: {stats.get('new_users', 0)}
• Total Interactions: {stats.get('interactions', 0)}
• Support Tickets: {stats.get('tickets', 0)}
• Content Added: {stats.get('new_content', 0)}

View Full Dashboard: {FRONTEND_URL}/admin

---
© {datetime.now().year} CineBrain Daily Reports"""
    
    return html, text

def _get_urgent_notification_template(base_css: str, **kwargs) -> tuple:
    """Urgent notification template"""
    title = kwargs.get('title', 'Urgent Notification')
    message = kwargs.get('message', '')
    action_url = kwargs.get('action_url', f'{FRONTEND_URL}/admin')
    
    return _get_admin_notification_template(base_css, 
                                          subject=title,
                                          content=message,
                                          is_urgent=True,
                                          **kwargs)

def _get_generic_admin_template(base_css: str, **kwargs) -> tuple:
    """Generic admin template"""
    subject = kwargs.get('subject', 'CineBrain Admin')
    content = kwargs.get('content', '')
    
    return _get_admin_notification_template(base_css,
                                          subject=subject,
                                          content=content,
                                          is_urgent=False,
                                          **kwargs)