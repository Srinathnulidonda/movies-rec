# support/report_issues.py

from flask import request, jsonify
from datetime import datetime
import logging
import re
import jwt
import os
import uuid
import cloudinary
import cloudinary.uploader
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET'),
    secure=True
)

class IssueReportService:
    def __init__(self, app, db, models, services):
        self.app = app
        self.db = db
        self.User = models['User']
        self.IssueReport = models.get('IssueReport')
        self.SupportTicket = models['SupportTicket']
        self.SupportCategory = models['SupportCategory']
        self.TicketActivity = models['TicketActivity']
        
        # Enhanced email service initialization
        self.email_service = self._initialize_email_service(services)
        self.redis_client = services.get('redis_client')
        
        # Cloudinary settings
        self.cloudinary_folder = "report_issues"
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'pdf'}
        
        logger.info("✅ IssueReportService initialized successfully")
    
    def _initialize_email_service(self, services):
        """Initialize email service with fallbacks"""
        email_service = services.get('email_service')
        if email_service:
            return email_service
        
        try:
            from auth.service import email_service as auth_email_service
            if auth_email_service and hasattr(auth_email_service, 'queue_email'):
                logger.info("✅ Email service loaded from auth module for issues")
                return auth_email_service
        except Exception as e:
            logger.warning(f"Could not load auth email service for issues: {e}")
        
        logger.warning("⚠️ No email service available for issues")
        return None
    
    def get_user_from_token(self):
        """Extract user from JWT token"""
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
        
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, self.app.secret_key, algorithms=['HS256'])
            return self.User.query.get(payload.get('user_id'))
        except:
            return None
    
    def get_request_info(self):
        """Extract request information"""
        ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
        if ip_address:
            ip_address = ip_address.split(',')[0].strip()
        
        return {
            'ip_address': ip_address,
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'page_url': request.headers.get('Referer', '')
        }
    
    def _allowed_file(self, filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def _upload_to_cloudinary(self, file, issue_id):
        """Upload file to Cloudinary"""
        try:
            if not file or file.filename == '':
                return None
            
            if not self._allowed_file(file.filename):
                raise ValueError(f"File type not allowed. Allowed types: {', '.join(self.allowed_extensions)}")
            
            # Check file size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            
            if file_size > self.max_file_size:
                raise ValueError("File size exceeds 10MB limit")
            
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{issue_id}_{uuid.uuid4().hex[:8]}_{filename}"
            
            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(
                file,
                folder=self.cloudinary_folder,
                public_id=unique_filename,
                resource_type="auto",
                overwrite=False,
                transformation=[
                    {'quality': "auto:good"},
                    {'fetch_format': "auto"}
                ]
            )
            
            return {
                'url': upload_result['secure_url'],
                'public_id': upload_result['public_id'],
                'format': upload_result.get('format'),
                'width': upload_result.get('width'),
                'height': upload_result.get('height'),
                'bytes': upload_result.get('bytes'),
                'original_filename': filename
            }
            
        except Exception as e:
            logger.error(f"Cloudinary upload error: {e}")
            raise e
    
    def report_issue(self):
        """Handle issue report submission with enhanced processing"""
        try:
            # Handle form data (multipart/form-data for file uploads)
            data = {}
            data['name'] = request.form.get('name', '').strip()
            data['email'] = request.form.get('email', '').strip()
            data['issue_type'] = request.form.get('issue_type', 'bug_error')
            data['severity'] = request.form.get('severity', 'normal')
            data['issue_title'] = request.form.get('issue_title', '').strip()
            data['description'] = request.form.get('description', '').strip()
            data['browser_version'] = request.form.get('browser_version', '').strip()
            data['device_os'] = request.form.get('device_os', '').strip()
            data['page_url'] = request.form.get('page_url', '').strip()
            data['steps_to_reproduce'] = request.form.get('steps_to_reproduce', '').strip()
            data['expected_behavior'] = request.form.get('expected_behavior', '').strip()
            
            # Validate required fields
            required_fields = ['email', 'issue_title', 'description']
            for field in required_fields:
                if not data.get(field):
                    return jsonify({'error': f'{field.replace("_", " ").title()} is required'}), 400
            
            # Use email as name if name is not provided
            if not data['name']:
                data['name'] = data['email'].split('@')[0].replace('.', ' ').title()
            
            # Validate email format
            if not EMAIL_REGEX.match(data['email']):
                return jsonify({'error': 'Please provide a valid email address'}), 400
            
            # Rate limiting
            if not self._check_rate_limit(data['email']):
                return jsonify({'error': 'Too many issue reports. Please try again in 15 minutes.'}), 429
            
            user = self.get_user_from_token()
            request_info = self.get_request_info()
            
            # Generate unique issue ID
            issue_id = f"issue_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Handle file uploads
            uploaded_files = []
            files = request.files.getlist('screenshots')
            
            if files and len(files) > 0:
                for file in files[:5]:  # Limit to 5 files
                    if file and file.filename:
                        try:
                            file_info = self._upload_to_cloudinary(file, issue_id)
                            if file_info:
                                uploaded_files.append(file_info)
                        except Exception as e:
                            logger.warning(f"File upload failed: {e}")
                            continue
            
            # Get or create Technical Issues category
            tech_category = self.SupportCategory.query.filter_by(name='Technical Issues').first()
            if not tech_category:
                tech_category = self.SupportCategory.query.first()
            
            # Create issue report record
            issue_report = self.IssueReport(
                issue_id=issue_id,
                name=data['name'],
                email=data['email'],
                issue_type=data['issue_type'],
                severity=data['severity'],
                issue_title=data['issue_title'],
                description=data['description'],
                browser_version=data['browser_version'],
                device_os=data['device_os'],
                page_url_reported=data['page_url'],
                steps_to_reproduce=data['steps_to_reproduce'],
                expected_behavior=data['expected_behavior'],
                screenshots=uploaded_files,
                user_id=user.id if user else None,
                **request_info
            )
            
            self.db.session.add(issue_report)
            self.db.session.flush()
            
            # Create corresponding support ticket
            ticket_number = self._generate_ticket_number()
            
            # Map severity to priority
            priority_mapping = {
                'low': 'low',
                'normal': 'normal', 
                'high': 'high',
                'critical': 'urgent'
            }
            
            priority = priority_mapping.get(data['severity'], 'normal')
            
            # Create detailed description for ticket
            detailed_description = f"""
ISSUE REPORT DETAILS:

🐛 Issue Type: {data['issue_type'].replace('_', ' ').title()}
⚡ Severity: {data['severity'].title()}

📝 Description:
{data['description']}

🔄 Steps to Reproduce:
{data['steps_to_reproduce'] or 'Not provided'}

✅ Expected Behavior:
{data['expected_behavior'] or 'Not provided'}

🌐 Technical Details:
• Browser: {data['browser_version'] or 'Not provided'}
• Device/OS: {data['device_os'] or 'Not provided'}
• Page URL: {data['page_url'] or 'Not provided'}
• User Agent: {request_info['user_agent']}
• IP Address: {request_info['ip_address']}

📎 Screenshots: {len(uploaded_files)} file(s) attached
            """.strip()
            
            # Create support ticket
            ticket = self.SupportTicket(
                ticket_number=ticket_number,
                subject=f"[BUG REPORT] {data['issue_title']}",
                description=detailed_description,
                user_id=user.id if user else None,
                user_email=data['email'],
                user_name=data['name'],
                category_id=tech_category.id if tech_category else 1,
                ticket_type='bug_report',
                priority=priority,
                status='open',
                sla_deadline=self._calculate_sla_deadline(priority),
                **request_info
            )
            
            self.db.session.add(ticket)
            self.db.session.flush()
            
            # Link issue report to ticket
            issue_report.ticket_id = ticket.id
            
            # Add activity log
            activity = self.TicketActivity(
                ticket_id=ticket.id,
                action='created',
                description=f'Issue report submitted by {data["name"]} - {data["issue_type"]} ({data["severity"]} severity)',
                actor_type='user',
                actor_id=user.id if user else None,
                actor_name=data['name']
            )
            self.db.session.add(activity)
            
            self.db.session.commit()
            
            # Send notifications
            self._send_user_confirmation(issue_report, ticket, data)
            self._send_admin_notification(issue_report, ticket, data)
            
            logger.info(f"✅ Issue report {issue_id} submitted by {data['email']}")
            
            return jsonify({
                'success': True,
                'message': 'Issue report submitted successfully. Thank you for helping us improve CineBrain!',
                'issue_id': issue_id,
                'ticket_number': ticket_number,
                'files_uploaded': len(uploaded_files)
            }), 201
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"❌ Error processing issue report: {e}")
            return jsonify({'error': 'Failed to submit issue report. Please try again.'}), 500
    
    def _generate_ticket_number(self):
        """Generate unique ticket number"""
        import random
        import string
        
        date_str = datetime.now().strftime('%Y%m%d')
        random_str = ''.join(random.choices(string.digits, k=4))
        ticket_number = f"CB-{date_str}-{random_str}"
        
        # Ensure uniqueness
        while self.SupportTicket.query.filter_by(ticket_number=ticket_number).first():
            random_str = ''.join(random.choices(string.digits, k=4))
            ticket_number = f"CB-{date_str}-{random_str}"
        
        return ticket_number
    
    def _calculate_sla_deadline(self, priority_str):
        """Calculate SLA deadline based on priority string"""
        from datetime import timedelta
        
        now = datetime.utcnow()
        sla_hours = {
            'urgent': 4,
            'high': 24,
            'normal': 48,
            'low': 72
        }
        
        hours = sla_hours.get(priority_str, 48)
        return now + timedelta(hours=hours)
    
    def _check_rate_limit(self, email: str) -> bool:
        """Check rate limit for issue reports"""
        if not self.redis_client:
            return True
        
        try:
            key = f"issue_rate_limit:{email}"
            current = self.redis_client.get(key)
            
            if current is None:
                self.redis_client.setex(key, 900, 1)  # 15 minutes
                return True
            
            if int(current) >= 5:  # Max 5 reports per 15 minutes
                return False
            
            self.redis_client.incr(key)
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True
    
    def _send_user_confirmation(self, issue_report, ticket, data):
        """Send confirmation email to user"""
        try:
            if not self.email_service:
                logger.warning("Email service not available - cannot send user confirmation")
                return
            
            from auth.support_mail_templates import get_support_template
            
            html, text = get_support_template(
                'issue_reported',
                user_name=data['name'],
                issue_title=data['issue_title'],
                issue_type=data['issue_type'].replace('_', ' ').title(),
                severity=data['severity'],
                ticket_number=ticket.ticket_number
            )
            
            self.email_service.queue_email(
                to=data['email'],
                subject=f"Issue Report Received #{ticket.ticket_number} - CineBrain",
                html=html,
                text=text,
                priority='high',
                to_name=data['name']
            )
            
            logger.info(f"✅ Issue confirmation email queued for {data['email']}")
            
        except Exception as e:
            logger.error(f"❌ Error sending issue confirmation: {e}")
    
    def _send_admin_notification(self, issue_report, ticket, data):
        """Send admin notification email"""
        try:
            if not self.email_service:
                logger.warning("Email service not available - cannot send admin notification")
                return
            
            admin_email = os.environ.get('ADMIN_EMAIL', 'srinathnulidonda.dev@gmail.com')
            admin_link = f"{os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')}/admin/support/issue/{issue_report.issue_id}"
            
            from auth.support_mail_templates import get_support_template
            
            # Prepare screenshot info
            screenshot_info = ""
            if issue_report.screenshots:
                screenshot_info = f"""
                <p><strong>📸 Screenshots ({len(issue_report.screenshots)} files):</strong></p>
                <ul>
                """
                for i, screenshot in enumerate(issue_report.screenshots, 1):
                    screenshot_info += f"""
                    <li><a href="{screenshot['url']}" target="_blank">{screenshot['original_filename']}</a> 
                    ({screenshot.get('bytes', 0) // 1024}KB, {screenshot.get('format', 'unknown').upper()})</li>
                    """
                screenshot_info += "</ul>"
            
            html, text = get_support_template(
                'admin_notification',
                notification_type='issue_report',
                title=f"🐛 New Issue Report: {data['issue_title']}",
                message=f"""
                <p><strong>New issue report submitted:</strong></p>
                <ul>
                    <li><strong>Ticket:</strong> #{ticket.ticket_number}</li>
                    <li><strong>Issue ID:</strong> {issue_report.issue_id}</li>
                    <li><strong>From:</strong> {data['name']} ({data['email']})</li>
                    <li><strong>Type:</strong> {data['issue_type'].replace('_', ' ').title()}</li>
                    <li><strong>Severity:</strong> <span style="color: {'#ef4444' if data['severity'] == 'critical' else '#f59e0b' if data['severity'] == 'high' else '#3b82f6'}; font-weight: bold;">{data['severity'].upper()}</span></li>
                    <li><strong>Browser:</strong> {data.get('browser_version', 'Not provided')}</li>
                    <li><strong>Device/OS:</strong> {data.get('device_os', 'Not provided')}</li>
                    <li><strong>Page URL:</strong> {data.get('page_url', 'Not provided')}</li>
                </ul>
                
                <p><strong>📝 Description:</strong></p>
                <div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #ef4444; margin: 15px 0;">
                    {data['description']}
                </div>
                
                {f'''
                <p><strong>🔄 Steps to Reproduce:</strong></p>
                <div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #3b82f6; margin: 15px 0;">
                    {data['steps_to_reproduce']}
                </div>
                ''' if data.get('steps_to_reproduce') else ''}
                
                {f'''
                <p><strong>✅ Expected Behavior:</strong></p>
                <div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #10b981; margin: 15px 0;">
                    {data['expected_behavior']}
                </div>
                ''' if data.get('expected_behavior') else ''}
                
                {screenshot_info}
                
                <p style="margin-top: 20px;">
                    <a href="{admin_link}" style="background: #113CCF; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">
                        View in Admin Dashboard
                    </a>
                </p>
                """,
                ticket_number=ticket.ticket_number,
                user_email=data['email']
            )
            
            priority = 'urgent' if data['severity'] in ['critical', 'high'] else 'high'
            
            self.email_service.queue_email(
                to=admin_email,
                subject=f"🚨 New Issue Report: {data['issue_title']} - CineBrain Admin",
                html=html,
                text=text,
                priority=priority,
                to_name='CineBrain Admin'
            )
            
            logger.info(f"✅ Admin notification email queued for issue {issue_report.issue_id}")
            
        except Exception as e:
            logger.error(f"❌ Error sending admin notification: {e}")

def init_issue_service(app, db, models, services):
    """Initialize issue service"""
    return IssueReportService(app, db, models, services)