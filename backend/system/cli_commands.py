# system/cli-commands.py

import os
import click
import time
import logging
import traceback
from datetime import datetime, timedelta
from sqlalchemy import or_, text, func
from werkzeug.security import generate_password_hash

logger = logging.getLogger(__name__)

class CineBrainCLI:
    """CineBrain CLI Commands Manager"""
    
    def __init__(self, app, db, models, services):
        self.app = app
        self.db = db
        self.models = models
        self.services = services
        self.register_commands()
    
    def register_commands(self):
        """Register all CLI commands with the Flask app"""
        
        # ===== CONTENT MANAGEMENT COMMANDS =====
        
        @self.app.cli.command('generate-slugs')
        def generate_slugs():
            """Generate slugs for all content and persons"""
            try:
                print("Starting CineBrain comprehensive slug generation...")
                
                details_service = self.services.get('details_service')
                if not details_service:
                    print("Error: CineBrain details service not available")
                    return
                
                stats = details_service.migrate_all_slugs(batch_size=50)
                
                print("CineBrain slug migration completed successfully!")
                print(f"Content updated: {stats['content_updated']}")
                print(f"Persons updated: {stats['persons_updated']}")
                print(f"Total processed: {stats['total_processed']}")
                print(f"Errors: {stats['errors']}")
                
                try:
                    Content = self.models['Content']
                    Person = self.models['Person']
                    
                    total_content = Content.query.count()
                    content_with_slugs = Content.query.filter(
                        Content.slug != None, Content.slug != ''
                    ).count()
                    
                    total_persons = Person.query.count()
                    persons_with_slugs = Person.query.filter(
                        Person.slug != None, Person.slug != ''
                    ).count()
                    
                    print(f"\nCineBrain Verification Results:")
                    print(f"Content: {content_with_slugs}/{total_content} ({round(content_with_slugs/total_content*100, 1)}% coverage)")
                    print(f"Persons: {persons_with_slugs}/{total_persons} ({round(persons_with_slugs/total_persons*100, 1)}% coverage)")
                    
                except Exception as e:
                    print(f"CineBrain error during verification: {e}")
                
            except Exception as e:
                print(f"Failed to generate CineBrain slugs: {e}")
                logger.error(f"CineBrain CLI slug generation error: {e}")

        @self.app.cli.command('populate-cast-crew')
        def populate_cast_crew_cli():
            """Populate cast and crew data for content"""
            try:
                print("Starting CineBrain cast/crew population...")
                
                details_service = self.services.get('details_service')
                if not details_service:
                    print("Error: CineBrain details service not available")
                    return
                
                Content = self.models['Content']
                ContentPerson = self.models['ContentPerson']
                
                content_items = Content.query.filter(
                    Content.tmdb_id.isnot(None),
                    ~Content.id.in_(
                        self.db.session.query(ContentPerson.content_id).distinct()
                    )
                ).all()
                
                print(f"Found {len(content_items)} CineBrain content items without cast/crew")
                
                processed = 0
                errors = 0
                
                for i, content in enumerate(content_items):
                    try:
                        print(f"Processing {i+1}/{len(content_items)}: {content.title}")
                        cast_crew = details_service._fetch_and_save_all_cast_crew(content)
                        processed += 1
                        print(f"  Added {len(cast_crew['cast'])} cast members and crew")
                    except Exception as e:
                        print(f"  Error: {e}")
                        errors += 1
                
                print(f"\nCineBrain cast/crew population completed!")
                print(f"Processed: {processed}")
                print(f"Errors: {errors}")
                
            except Exception as e:
                print(f"Failed to populate CineBrain cast/crew: {e}")
                logger.error(f"CineBrain CLI cast/crew population error: {e}")

        @self.app.cli.command('cinebrain-new-releases-refresh')
        def cinebrain_new_releases_refresh_cli():
            """Refresh new releases data"""
            try:
                print("Starting CineBrain new releases manual refresh...")
                
                new_releases_service = self.services.get('new_releases_service')
                if not new_releases_service:
                    print("Error: CineBrain new releases service not available")
                    return
                
                new_releases_service.refresh_new_releases()
                stats = new_releases_service.get_stats()
                
                print("CineBrain new releases refresh completed!")
                print(f"Total items: {stats.get('total_items', 0)}")
                print(f"Priority items: {stats.get('priority_items', 0)}")
                print(f"Movies: {stats.get('movies', 0)}")
                print(f"TV Shows: {stats.get('tv_shows', 0)}")
                print(f"Anime: {stats.get('anime', 0)}")
                
            except Exception as e:
                print(f"Failed to refresh CineBrain new releases: {e}")
                logger.error(f"CineBrain CLI new releases refresh error: {e}")

        # ===== BREVO EMAIL SERVICE COMMANDS =====
        
        @self.app.cli.command('test-brevo')
        @click.option('--email', default=None, help='Email address to send test to')
        def test_brevo_connection(email):
            """Test Brevo email service connection and configuration"""
            try:
                print("🔧 Testing Brevo Email Service...")
                print("-" * 50)
                
                # Check environment variables
                api_key = os.environ.get('BREVO_API_KEY')
                sender_email = os.environ.get('BREVO_SENDER_EMAIL')
                sender_name = os.environ.get('BREVO_SENDER_NAME')
                
                print("📋 Configuration Check:")
                print(f"   API Key: {'✅ Set' if api_key else '❌ Missing'}")
                print(f"   Sender Email: {sender_email or '❌ Missing'}")
                print(f"   Sender Name: {sender_name or '❌ Missing'}")
                
                if not all([api_key, sender_email]):
                    print("\n❌ Missing required Brevo configuration!")
                    return
                
                # Test email service
                email_service = self.services.get('email_service')
                if email_service:
                    print(f"\n📧 Email Service Status: {'✅ Active' if email_service.email_enabled else '❌ Disabled'}")
                    
                    # Send test email if address provided
                    if email and email_service.email_enabled:
                        print(f"\n📮 Sending test email to {email}...")
                        
                        from auth.admin_mail_templates import get_admin_template
                        html, text = get_admin_template(
                            'admin_notification',
                            subject='CineBrain Email Test',
                            content='This is a test email from CineBrain admin CLI to verify Brevo integration.',
                            is_urgent=False
                        )
                        
                        try:
                            email_service.queue_email(
                                to=email,
                                subject='CineBrain - Brevo Test Email',
                                html=html,
                                text=text,
                                priority='high',
                                to_name='Test Recipient'
                            )
                            
                            # Process queue immediately for testing
                            email_service.process_email_queue()
                            print("✅ Test email sent successfully!")
                            print("   Check your inbox (and spam folder)")
                            
                        except Exception as e:
                            print(f"❌ Failed to send test email: {e}")
                    
                    # Show email queue stats
                    if hasattr(email_service, 'email_queue'):
                        print(f"\n📊 Email Queue Stats:")
                        print(f"   Pending: {len(email_service.email_queue)}")
                        print(f"   Processed Today: {email_service.emails_sent_today if hasattr(email_service, 'emails_sent_today') else 'N/A'}")
                else:
                    print("\n❌ Email service not initialized!")
                    
            except Exception as e:
                print(f"❌ Error testing Brevo: {e}")
                logger.error(f"Brevo test error: {e}")

        @self.app.cli.command('process-email-queue')
        def process_email_queue_cli():
            """Manually process pending emails in Brevo queue"""
            try:
                print("📧 Processing Email Queue...")
                
                email_service = self.services.get('email_service')
                if not email_service:
                    print("❌ Email service not available!")
                    return
                
                if hasattr(email_service, 'process_email_queue'):
                    processed = email_service.process_email_queue()
                    print(f"✅ Processed {processed} emails")
                else:
                    print("❌ Email queue processing not available")
                    
            except Exception as e:
                print(f"❌ Error processing email queue: {e}")
                logger.error(f"Email queue processing error: {e}")

        # ===== ADMIN MANAGEMENT COMMANDS =====
        
        @self.app.cli.command('create-admin')
        @click.option('--username', prompt=True, help='Admin username')
        @click.option('--email', prompt=True, help='Admin email address')
        @click.option('--password', prompt=True, hide_input=True, confirmation_prompt=True, help='Admin password')
        def create_admin_user(username, email, password):
            """Create a new admin user"""
            try:
                print(f"\n🔐 Creating admin user: {username}")
                
                User = self.models['User']
                
                # Check if user already exists
                existing = User.query.filter(
                    or_(User.username == username, User.email == email)
                ).first()
                
                if existing:
                    print(f"❌ User already exists with username '{username}' or email '{email}'")
                    return
                
                # Create admin user
                admin = User(
                    username=username,
                    email=email,
                    password_hash=generate_password_hash(password),
                    is_admin=True,
                    created_at=datetime.utcnow()
                )
                
                self.db.session.add(admin)
                self.db.session.commit()
                
                print(f"✅ Admin user created successfully!")
                print(f"   Username: {username}")
                print(f"   Email: {email}")
                print(f"   Admin ID: {admin.id}")
                
                # Send welcome email
                email_service = self.services.get('email_service')
                if email_service:
                    try:
                        from auth.admin_mail_templates import get_admin_template
                        html, text = get_admin_template(
                            'admin_notification',
                            subject='Welcome to CineBrain Admin',
                            content=f'Admin account created for {username}. You now have full admin access to CineBrain.',
                            is_urgent=False
                        )
                        
                        email_service.queue_email(
                            to=email,
                            subject='Welcome to CineBrain Admin',
                            html=html,
                            text=text,
                            priority='high',
                            to_name=username
                        )
                        print("📧 Welcome email queued")
                    except:
                        pass
                        
            except Exception as e:
                self.db.session.rollback()
                print(f"❌ Error creating admin: {e}")
                logger.error(f"Admin creation error: {e}")

        @self.app.cli.command('list-admins')
        def list_admin_users():
            """List all admin users"""
            try:
                print("\n👥 CineBrain Admin Users:")
                print("-" * 60)
                
                User = self.models['User']
                admins = User.query.filter_by(is_admin=True).order_by(User.created_at).all()
                
                if not admins:
                    print("❌ No admin users found!")
                    return
                
                for admin in admins:
                    print(f"\n🔐 Admin ID: {admin.id}")
                    print(f"   Username: {admin.username}")
                    print(f"   Email: {admin.email}")
                    print(f"   Created: {admin.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
                    print(f"   Last Active: {admin.last_active.strftime('%Y-%m-%d %H:%M UTC') if admin.last_active else 'Never'}")
                    
                print(f"\n📊 Total Admins: {len(admins)}")
                
            except Exception as e:
                print(f"❌ Error listing admins: {e}")
                logger.error(f"Admin list error: {e}")

        @self.app.cli.command('toggle-admin')
        @click.argument('username')
        def toggle_admin_status(username):
            """Toggle admin status for a user"""
            try:
                User = self.models['User']
                user = User.query.filter_by(username=username).first()
                
                if not user:
                    print(f"❌ User '{username}' not found!")
                    return
                
                # Toggle admin status
                user.is_admin = not user.is_admin
                self.db.session.commit()
                
                status = "granted" if user.is_admin else "revoked"
                print(f"✅ Admin access {status} for user '{username}'")
                
                # Send notification
                admin_notification_service = self.services.get('admin_notification_service')
                if admin_notification_service:
                    try:
                        admin_notification_service.create_notification(
                            'user_activity',
                            f'Admin Status Changed: {username}',
                            f'Admin access {status} for user {username} (ID: {user.id})',
                            admin_id=user.id if user.is_admin else None
                        )
                    except:
                        pass
                        
            except Exception as e:
                self.db.session.rollback()
                print(f"❌ Error toggling admin status: {e}")
                logger.error(f"Admin toggle error: {e}")

        # ===== SUPPORT SYSTEM COMMANDS =====
        
        @self.app.cli.command('support-stats')
        def show_support_stats():
            """Show support system statistics"""
            try:
                print("\n📊 CineBrain Support System Statistics")
                print("=" * 60)
                
                SupportTicket = self.models['SupportTicket']
                ContactMessage = self.models['ContactMessage']
                IssueReport = self.models['IssueReport']
                SupportCategory = self.models['SupportCategory']
                
                # Ticket stats
                total_tickets = SupportTicket.query.count()
                open_tickets = SupportTicket.query.filter_by(status='open').count()
                urgent_tickets = SupportTicket.query.filter_by(priority='urgent', status='open').count()
                sla_breached = SupportTicket.query.filter_by(sla_breached=True, status='open').count()
                
                print(f"\n🎫 Support Tickets:")
                print(f"   Total: {total_tickets}")
                print(f"   Open: {open_tickets}")
                print(f"   Urgent: {urgent_tickets}")
                print(f"   SLA Breached: {sla_breached}")
                
                # Contact messages
                total_contacts = ContactMessage.query.count()
                unread_contacts = ContactMessage.query.filter_by(is_read=False).count()
                
                print(f"\n💌 Contact Messages:")
                print(f"   Total: {total_contacts}")
                print(f"   Unread: {unread_contacts}")
                
                # Issue reports
                total_issues = IssueReport.query.count()
                unresolved_issues = IssueReport.query.filter_by(is_resolved=False).count()
                critical_issues = IssueReport.query.filter_by(severity='critical', is_resolved=False).count()
                
                print(f"\n🐛 Issue Reports:")
                print(f"   Total: {total_issues}")
                print(f"   Unresolved: {unresolved_issues}")
                print(f"   Critical: {critical_issues}")
                
                # Recent activity
                print(f"\n📈 Recent Activity (Last 24 Hours):")
                yesterday = datetime.utcnow() - timedelta(days=1)
                
                recent_tickets = SupportTicket.query.filter(
                    SupportTicket.created_at >= yesterday
                ).count()
                recent_contacts = ContactMessage.query.filter(
                    ContactMessage.created_at >= yesterday
                ).count()
                recent_issues = IssueReport.query.filter(
                    IssueReport.created_at >= yesterday
                ).count()
                
                print(f"   New Tickets: {recent_tickets}")
                print(f"   New Contacts: {recent_contacts}")
                print(f"   New Issues: {recent_issues}")
                
                # Categories
                print(f"\n📁 Support Categories:")
                categories = SupportCategory.query.filter_by(is_active=True).all()
                for cat in categories:
                    ticket_count = cat.tickets.count()
                    print(f"   {cat.icon} {cat.name}: {ticket_count} tickets")
                    
            except Exception as e:
                print(f"❌ Error getting support stats: {e}")
                logger.error(f"Support stats error: {e}")

        @self.app.cli.command('test-admin-notification')
        @click.option('--type', default='test', help='Notification type')
        @click.option('--urgent', is_flag=True, help='Mark as urgent')
        def test_admin_notification(type, urgent):
            """Send a test notification to all admins"""
            try:
                print(f"\n📢 Sending test notification to admins...")
                
                admin_notification_service = self.services.get('admin_notification_service')
                if not admin_notification_service:
                    print("❌ Admin notification service not available!")
                    return
                
                # Create test notification
                notification = admin_notification_service.create_notification(
                    notification_type=type,
                    title='Test Notification from CLI',
                    message=f'This is a test {type} notification sent from CineBrain CLI at {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}',
                    is_urgent=urgent,
                    action_required=urgent,
                    action_url='/admin'
                )
                
                if notification:
                    print("✅ Notification created successfully!")
                    print("   Check admin emails and dashboard")
                else:
                    print("❌ Failed to create notification")
                    
            except Exception as e:
                print(f"❌ Error sending test notification: {e}")
                logger.error(f"Test notification error: {e}")

        @self.app.cli.command('cleanup-notifications')
        @click.option('--days', default=30, help='Delete notifications older than X days')
        def cleanup_old_notifications(days):
            """Clean up old admin notifications"""
            try:
                print(f"\n🧹 Cleaning up notifications older than {days} days...")
                
                AdminNotification = self.models['AdminNotification']
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Count notifications to delete
                old_notifications = AdminNotification.query.filter(
                    AdminNotification.created_at < cutoff_date,
                    AdminNotification.is_read == True
                ).count()
                
                if old_notifications == 0:
                    print("✅ No old notifications to clean up")
                    return
                
                # Confirm deletion
                if click.confirm(f"Delete {old_notifications} read notifications older than {days} days?"):
                    AdminNotification.query.filter(
                        AdminNotification.created_at < cutoff_date,
                        AdminNotification.is_read == True
                    ).delete()
                    
                    self.db.session.commit()
                    print(f"✅ Deleted {old_notifications} old notifications")
                else:
                    print("❌ Cleanup cancelled")
                    
            except Exception as e:
                self.db.session.rollback()
                print(f"❌ Error cleaning up notifications: {e}")
                logger.error(f"Notification cleanup error: {e}")

        @self.app.cli.command('reset-support-categories')
        def reset_support_categories():
            """Reset support categories to defaults"""
            try:
                print("\n🔧 Resetting support categories...")
                
                SupportCategory = self.models['SupportCategory']
                
                # Default categories
                default_categories = [
                    {'name': 'Account & Login', 'description': 'Issues with account creation, login, password reset', 'icon': '👤', 'sort_order': 1},
                    {'name': 'Technical Issues', 'description': 'App crashes, loading issues, performance problems', 'icon': '🔧', 'sort_order': 2},
                    {'name': 'Features & Functions', 'description': 'How to use features, feature requests', 'icon': '⚡', 'sort_order': 3},
                    {'name': 'Content & Recommendations', 'description': 'Issues with movies, shows, recommendations', 'icon': '🎬', 'sort_order': 4},
                    {'name': 'Billing & Subscription', 'description': 'Payment issues, subscription questions', 'icon': '💳', 'sort_order': 5},
                    {'name': 'General Support', 'description': 'Other questions and general inquiries', 'icon': '❓', 'sort_order': 6}
                ]
                
                # Update or create categories
                for cat_data in default_categories:
                    category = SupportCategory.query.filter_by(name=cat_data['name']).first()
                    if category:
                        category.description = cat_data['description']
                        category.icon = cat_data['icon']
                        category.sort_order = cat_data['sort_order']
                        category.is_active = True
                        print(f"   ✅ Updated: {cat_data['name']}")
                    else:
                        category = SupportCategory(**cat_data)
                        self.db.session.add(category)
                        print(f"   ✅ Created: {cat_data['name']}")
                
                self.db.session.commit()
                print("\n✅ Support categories reset successfully!")
                
            except Exception as e:
                self.db.session.rollback()
                print(f"❌ Error resetting categories: {e}")
                logger.error(f"Category reset error: {e}")

        @self.app.cli.command('check-sla-breaches')
        def check_sla_breaches():
            """Check and report SLA breaches"""
            try:
                print("\n⏰ Checking SLA Breaches...")
                print("-" * 50)
                
                SupportTicket = self.models['SupportTicket']
                
                # Find breached tickets
                breached_tickets = SupportTicket.query.filter(
                    SupportTicket.sla_deadline < datetime.utcnow(),
                    SupportTicket.sla_breached == False,
                    SupportTicket.status.in_(['open', 'in_progress'])
                ).all()
                
                if not breached_tickets:
                    print("✅ No SLA breaches found!")
                    return
                
                print(f"❌ Found {len(breached_tickets)} SLA breaches:")
                
                for ticket in breached_tickets:
                    # Mark as breached
                    ticket.sla_breached = True
                    
                    print(f"\n🎫 Ticket #{ticket.ticket_number}")
                    print(f"   Subject: {ticket.subject}")
                    print(f"   Priority: {ticket.priority}")
                    print(f"   Created: {ticket.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
                    print(f"   Deadline: {ticket.sla_deadline.strftime('%Y-%m-%d %H:%M UTC')}")
                    print(f"   Overdue: {(datetime.utcnow() - ticket.sla_deadline).total_seconds() / 3600:.1f} hours")
                    
                    # Send notification
                    admin_notification_service = self.services.get('admin_notification_service')
                    if admin_notification_service:
                        try:
                            admin_notification_service.notify_sla_breach(ticket)
                        except:
                            pass
                
                self.db.session.commit()
                print(f"\n✅ Updated {len(breached_tickets)} tickets and sent notifications")
                
            except Exception as e:
                self.db.session.rollback()
                print(f"❌ Error checking SLA breaches: {e}")
                logger.error(f"SLA check error: {e}")

        @self.app.cli.command('send-daily-summary')
        def send_daily_summary():
            """Send daily summary email to admins"""
            try:
                print("\n📊 Sending daily summary to admins...")
                
                SupportTicket = self.models['SupportTicket']
                Content = self.models['Content']
                User = self.models['User']
                
                # Gather statistics
                today = datetime.utcnow().date()
                yesterday = today - timedelta(days=1)
                
                # Today's tickets
                today_tickets = SupportTicket.query.filter(
                    func.date(SupportTicket.created_at) == today
                ).count()
                
                # Yesterday's comparison
                yesterday_tickets = SupportTicket.query.filter(
                    func.date(SupportTicket.created_at) == yesterday
                ).count()
                
                # Open tickets by priority
                urgent_open = SupportTicket.query.filter_by(priority='urgent', status='open').count()
                high_open = SupportTicket.query.filter_by(priority='high', status='open').count()
                
                # Content stats
                total_content = Content.query.count()
                new_content = Content.query.filter(
                    func.date(Content.created_at) == today
                ).count()
                
                # User stats
                total_users = User.query.count()
                active_today = User.query.filter(
                    func.date(User.last_active) == today
                ).count()
                
                # Create summary message
                summary = f"""
📊 DAILY SUMMARY - {today.strftime('%B %d, %Y')}

SUPPORT METRICS:
• New Tickets Today: {today_tickets} {'↑' if today_tickets > yesterday_tickets else '↓' if today_tickets < yesterday_tickets else '→'} (Yesterday: {yesterday_tickets})
• Urgent Open: {urgent_open}
• High Priority Open: {high_open}

PLATFORM STATS:
• Total Content: {total_content:,} items
• New Content Today: {new_content}
• Total Users: {total_users:,}
• Active Today: {active_today}

SYSTEM HEALTH: {'🟢 All Systems Operational' if urgent_open < 5 else '🟡 Attention Required' if urgent_open < 10 else '🔴 Critical'}
"""
                
                # Send to admins
                admin_notification_service = self.services.get('admin_notification_service')
                if admin_notification_service:
                    notification = admin_notification_service.create_notification(
                        notification_type='daily_summary',
                        title='Daily Platform Summary',
                        message=summary,
                        is_urgent=False,
                        action_url='/admin/dashboard'
                    )
                    
                    if notification:
                        print("✅ Daily summary sent to all admins!")
                        print(summary)
                    else:
                        print("❌ Failed to send daily summary")
                else:
                    print("❌ Admin notification service not available")
                    
            except Exception as e:
                print(f"❌ Error sending daily summary: {e}")
                logger.error(f"Daily summary error: {e}")

def init_cli_commands(app, db, models, services):
    """Initialize CLI commands with the app"""
    try:
        cli_manager = CineBrainCLI(app, db, models, services)
        return cli_manager
    except Exception as e:
        logger.error(f"Failed to initialize CLI commands: {e}")
        return None