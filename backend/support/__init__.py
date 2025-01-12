# support/__init__.py

from .routes import support_bp, init_support_routes
from .tickets import TicketService, TicketStatus, TicketPriority, TicketType
from .contact import ContactService
from .report_issues import IssueReportService

def init_support(app, db, models, services):
    """Initialize support system with proper email service"""
    
    # Ensure email service is available
    if 'email_service' not in services:
        try:
            from auth.service import email_service as auth_email_service
            services['email_service'] = auth_email_service
            app.logger.info("✅ Email service added to support services")
        except Exception as e:
            app.logger.warning(f"Could not import email service: {e}")
    
    # Initialize routes with existing models
    init_support_routes(app, db, models, services)
    
    return models

__all__ = [
    'support_bp',
    'init_support',
    'TicketService',
    'TicketStatus', 
    'TicketPriority',
    'TicketType',
    'ContactService',
    'IssueReportService'
]
__version__ = '2.0.0'