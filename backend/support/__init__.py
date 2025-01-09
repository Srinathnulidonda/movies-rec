# support/__init__.py

from .routes import support_bp, init_support_routes
from .tickets import TicketService, TicketStatus, TicketPriority, TicketType
from .contact import ContactService
from .report_issues import IssueReportService

def init_support(app, db, models, services):
    """Initialize support system"""
    
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