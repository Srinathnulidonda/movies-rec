# admin/__init__.py

from .routes import admin_bp, init_admin_routes
from .service import AdminNotificationService, AdminEmailService
from .dashboard import AdminDashboard
from .telegram import TelegramAdminService, TelegramService

def init_admin(app, db, models, services):
    """Initialize admin system"""
    
    # Initialize routes with existing models
    init_admin_routes(app, db, models, services)
    
    return models

__all__ = [
    'admin_bp',
    'init_admin',
    'AdminNotificationService',
    'AdminEmailService',
    'AdminDashboard',
    'TelegramAdminService',
    'TelegramService'
]
__version__ = '3.0.0'