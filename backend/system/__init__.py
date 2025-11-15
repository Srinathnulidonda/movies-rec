# system/init.py

"""
CineBrain System Module
Advanced system monitoring, health checks, admin monitoring, and performance management
"""

from .routes import system_bp
from .system import SystemService
from .admin_monitor import AdminMonitoringService

__all__ = ['system_bp', 'SystemService', 'AdminMonitoringService']
__version__ = '2.0.0'