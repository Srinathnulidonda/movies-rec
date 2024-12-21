# system/init.py

"""
CineBrain System Module
System monitoring, health checks, and performance management
"""

from .routes import system_bp

__all__ = ['system_bp']
__version__ = '1.0.0'