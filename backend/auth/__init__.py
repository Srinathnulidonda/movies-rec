# auth/__init__.py

from .routes import auth_bp
from .service import init_auth, require_auth, generate_reset_token, verify_reset_token, validate_password

__all__ = [
    'auth_bp',
    'init_auth', 
    'require_auth',
    'generate_reset_token',
    'verify_reset_token', 
    'validate_password'
]
__version__ = '2.0.0'