"""
CineBrain Details Error Handling
Centralized error management for the details module
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DetailsError(Exception):
    """Base exception for details module"""
    
    def __init__(self, message: str, code: str = "DETAILS_ERROR", 
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)
        
        # Log the error
        logger.error(f"DetailsError [{self.code}]: {self.message} - {self.details}")

class APIError(DetailsError):
    """API-related errors"""
    
    def __init__(self, service: str, message: str, status_code: Optional[int] = None):
        self.service = service
        self.status_code = status_code
        details = {
            'service': service,
            'status_code': status_code
        }
        super().__init__(
            f"{service} API error: {message}",
            "API_ERROR",
            details
        )

class ValidationError(DetailsError):
    """Input validation errors"""
    
    def __init__(self, field: str, value: Any, message: str):
        self.field = field
        self.value = value
        details = {
            'field': field,
            'value': value
        }
        super().__init__(
            f"Validation error for {field}: {message}",
            "VALIDATION_ERROR",
            details
        )

class CacheError(DetailsError):
    """Cache-related errors"""
    
    def __init__(self, operation: str, message: str):
        self.operation = operation
        details = {'operation': operation}
        super().__init__(
            f"Cache {operation} error: {message}",
            "CACHE_ERROR",
            details
        )

class ContentNotFoundError(DetailsError):
    """Content not found errors"""
    
    def __init__(self, identifier: str, identifier_type: str = "slug"):
        self.identifier = identifier
        self.identifier_type = identifier_type
        details = {
            'identifier': identifier,
            'type': identifier_type
        }
        super().__init__(
            f"Content not found: {identifier_type}={identifier}",
            "CONTENT_NOT_FOUND",
            details
        )

class PersonNotFoundError(DetailsError):
    """Person not found errors"""
    
    def __init__(self, identifier: str, identifier_type: str = "slug"):
        self.identifier = identifier
        self.identifier_type = identifier_type
        details = {
            'identifier': identifier,
            'type': identifier_type
        }
        super().__init__(
            f"Person not found: {identifier_type}={identifier}",
            "PERSON_NOT_FOUND",
            details
        )

class SlugError(DetailsError):
    """Slug generation/management errors"""
    
    def __init__(self, title: str, message: str):
        self.title = title
        details = {'title': title}
        super().__init__(
            f"Slug error for '{title}': {message}",
            "SLUG_ERROR",
            details
        )

def handle_api_error(func):
    """Decorator for handling API errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, DetailsError):
                raise e
            else:
                # Convert generic exceptions to DetailsError
                raise DetailsError(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    "UNEXPECTED_ERROR",
                    {'function': func.__name__, 'args': str(args), 'kwargs': str(kwargs)}
                )
    return wrapper

def log_performance_warning(func_name: str, duration: float, threshold: float = 5.0):
    """Log performance warnings for slow operations"""
    if duration > threshold:
        logger.warning(
            f"Performance warning: {func_name} took {duration:.2f}s (threshold: {threshold}s)"
        )

class ErrorReporter:
    """Centralized error reporting and metrics"""
    
    def __init__(self):
        self.error_counts = {}
    
    def report_error(self, error: DetailsError):
        """Report error for metrics and monitoring"""
        error_type = error.code
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log structured error data
        logger.error(
            f"Error reported: {error_type}",
            extra={
                'error_code': error.code,
                'error_message': error.message,
                'error_details': error.details,
                'error_count': self.error_counts[error_type]
            }
        )
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics"""
        return self.error_counts.copy()
    
    def reset_stats(self):
        """Reset error statistics"""
        self.error_counts.clear()

# Global error reporter instance
error_reporter = ErrorReporter()