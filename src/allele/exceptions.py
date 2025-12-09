"""
Exception classes for Allele SDK.

Author: Bravetto AI Systems
Version: 1.0.0
"""

from typing import Optional, Dict, Any


class AbeNLPError(Exception):
    """Base exception for all Allele errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize AbeNLP exception.
        
        Args:
            message: Error message
            error_code: Optional error code for programmatic handling
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return string representation of error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class GenomeError(AbeNLPError):
    """Exception raised for genome-related errors."""
    pass


class EvolutionError(AbeNLPError):
    """Exception raised for evolution process errors."""
    pass


class AgentError(AbeNLPError):
    """Exception raised for agent-related errors."""
    pass


class ValidationError(AbeNLPError):
    """Exception raised for validation errors."""
    pass


class ConfigurationError(AbeNLPError):
    """Exception raised for configuration errors."""
    pass


class APIError(AbeNLPError):
    """Exception raised for API communication errors."""
    pass


class TimeoutError(AbeNLPError):
    """Exception raised when operations timeout."""
    pass

