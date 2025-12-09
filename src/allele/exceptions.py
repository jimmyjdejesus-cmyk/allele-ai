# Copyright (C) 2025 Bravetto AI Systems & Jimmy De Jesus
#
# This file is part of Allele.
#
# Allele is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Allele is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Allele.  If not, see <https://www.gnu.org/licenses/>.
#
# =============================================================================
# COMMERCIAL LICENSE:
# If you wish to use this software in a proprietary/closed-source application
# without releasing your source code, you must purchase a Commercial License
# from: https://gumroad.com/l/[YOUR_LINK]
# =============================================================================

"""Exception classes for Allele SDK.

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

