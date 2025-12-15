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

"""LLM-specific exceptions with detailed error handling."""

from typing import Any, Dict, List, Optional

from .exceptions import AbeNLPError


class LLMError(AbeNLPError):
    """Base exception for LLM-related errors.

    Attributes:
        provider: The LLM provider where the error occurred (e.g., 'openai', 'anthropic')
        model: The specific model being used when the error occurred
        details: Additional error details like request_id, attempt count, etc.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.details = details or {}


class LLMInitializationError(LLMError):
    """Raised when LLM client fails to initialize.

    This includes API key validation failures, network connectivity issues,
    and provider-specific initialization errors.
    """
    pass


class LLMGenerationError(LLMError):
    """Raised when LLM generation fails after retry attempts.

    This includes content generation failures, invalid parameters,
    and provider-specific generation errors.
    """
    pass


class LLMRateLimitError(LLMError):
    """Raised when API rate limits are exceeded.

    Includes information about when to retry and current limit status.
    """
    pass


class LLMAuthenticationError(LLMError):
    """Raised when API authentication fails.

    This could be due to invalid API keys, expired tokens,
    or insufficient permissions.
    """
    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM requests timeout.

    Includes timeout duration and any partial responses if available.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None
    ):
        super().__init__(message, provider, model, details)
        self.timeout_seconds = timeout_seconds


class LLMQuotaExceededError(LLMError):
    """Raised when API quota/billing limits are exceeded.

    Includes information about quota reset times and current usage.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        reset_time: Optional[str] = None
    ):
        super().__init__(message, provider, model, details)
        self.reset_time = reset_time


class LLMContentFilterError(LLMError):
    """Raised when content violates provider's safety/content filters.

    Includes information about filtered content and suggested alternatives.
    """
    pass


class LLMModelNotAvailableError(LLMError):
    """Raised when requested model is not available.

    Includes list of available models and suggested alternatives.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        available_models: Optional[List[str]] = None
    ):
        super().__init__(message, provider, model, details)
        self.available_models = available_models or []


class LLMConfigurationError(LLMError):
    """Raised when LLM configuration is invalid.

    This includes invalid parameters, unsupported combinations,
    and configuration conflicts.
    """
    pass
