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

"""Abstract LLM client with comprehensive error handling and logging."""

import asyncio
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import structlog

from .llm_exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMGenerationError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = structlog.get_logger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None  # NEW: Custom headers for authentication
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff_multiplier: float = 2.0
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 10000

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.provider or not isinstance(self.provider, str):
            raise ValueError("provider must be a non-empty string")
        if not self.model or not isinstance(self.model, str):
            raise ValueError("model must be a non-empty string")
        # Allow empty API key for local providers like ollama
        if self.api_key is None:
            self.api_key = ""
            
        if not isinstance(self.api_key, str):
            raise ValueError("api_key must be a string")
        if self.api_key == "" and self.provider not in ['ollama']:
            raise ValueError("api_key must be a non-empty string for cloud providers")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.retry_delay <= 0:
            raise ValueError("retry_delay must be positive")
        if self.rate_limit_requests_per_minute <= 0:
            raise ValueError("rate_limit_requests_per_minute must be positive")
        if self.rate_limit_tokens_per_minute <= 0:
            raise ValueError("rate_limit_tokens_per_minute must be positive")
        return None
        return None

@dataclass
class LLMUsageMetrics:
    """Track LLM usage for monitoring and billing."""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_latency_ms: float = 0.0
    error_rate: float = 0.0
    rate_limit_hits: int = 0
    last_request_time: Optional[float] = None
    uptime_seconds: float = 0.0

class RateLimiter:
    """Simple token bucket rate limiter with sliding window."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_timestamps: List[float] = []
        self.token_usage: List[float] = []
        self.lock = asyncio.Lock()

    async def wait_if_needed(self, tokens_used: int = 1) -> float:
        """Wait if rate limit would be exceeded. Returns wait time in seconds."""
        async with self.lock:
            current_time = time.time()

            # Clean old entries (sliding window of 60 seconds)
            cutoff_time = current_time - 60
            self.request_timestamps = [t for t in self.request_timestamps if t > cutoff_time]
            self.token_usage = self.token_usage[len([t for t in self.request_timestamps if t <= cutoff_time]):]

            # Check limits
            request_count = len(self.request_timestamps)
            total_tokens = sum(self.token_usage)

            if request_count >= self.requests_per_minute:
                # Rate limited by requests
                oldest_request = min(self.request_timestamps)
                wait_time = 60 - (current_time - oldest_request)
            elif total_tokens + tokens_used > self.tokens_per_minute:
                # Rate limited by tokens
                # Estimate wait time (simplified)
                excess_tokens = total_tokens + tokens_used - self.tokens_per_minute
                wait_time = 60 * (excess_tokens / self.tokens_per_minute)
            else:
                wait_time = 0

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Record usage
            self.request_timestamps.append(current_time)
            self.token_usage.append(tokens_used)

            # Maintain list sizes
            max_entries = max(2 * self.requests_per_minute, 100)
            if len(self.request_timestamps) > max_entries:
                excess = len(self.request_timestamps) - max_entries
                self.request_timestamps = self.request_timestamps[excess:]
                self.token_usage = self.token_usage[excess:]

            return wait_time

class LLMClient(ABC):
    """Abstract base class for LLM providers with comprehensive error handling."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.metrics = LLMUsageMetrics()
        self.rate_limiter = RateLimiter(
            config.rate_limit_requests_per_minute,
            config.rate_limit_tokens_per_minute
        )
        self.logger = logger.bind(
            provider=config.provider,
            model=config.model,
            client_id=id(self)
        )
        self._client = None
        self._initialized = False
        self._start_time = time.time()

    @property
    def initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the LLM client."""
        pass

    @abstractmethod
    def chat_completion(
        self, messages: List[Dict[str, str]], stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion (async generator). Implementations should be async generators."""
        ...

    @abstractmethod
    async def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for given token usage."""
        pass

    @abstractmethod
    async def get_available_models(self) -> List[str]:
        """Get list of available models from provider."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up client resources."""
        pass

    @asynccontextmanager
    async def session(self) -> AsyncGenerator["LLMClient", None]:
        """Context manager for LLM operations with automatic cleanup."""
        try:
            if not self.initialized:
                await self.initialize()
            yield self
        finally:
            await self.close()

    async def _retry_with_exponential_backoff(
        self,
        operation: Callable[..., Any],
        operation_name: str,
        tokens_used: int = 1,
        *args: Any,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Retry operation with exponential backoff and circuit breaker pattern."""
        last_exception = None
        total_wait_time: float = 0.0

        for attempt in range(self.config.max_retries):
            try:
                # Apply rate limiting
                wait_time = await self.rate_limiter.wait_if_needed(tokens_used)
                if wait_time > 0:
                    self.metrics.rate_limit_hits += 1
                    self.logger.debug(
                        f"Rate limited, waited {wait_time:.2f}s",
                        attempt=attempt,
                        wait_time=wait_time
                    )
                    total_wait_time += wait_time

                start_time = time.time()
                result = operation(*args, **kwargs)

                # Handle async generator specially
                if asyncio.iscoroutine(result):
                    result = await result

                # Update metrics
                duration_ms = (time.time() - start_time) * 1000
                self.metrics.total_requests += 1
                self.metrics.last_request_time = time.time()

                # Update average latency
                prev_avg = self.metrics.average_latency_ms
                self.metrics.average_latency_ms = (
                    prev_avg * (self.metrics.total_requests - 1) + duration_ms
                ) / self.metrics.total_requests

                self.logger.info(
                    f"{operation_name} successful",
                    attempt=attempt,
                    duration_ms=round(duration_ms, 2),
                    tokens_used=tokens_used,
                    total_wait_time=round(total_wait_time, 2)
                )

                # Yield results for async generators
                if hasattr(result, '__aiter__'):
                    async for item in result:
                        yield item
                else:
                    yield result
                return

            except Exception as e:
                last_exception = e
                self.metrics.error_rate = (
                    (self.metrics.error_rate * (self.metrics.total_requests or 1) + 1) /
                    ((self.metrics.total_requests or 1) + 1)
                )

                # Classify and handle different error types
                error_type, should_retry, delay = self._classify_error(e, attempt)

                self.logger.warning(
                    f"{operation_name} failed",
                    attempt=attempt,
                    error=str(e),
                    error_type=error_type,
                    will_retry=should_retry,
                    delay=round(delay, 2),
                    max_retries=self.config.max_retries
                )

                if not should_retry or attempt >= self.config.max_retries - 1:
                    # Final failure - raise appropriate exception
                    raise self._create_final_exception(
                        last_exception, attempt, total_wait_time, operation_name, error_type
                    ) from last_exception
                else:
                    await asyncio.sleep(delay)
                    total_wait_time += delay

    def _classify_error(self, error: Exception, attempt: int) -> tuple:
        """Classify error type and determine retry strategy."""
        error_str = str(error).lower()

        if "rate limit" in error_str or "rate_limit" in error_str:
            delay = self.config.retry_delay * (2 ** attempt) * 3  # Longer delay for rate limits
            return "rate_limit", True, delay
        elif any(keyword in error_str for keyword in ["auth", "unauthorized", "invalid api key"]):
            return "authentication", False, 0  # Don't retry auth errors
        elif "timeout" in error_str:
            delay = self.config.retry_delay * (2 ** attempt)
            return "timeout", True, delay
        elif any(keyword in error_str for keyword in ["quota", "billing", "credit"]):
            return "quota", False, 0  # Don't retry quota errors
        elif "content" in error_str and "filter" in error_str:
            return "content_filter", False, 0  # Don't retry content filter errors
        elif "model" in error_str and ("not found" in error_str or "not available" in error_str):
            return "model_not_found", False, 0  # Don't retry model errors
        else:
            delay = self.config.retry_delay * (self.config.retry_backoff_multiplier ** attempt)
            return "general", True, min(delay, 30)  # Cap max delay at 30 seconds

    def _create_final_exception(
        self,
        original_error: Exception,
        attempt: int,
        total_wait_time: float,
        operation_name: str,
        error_type: str
    ) -> LLMError:
        """Create appropriate final exception based on error type."""
        base_message = f"{operation_name} failed after {attempt + 1} attempts"

        details = {
            "last_error": str(original_error),
            "attempts": attempt + 1,
            "total_wait_time": round(total_wait_time, 2),
            "error_type": error_type,
            "provider": self.config.provider,
            "model": self.config.model
        }

        # Map error types to specific exceptions
        if error_type == "authentication":
            return LLMAuthenticationError(base_message, self.config.provider, self.config.model, details)
        elif error_type == "rate_limit":
            return LLMRateLimitError(base_message, self.config.provider, self.config.model, details)
        elif error_type == "timeout":
            return LLMTimeoutError(
                base_message,
                self.config.provider,
                self.config.model,
                details,
                timeout_seconds=self.config.timeout
            )
        elif error_type == "quota":
            return LLMQuotaExceededError(base_message, self.config.provider, self.config.model, details)
        else:
            return LLMGenerationError(base_message, self.config.provider, self.config.model, details)

    def get_uptime(self) -> float:
        """Get client uptime in seconds."""
        return time.time() - self._start_time

    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        uptime = self.get_uptime()

        try:
            available_models = await self.get_available_models() if self.initialized else []
        except Exception:
            available_models = []

        return {
            "initialized": self.initialized,
            "provider": self.config.provider,
            "model": self.config.model,
            "uptime_seconds": round(uptime, 2),
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "total_tokens": self.metrics.total_tokens,
                "total_cost": round(self.metrics.total_cost, 4),
                "average_latency_ms": round(self.metrics.average_latency_ms, 2),
                "error_rate": round(self.metrics.error_rate, 4),
                "rate_limit_hits": self.metrics.rate_limit_hits
            },
            "config": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries
            },
            "available_models": available_models[:10],  # Limit for brevity
            "status": "healthy" if self.initialized else "uninitialized"
        }


class MockLLMClient(LLMClient):
    """Simple mock LLM client for local development and CI testing.

    This mock client provides deterministic, fast responses and minimal
    behavior so unit tests and local runs do not require network access
    or API keys.
    """

    async def initialize(self) -> None:
        self._initialized = True

    async def chat_completion(self, messages: List[Dict[str, str]], stream: bool = True) -> AsyncGenerator[str, None]:
        # Simple mock response echoing the last user message
        content = ""
        if messages:
            last = messages[-1].get("content", "")
            content = f"Mock response to: {last}"
        if stream:
            # Yield as one chunk for simplicity
            yield content
        else:
            yield content

    async def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    async def get_available_models(self) -> List[str]:
        return ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview"]

    async def close(self) -> None:
        self._initialized = False
