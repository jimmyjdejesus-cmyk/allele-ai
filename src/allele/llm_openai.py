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

"""OpenAI LLM client implementation with comprehensive error handling."""

from typing import AsyncGenerator, Dict, List, Optional

import structlog
from openai import (
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from .llm_client import LLMClient, LLMConfig
from .llm_exceptions import (
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMGenerationError,
    LLMInitializationError,
    LLMModelNotAvailableError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = structlog.get_logger(__name__)

class OpenAIClient(LLMClient):
    """OpenAI LLM client implementation with full error handling and monitoring."""

    # OpenAI pricing as of December 2024 (update periodically)
    MODEL_PRICING: Dict[str, Dict[str, float]] = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-2024-04-09": {"input": 0.01, "output": 0.03},
        "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-vision-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-1106-vision-preview": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-2024-05-13": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        "gpt-3.5-turbo-instruct": {"input": 0.0015, "output": 0.002},
    }

    # Known models (fallback if API doesn't provide list)
    FALLBACK_MODELS = [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-turbo-preview",
        "gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
    ]

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._openai_client: Optional[AsyncOpenAI] = None
        self._available_models_cache: Optional[List[str]] = None
        self._models_cache_time: float = 0
        self._models_cache_ttl: int = 300  # 5 minutes

    async def initialize(self) -> None:
        """Initialize OpenAI client with comprehensive validation."""
        try:
            self.logger.info("Initializing OpenAI client")

            # Validate API key format
            if not self.config.api_key.startswith("sk-"):
                raise LLMAuthenticationError(
                    "Invalid OpenAI API key format. Key must start with 'sk-'",
                    "openai",
                    self.config.model
                )

            # Create client
            self._openai_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=0  # We handle retries ourselves
            )

            # Validate connection and permissions
            await self._validate_connection_and_permissions()

            self._initialized = True
            self.logger.info("OpenAI client initialized successfully",
                           model=self.config.model,
                           temperature=self.config.temperature)

        except Exception as e:
            self.logger.error("OpenAI client initialization failed", error=str(e))
            raise LLMInitializationError(
                f"Failed to initialize OpenAI client: {e}",
                "openai",
                self.config.model,
                {
                    "api_key_provided": bool(self.config.api_key),
                    "api_key_format_valid": self.config.api_key.startswith("sk-") if self.config.api_key else False
                }
            ) from e

    async def _validate_connection_and_permissions(self) -> None:
        """Validate OpenAI connection and API permissions."""
        try:
            # Test basic connectivity with a minimal call
            models = await self._openai_client.models.list(limit=1)
            if not hasattr(models, 'data') or not models.data:
                raise LLMInitializationError("No models available - check API permissions")

            # Verify target model is accessible
            available_models = await self.get_available_models()
            if self.config.model not in available_models:
                raise LLMModelNotAvailableError(
                    f"Model '{self.config.model}' is not available",
                    "openai",
                    self.config.model,
                    available_models=available_models[:20]  # Limit for error message
                )

            self.logger.debug("OpenAI connection and permissions validated successfully",
                            available_models_count=len(available_models),
                            target_model=self.config.model)

        except AuthenticationError as e:
            raise LLMAuthenticationError(
                f"OpenAI authentication failed: {e}",
                "openai",
                self.config.model
            ) from e
        except APIError as e:
            raise LLMInitializationError(
                f"OpenAI API error during validation: {e}",
                "openai",
                self.config.model
            ) from e
        except Exception as e:
            raise LLMInitializationError(
                f"Unexpected error during OpenAI validation: {e}",
                "openai",
                self.config.model
            ) from e

    async def chat_completion(
        self, messages: List[Dict[str, str]], stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Generate streaming or non-streaming chat completion with error handling."""
        if not self._initialized or not self._openai_client:
            raise LLMGenerationError("Client not initialized", "openai", self.config.model)

        # Estimate input tokens for rate limiting
        estimated_input_tokens = self._estimate_token_count(messages)

        async def _call_openai():
            try:
                # Prepare request parameters
                request_params = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "stream": stream
                }

                # Add usage tracking for streaming responses
                if stream:
                    request_params["stream_options"] = {"include_usage": True}

                # Make API call
                response = await self._openai_client.chat.completions.create(**request_params)

                total_output_tokens = 0
                response_content = ""

                if stream:
                    async for chunk in response:
                        if hasattr(chunk, 'choices') and chunk.choices:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                content = delta.content
                                response_content += content
                                total_output_tokens += self._estimate_token_count([{"content": content}])
                                yield content

                        # Handle usage information in final chunk
                        if hasattr(chunk, 'usage') and chunk.usage:
                            self._update_metrics(chunk.usage)
                else:
                    if hasattr(response, 'choices') and response.choices:
                        response_content = response.choices[0].message.content
                        yield response_content

                    if hasattr(response, 'usage') and response.usage:
                        self._update_metrics(response.usage)

            except RateLimitError as e:
                self.logger.warning("OpenAI rate limit hit", retry_after=e.retry_after if hasattr(e, 'retry_after') else None)
                raise LLMRateLimitError(
                    f"OpenAI rate limit exceeded: {e}",
                    "openai",
                    self.config.model,
                    {
                        "retry_after": e.retry_after if hasattr(e, 'retry_after') else None,
                        "estimated_tokens": estimated_input_tokens
                    }
                ) from e
            except APITimeoutError as e:
                raise LLMTimeoutError(
                    f"OpenAI request timeout: {e}",
                    "openai",
                    self.config.model,
                    {"timeout_seconds": self.config.timeout}
                ) from e
            except AuthenticationError as e:
                raise LLMAuthenticationError(
                    f"OpenAI authentication failed during generation: {e}",
                    "openai",
                    self.config.model
                ) from e
            except APIError as e:
                # Handle specific OpenAI error types
                if e.code == "content_filter":
                    raise LLMContentFilterError(
                        f"Content filtered by OpenAI: {e}",
                        "openai",
                        self.config.model
                    ) from e
                elif e.code == "model_not_found":
                    available_models = await self.get_available_models()
                    raise LLMModelNotAvailableError(
                        f"OpenAI model '{self.config.model}' not found: {e}",
                        "openai",
                        self.config.model,
                        available_models=available_models
                    ) from e
                elif "quota" in str(e).lower() or "billing" in str(e).lower():
                    raise LLMQuotaExceededError(
                        f"OpenAI quota exceeded: {e}",
                        "openai",
                        self.config.model,
                        {"error_code": e.code if hasattr(e, 'code') else None}
                    ) from e
                else:
                    raise LLMGenerationError(
                        f"OpenAI API error: {e}",
                        "openai",
                        self.config.model,
                        {
                            "error_code": e.code if hasattr(e, 'code') else None,
                            "error_type": e.type if hasattr(e, 'type') else None,
                            "estimated_input_tokens": estimated_input_tokens
                        }
                    ) from e
            except Exception as e:
                raise LLMGenerationError(
                    f"Unexpected error during OpenAI generation: {e}",
                    "openai",
                    self.config.model,
                    {"estimated_input_tokens": estimated_input_tokens}
                ) from e

        # Apply retry logic with error handling
        async for chunk in self._retry_with_exponential_backoff(
            _call_openai,
            "OpenAI chat completion",
            estimated_input_tokens
        ):
            yield chunk

    def _estimate_token_count(self, messages: List[Dict[str, str]]) -> int:
        """Estimate token count using tiktoken for accurate OpenAI tokenization."""
        try:
            import tiktoken
            # Use the appropriate encoding for the model
            # For GPT-4 and newer models, use cl100k_base
            encoding = tiktoken.get_encoding("cl100k_base")

            total_tokens = 0
            for msg in messages:
                content = str(msg.get("content", ""))

                # Count tokens in content
                tokens_in_content = len(encoding.encode(content))
                total_tokens += tokens_in_content

                # Add overhead for message formatting (role, content keys, etc.)
                # This is an approximation but much better than character-based
                formatting_overhead = 4  # Rough estimate for JSON structure
                total_tokens += formatting_overhead

            return total_tokens

        except ImportError:
            # Fallback to improved character-based estimation if tiktoken unavailable
            self.logger.warning("tiktoken not available, using improved character estimation")
            total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)

            # Better approximation: ~3.5-4 characters per token for English
            # Use 3.8 for conservative estimation (avoid API errors)
            estimated_tokens = max(1, total_chars // 3.8)

            # Add overhead proportional to messages
            overhead_tokens = len(messages) * 6  # Reduced from 10
            return estimated_tokens + overhead_tokens

    def _update_metrics(self, usage) -> None:
        """Update usage metrics from OpenAI usage data."""
        if hasattr(usage, 'total_tokens') and usage.total_tokens:
            self.metrics.total_tokens += usage.total_tokens

        # Estimate cost if usage details are available
        if hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
            cost = self.estimate_cost(usage.prompt_tokens, usage.completion_tokens)
            self.metrics.total_cost += cost

    async def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage and current pricing."""
        pricing = self.MODEL_PRICING.get(self.config.model, {"input": 0.01, "output": 0.03})
        total_cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000
        return total_cost

    async def get_available_models(self) -> List[str]:
        """Get list of available models from OpenAI API with caching."""
        import time

        # Check cache first
        current_time = time.time()
        if (self._available_models_cache is not None and
            current_time - self._models_cache_time < self._models_cache_ttl):
            return self._available_models_cache.copy()

        # Fetch from API
        try:
            if not self._initialized or not self._openai_client:
                self.logger.warning("Client not initialized, returning fallback models")
                return self.FALLBACK_MODELS.copy()

            models_response = await self._openai_client.models.list()
            available_models = [model.id for model in models_response.data if hasattr(model, 'id')]

            # Filter for chat models (exclude other types like embeddings, etc.)
            chat_models = [model for model in available_models if
                          model.startswith(('gpt-3.5', 'gpt-4')) and
                          'turbo' in model or model in ['gpt-4', 'gpt-4o', 'gpt-4o-mini']]

            if not chat_models:
                self.logger.warning("No chat models found in API response, using fallback")
                chat_models = self.FALLBACK_MODELS.copy()
            else:
                # Add fallback models that might not be in the API response
                for fallback in self.FALLBACK_MODELS:
                    if fallback not in chat_models:
                        chat_models.append(fallback)

            # Cache result
            self._available_models_cache = chat_models
            self._models_cache_time = current_time

            self.logger.debug("Retrieved available models",
                            model_count=len(chat_models),
                            cached=True)

            return chat_models

        except Exception as e:
            self.logger.warning("Failed to fetch available models from API, using fallback",
                              error=str(e))
            # Return fallback models on error
            return self.FALLBACK_MODELS.copy()

    async def close(self) -> None:
        """Clean up OpenAI client resources."""
        if self._openai_client:
            try:
                await self._openai_client.close()
                self.logger.debug("OpenAI client closed")
            except Exception as e:
                self.logger.warning("Error closing OpenAI client", error=str(e))
            finally:
                self._openai_client = None
                self._initialized = False
