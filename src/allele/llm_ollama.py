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

"""Ollama LLM client implementation for local and cloud AI models."""

import json
import os
from typing import AsyncGenerator, Dict, List, Optional

import httpx
import structlog

from .llm_client import LLMClient, LLMConfig
from .llm_exceptions import (
    LLMGenerationError,
    LLMInitializationError,
    LLMModelNotAvailableError,
    LLMTimeoutError,
)

logger = structlog.get_logger(__name__)

class OllamaClient(LLMClient):
    """Ollama LLM client implementation for local and cloud AI models."""

    # Common Ollama models (can be extended)
    COMMON_MODELS = [
        "llama2", "llama2:7b", "llama2:13b", "llama2:70b",
        "codellama", "codellama:13b", "codellama:34b",
        "mistral", "mistral:7b",
        "orca-mini", "orca2:13b", "orca2:7b",
        "vicuna", "vicuna:13b", "vicuna:7b",
        "llava", "llava:13b",
        "neural-chat", "starling-lm", "openchat"
    ]

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._http_client: Optional[httpx.AsyncClient] = None

        # Enhanced base URL with cloud support
        self._base_url = getattr(config, 'base_url', None) or 'http://localhost:11434'
        self._base_url = self._base_url.rstrip('/')  # Remove trailing slash

        # Headers for authentication (especially for Ollama Cloud)
        self._headers = getattr(config, 'headers', None) or {}

    async def initialize(self) -> None:
        """Initialize Ollama client with connectivity test."""
        try:
            self.logger.info("Initializing Ollama client", base_url=self._base_url)

            # Auto-configure authentication for Ollama Cloud
            headers = self._headers.copy()
            if 'ollama.com' in self._base_url:
                api_key = os.getenv('OLLAMA_API_KEY')
                if api_key:
                    headers['Authorization'] = f'Bearer {api_key}'
                    self.logger.debug("Using Ollama Cloud authentication")
                else:
                    self.logger.warning("OLLAMA_API_KEY not set for Ollama Cloud")

            # Test connection with authentication
            async with httpx.AsyncClient(timeout=self.config.timeout, headers=headers) as client:
                response = await client.get(f"{self._base_url}/api/version")
                if response.status_code != 200:
                    raise LLMInitializationError(
                        f"Ollama server not accessible at {self._base_url}",
                        "ollama",
                        self.config.model
                    )

            # Create persistent client with authentication
            self._http_client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self.config.timeout,
                headers=headers
            )

            # Validate model availability
            await self._validate_model_available()

            self._initialized = True
            self.logger.info("Ollama client initialized successfully",
                           model=self.config.model,
                           base_url=self._base_url,
                           cloud_auth='ollama.com' in self._base_url)

        except httpx.TimeoutException:
            raise LLMTimeoutError(
                f"Timeout connecting to Ollama server at {self._base_url}",
                "ollama",
                self.config.model,
                timeout_seconds=self.config.timeout
            )
        except Exception as e:
            self.logger.error("Ollama client initialization failed", error=str(e))
            raise LLMInitializationError(
                f"Failed to initialize Ollama client: {e}",
                "ollama",
                self.config.model
            ) from e

    async def _validate_model_available(self) -> None:
        """Validate that the requested model is available."""
        available_models = await self.get_available_models()
        if self.config.model not in available_models:
            raise LLMModelNotAvailableError(
                f"Model '{self.config.model}' not available in Ollama",
                "ollama",
                self.config.model,
                available_models=available_models[:20]
            )

    async def chat_completion(
        self, messages: List[Dict[str, str]], stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion using Ollama API."""
        if not self._initialized or not self._http_client:
            raise LLMGenerationError("Client not initialized", "ollama", self.config.model)

        # Convert messages to Ollama format
        ollama_messages = []
        system_message = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_message = content
            else:
                ollama_messages.append({"role": role, "content": content})

        # Ollama API payload
        payload = {
            "model": self.config.model,
            "messages": ollama_messages,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }

        if system_message:
            payload["system"] = system_message

        async def _ollama_request():
            try:
                response = await self._http_client.post(
                    "/api/chat",
                    json=payload,
                    timeout=self.config.timeout
                )

                if response.status_code == 404:
                    raise LLMModelNotAvailableError(
                        f"Model '{self.config.model}' not found",
                        "ollama",
                        self.config.model
                    )
                elif response.status_code == 400:
                    raise LLMGenerationError(
                        "Invalid request parameters",
                        "ollama",
                        self.config.model
                    )
                elif response.status_code >= 500:
                    raise LLMGenerationError(
                        f"Ollama server error: {response.status_code}",
                        "ollama",
                        self.config.model
                    )

                if stream:
                    # Handle streaming response
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                if "message" in chunk and "content" in chunk["message"]:
                                    content = chunk["message"]["content"]
                                    if content:  # Only yield non-empty content
                                        yield content

                                # Update metrics (approximate)
                                self.metrics.total_tokens += 1

                            except json.JSONDecodeError:
                                continue
                else:
                    # Handle non-streaming response
                    result = response.json()
                    if "message" in result and "content" in result["message"]:
                        content = result["message"]["content"]
                        yield content
                        self.metrics.total_tokens += len(content.split()) * 2  # Rough estimate

            except httpx.TimeoutException:
                raise LLMTimeoutError(
                    f"Ollama request timeout after {self.config.timeout}s",
                    "ollama",
                    self.config.model,
                    timeout_seconds=self.config.timeout
                )
            except httpx.ConnectError:
                raise LLMInitializationError(
                    f"Cannot connect to Ollama server at {self._base_url}",
                    "ollama",
                    self.config.model
                )
            except Exception as e:
                raise LLMGenerationError(
                    f"Ollama generation error: {e}",
                    "ollama",
                    self.config.model
                ) from e

        # Apply rate limiting and retry logic
        async for chunk in self._retry_with_exponential_backoff(
            _ollama_request,
            "Ollama chat completion",
            tokens_used=1
        ):
            yield chunk

    async def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Ollama models are local, so cost is effectively zero."""
        return 0.0

    async def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            if not self._http_client:
                # Fallback if not initialized
                return self.COMMON_MODELS.copy()

            response = await self._http_client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                return models if models else self.COMMON_MODELS.copy()
            else:
                self.logger.warning("Failed to fetch models from Ollama API")
                return self.COMMON_MODELS.copy()

        except Exception as e:
            self.logger.warning("Error fetching Ollama models, using fallback", error=str(e))
            return self.COMMON_MODELS.copy()

    async def close(self) -> None:
        """Clean up HTTP client resources."""
        if self._http_client:
            try:
                await self._http_client.aclose()
                self.logger.debug("Ollama client closed")
            except Exception as e:
                self.logger.warning("Error closing Ollama client", error=str(e))
            finally:
                self._http_client = None
                self._initialized = False
