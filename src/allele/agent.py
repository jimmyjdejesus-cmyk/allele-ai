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

"""NLP Agent creation and management for Allele.

This module provides high-level agent creation using conversational genomes.

Author: Bravetto AI Systems
Version: 1.0.0
"""

import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

import structlog

from .config import settings as allele_settings
from .exceptions import AgentError
from .genome import ConversationalGenome
from .kraken_lnn import KrakenLNN
from .llm_client import LLMClient, LLMConfig
from .types import ConversationTurn

logger = structlog.get_logger(__name__)

@dataclass
class AgentConfig:
    """Enhanced configuration for NLP agents with full LLM support.

    Attributes:
        # LLM Configuration
        llm_provider: LLM provider ('openai', 'anthropic', 'ollama')
        api_key: API key (None to read from environment)
        model_name: Specific model name (legacy, maps to llm_model)

        # Generation Parameters
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        streaming: Whether to enable streaming responses

        # Conversation Management
        conversation_memory: Messages to keep in history
        context_window: Recent messages for context
        max_context_length: Maximum context length in tokens

        # System Prompt Customization
        system_prompt_template: Template for system prompts

        # Feature Flags
        memory_enabled: Whether to enable conversation memory
        evolution_enabled: Whether to enable evolutionary adaptation
        kraken_enabled: Whether to use Kraken LNN processing

        # Reliability & Error Handling
        max_retry_attempts: Maximum retry attempts
        fallback_to_mock: Enable mock responses for development
        request_timeout: Request timeout in seconds

        # Rate Limiting
        rate_limit_requests_per_minute: Request rate limit
        rate_limit_tokens_per_minute: Token rate limit

        # Logging & Monitoring
        log_level: Logging level
        enable_metrics: Whether to enable metrics collection
        correlation_id_enabled: Whether to enable request correlation IDs
    """

    # LLM Configuration
    llm_provider: str = "openai"
    api_key: Optional[str] = None
    model_name: str = "gpt-4"  # Default to stable GPT-4 alias for tests

    # Generation Parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    streaming: bool = True

    # Conversation Management
    conversation_memory: int = 50
    context_window: int = 10
    max_context_length: int = 8000

    # System Prompt Customization
    system_prompt_template: str = """
You are an AI assistant with these personality traits:
{trait_descriptions}

Your genome ID: {genome_id}
Current generation: {generation}

{additional_context}

Respond naturally while embodying these traits in your communication style.
""".strip()

    # Feature Flags
    memory_enabled: bool = True
    evolution_enabled: bool = True
    kraken_enabled: bool = True

    # Reliability & Error Handling
    max_retry_attempts: int = 3
    # When true, missing external API keys will use an internal mock LLM for
    # development and CI environments so tests and local runs don't require
    # external credentials.
    fallback_to_mock: bool = True
    request_timeout: int = 60

    # Rate Limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 10000

    # Logging & Monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    correlation_id_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.llm_provider not in ['openai', 'anthropic', 'ollama']:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

        if not (0 <= self.temperature <= 2):
            raise ValueError(f"Temperature must be between 0 and 2, got {self.temperature}")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")

        if self.conversation_memory <= 0:
            raise ValueError("conversation_memory must be positive")

        if self.context_window <= 0:
            raise ValueError("context_window must be positive")

        # Defensive limits to avoid OOM from unbounded memory settings
        if self.conversation_memory > 1000:
            raise ValueError("conversation_memory is unreasonably large; must be <= 1000")

        if self.context_window > 100:
            raise ValueError("context_window is unreasonably large; must be <= 100")
        return None

    def validate(self) -> None:
        """Validate configuration parameters."""
        self.__post_init__()

    @classmethod
    def from_settings(cls, settings: Optional[Any] = None) -> "AgentConfig":
        """Create an AgentConfig from central settings."""
        if settings is None:
            settings = allele_settings
        agent = settings.agent
        return cls(
            model_name=agent.model_name,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
            streaming=agent.streaming,
            memory_enabled=agent.memory_enabled,
            evolution_enabled=agent.evolution_enabled,
            kraken_enabled=agent.kraken_enabled,
        )

class NLPAgent:
    """Enhanced NLP Agent with production-ready LLM integration and genome-based personality.

    This agent combines conversational genome traits with real LLM capabilities,
    providing dynamic personality-driven responses powered by state-of-the-art
    language models with comprehensive error handling and monitoring.

    Features:
    - Real LLM integration (OpenAI, Anthropic, Ollama)
    - Genome-based system prompts with trait-driven behavior
    - Conversation history and context management
    - Comprehensive error handling with fallback modes
    - Performance monitoring and logging
    - Rate limiting and retry logic
    - Kraken LNN integration for enhanced processing

    Example:
        >>> genome = ConversationalGenome("agent_001", {'empathy': 0.9})
        >>> config = AgentConfig(llm_provider="openai", model_name="gpt-4-turbo")
        >>> agent = NLPAgent(genome, config)
        >>> await agent.initialize()
        >>> async for response in agent.chat("Hello, how are you?"):
        ...     print(response, end='')
    """

    def __init__(
        self,
        genome: ConversationalGenome,
        config: AgentConfig
    ):
        """Initialize enhanced NLP agent with LLM integration.

        Args:
            genome: Conversational genome defining agent personality traits
            config: Comprehensive agent configuration

        Raises:
            ValueError: If genome or config validation fails
        """
        self.genome = genome
        self.config = config

        # Validate inputs
        self._validate_genome()
        self.config.validate()

        # Logger setup
        self.logger = logger.bind(
            agent_id=genome.genome_id,
            genome_generation=genome.generation,
            llm_provider=config.llm_provider
        )

        # LLM Client setup - resolve API key, with optional fallback to mock
        use_mock = False
        try:
            resolved_api_key = self._resolve_api_key()
        except ValueError as e:
            if self.config.fallback_to_mock:
                self.logger.warning("API key not found; falling back to Mock LLM", error=str(e))
                resolved_api_key = "sk-mock"
                use_mock = True
            else:
                raise

        self._use_mock = use_mock

        self.llm_config = LLMConfig(
            provider=config.llm_provider,
            model=config.model_name,
            api_key=resolved_api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.request_timeout,
            max_retries=config.max_retry_attempts,
            rate_limit_requests_per_minute=config.rate_limit_requests_per_minute,
            rate_limit_tokens_per_minute=config.rate_limit_tokens_per_minute
        )

        # Initialize LLM client (lazy loaded)
        self.llm_client: Optional[LLMClient] = None

        # Advanced conversation management
        self.conversation_buffer: List[ConversationTurn] = []
        self.conversation_metadata: Dict[str, Any] = {
            "total_turns": 0,
            "average_response_length": 0,
            "total_tokens_used": 0,
            "conversation_topics": set(),
            "last_interaction": None,
            "context_quality_score": 0.0
        }

        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency_ms": 0,
            "error_rate": 0,
            "total_cost": 0.0,
            "total_tokens_used": 0,
            "uptime_start": time.time()
        }

        # Component initialization flags
        self.is_initialized = False
        self.kraken_lnn: Optional[KrakenLNN] = None

        # Initialize Kraken LNN if enabled
        if config.kraken_enabled:
            try:
                self.kraken_lnn = KrakenLNN(reservoir_size=100)
                self.logger.debug("Kraken LNN initialized")
            except Exception as e:
                self.logger.warning("Failed to initialize Kraken LNN", error=str(e))
                self.kraken_lnn = None

        self.logger.info("NLP Agent created successfully",
                        traits=genome.traits,
                        provider=config.llm_provider)

    def _validate_genome(self) -> None:
        """Validate genome has required conversational traits."""
        required_traits = [
            'empathy', 'engagement', 'technical_knowledge', 'creativity',
            'conciseness', 'context_awareness', 'adaptability', 'personability'
        ]

        missing_traits = [t for t in required_traits if t not in self.genome.traits]
        if missing_traits:
            raise ValueError(f"Genome missing required traits: {missing_traits}")

        # Validate trait ranges
        for trait_name, value in self.genome.traits.items():
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid trait value for {trait_name}: {value}")

    @property
    def conversation_history(self) -> List[ConversationTurn]:
        """Compatibility alias for tests that expect `conversation_history` attribute."""
        return self.conversation_buffer

    def _resolve_api_key(self) -> str:
        """Resolve API key from config or environment variables."""
        if self.config.api_key:
            return self.config.api_key

        # Map provider to environment variable
        key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "ollama": "OLLAMA_API_KEY"  # Ollama doesn't actually use API keys for local models
        }

        env_var = key_map.get(self.config.llm_provider)
        if not env_var:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

        # Ollama uses local models, so no API key required
        if self.config.llm_provider == "ollama":
            return ""  # Empty string for Ollama (no auth needed)

        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(
                f"API key not found. Set {env_var} environment variable "
                "or provide api_key in AgentConfig."
            )

        return api_key

    async def initialize(self) -> bool:
        """Comprehensive agent initialization with LLM client setup.

        Initializes the LLM client, validates connectivity, and prepares
        all components for operation.

        Returns:
            bool: True if initialization successful

        Raises:
            AgentError: If initialization fails
        """
        try:
            self.logger.info("Starting agent initialization")

            # Step 1: Initialize LLM client
            await self._initialize_llm_client()

            # Step 2: Validate genome and configuration
            self._validate_genome()
            self.config.validate()

            # Step 3: Initialize Kraken if enabled
            if self.kraken_lnn and self.config.kraken_enabled:
                try:
                    # Kraken initialization would go here if it had async init
                    pass
                except Exception as e:
                    self.logger.warning("Kraken initialization failed, disabling", error=str(e))
                    self.kraken_lnn = None

            # Step 4: Test basic functionality (optional)
            if os.getenv("AGENT_TEST_ON_INIT", "false").lower() == "true":
                await self._test_basic_functionality()

            self.is_initialized = True
            self.logger.info("Agent initialization completed successfully",
                           llm_provider=self.config.llm_provider,
                           model=self.config.model_name,
                           kraken_enabled=self.kraken_lnn is not None)

            return True

        except Exception as e:
            self.logger.error("Agent initialization failed", error=str(e), exc_info=True)
            await self._cleanup_on_failure()
            raise AgentError(f"Agent initialization failed: {e}") from e

    async def _initialize_llm_client(self) -> None:
        """Initialize and validate LLM client based on provider."""
        try:
            # Create appropriate client based on provider (lazy imports to avoid dependency issues)
            if getattr(self, "_use_mock", False):
                # Use internal mock client for local/CI testing
                from .llm_client import MockLLMClient
                self.llm_client = MockLLMClient(self.llm_config)
            elif self.config.llm_provider == "openai":
                from .llm_openai import OpenAIClient
                self.llm_client = OpenAIClient(self.llm_config)
            elif self.config.llm_provider == "ollama":
                from .llm_ollama import OllamaClient
                self.llm_client = OllamaClient(self.llm_config)
            elif self.config.llm_provider == "anthropic":
                # Anthropic client would be imported and instantiated here
                raise NotImplementedError("Anthropic support coming soon")
            else:
                raise ValueError(f"Unsupported provider: {self.config.llm_provider}")

            # Initialize the client
            await self.llm_client.initialize()
            self.logger.debug("LLM client initialized successfully")

        except Exception as e:
            self.logger.error("LLM client initialization failed", error=str(e))
            raise

    async def _test_basic_functionality(self) -> None:
        """Test basic agent functionality with a minimal request."""
        test_messages = [{"role": "user", "content": "Hello"}]

        try:
            assert self.llm_client is not None
            async for _ in self.llm_client.chat_completion(test_messages, stream=False):
                break  # Just test that it doesn't error
            self.logger.debug("Basic functionality test passed")
        except Exception as e:
            raise AgentError(f"Basic functionality test failed: {e}") from e

    async def _cleanup_on_failure(self) -> None:
        """Clean up resources when initialization fails."""
        try:
            if self.llm_client:
                await self.llm_client.close()
        except Exception:
            pass  # Ignore cleanup errors during failure

    async def chat(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Enhanced conversational chat with real LLM integration.

        Generates personality-driven responses using the conversational genome
        and real LLM APIs, with comprehensive error handling and performance tracking.

        Args:
            message: User input message
            context: Optional context dictionary for conversation state

        Yields:
            Streaming response chunks if enabled, otherwise full response

        Raises:
            AgentError: If agent is not initialized or other errors occur
        """
        if not self.is_initialized:
            raise AgentError("Agent not initialized. Call initialize() first.")

        request_id = f"req_{uuid.uuid4().hex[:12]}" if self.config.correlation_id_enabled else None
        log_context = self.logger.bind(request_id=request_id) if request_id else self.logger

        start_time = time.time()
        response_chunks: List[str] = []

        try:
            log_context.info("Processing chat request",
                           message_length=len(message),
                           has_context=context is not None,
                           conversation_turns=len(self.conversation_buffer))

            # Step 1: Add user message to conversation history
            await self._add_conversation_turn("user", message, context)

            # Step 2: Generate system prompt with genome integration
            system_prompt = self._create_system_prompt(context)

            # Step 3: Prepare conversation messages for LLM
            messages = self._prepare_conversation_messages(system_prompt)

            # Step 4: Apply Kraken enhancement if available
            if self.kraken_lnn and self.config.kraken_enabled:
                messages = await self._enhance_with_kraken(messages)

            # Step 5: Truncate context if needed
            messages = self._truncate_context(messages)

            # Step 6: Generate response using LLM
            assert self.llm_client is not None
            async for chunk in self.llm_client.chat_completion(
                messages,
                stream=self.config.streaming,
            ):
                response_chunks.append(chunk)
                yield chunk

            # Step 7: Record assistant response
            full_response = "".join(response_chunks)
            await self._add_conversation_turn("assistant", full_response, context)

            # Step 8: Update performance metrics
            await self._update_performance_metrics(start_time, len(full_response), True)

            log_context.info("Chat request completed",
                           response_length=len(full_response),
                           duration_ms=round((time.time() - start_time) * 1000, 2))

        except Exception as e:
            # Record failure metrics
            await self._update_performance_metrics(start_time, 0, False)

            error_type = type(e).__name__
            log_context.error("Chat request failed",
                            error=str(e),
                            error_type=error_type,
                            exc_info=True)

            # Fallback to mock response if enabled
            if self.config.fallback_to_mock:
                log_context.warning("Using fallback mock response")
                async for chunk in self._generate_fallback_response(message, str(e)):
                    yield chunk
            else:
                raise AgentError(f"Chat failed: {e}") from e

    def _create_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Create comprehensive system prompt with genome trait integration."""
        traits = self.genome.traits

        # Enhanced trait descriptions with specific behavioral guidance
        trait_descriptions = []
        for trait_name, value in traits.items():
            if trait_name == 'empathy':
                desc = f"emotional understanding and compassionate responses ({value:.1f}/1.0)"
            elif trait_name == 'engagement':
                desc = f"conversational energy and enthusiasm ({value:.1f}/1.0)"
            elif trait_name == 'technical_knowledge':
                desc = f"depth of technical expertise and accuracy ({value:.1f}/1.0)"
            elif trait_name == 'creativity':
                desc = f"innovative thinking and creative problem-solving ({value:.1f}/1.0)"
            elif trait_name == 'conciseness':
                desc = f"balancing completeness with brevity ({value:.1f}/1.0)"
            elif trait_name == 'context_awareness':
                desc = f"understanding conversation history and maintaining continuity ({value:.1f}/1.0)"
            elif trait_name == 'adaptability':
                desc = f"flexible adaptation to different conversation styles ({value:.1f}/1.0)"
            elif trait_name == 'personability':
                desc = f"friendliness and natural human-like communication ({value:.1f}/1.0)"
            else:
                desc = f"{trait_name.replace('_', ' ')} ({value:.1f}/1.0)"

            # Include trait name explicitly to aid tests and readability
            trait_label = trait_name.replace("_", " ")
            trait_descriptions.append(f"- {trait_label}: {desc}")

        # Build context additions
        context_text = ""
        if context:
            context_items = []
            for key, value in context.items():
                if key == "user_id":
                    context_items.append(f"User: {value}")
                elif key == "session_id":
                    context_items.append(f"Session: {value}")
                elif key == "topic" and value:
                    context_items.append(f"Topic: {value}")
                elif key == "tone" and value:
                    context_items.append(f"Requested tone: {value}")

            if context_items:
                context_text = "\nContext:\n" + "\n".join(f"  {item}" for item in context_items)

        # Format with template
        prompt = self.config.system_prompt_template.format(
            trait_descriptions="\n".join(trait_descriptions),
            genome_id=self.genome.genome_id,
            generation=self.genome.generation,
            additional_context=context_text
        )

        return prompt

    def _prepare_conversation_messages(self, system_prompt: str) -> List[Dict[str, str]]:
        """Prepare conversation history for LLM context window."""
        messages = [{"role": "system", "content": system_prompt}]

        # Add recent conversation turns within context window
        recent_turns = self.conversation_buffer[-self.config.context_window:]

        for turn in recent_turns:
            # ConversationTurn uses `user_input` as the attribute name
            # (kept for backward compatibility with older tests that may
            # reference `user_message`). Prefer `user_input` when present.
            user_text = getattr(turn, "user_input", None) or getattr(turn, "user_message", None)
            if user_text:
                messages.append({
                    "role": "user",
                    "content": user_text
                })
            if turn.agent_response:
                messages.append({
                    "role": "assistant",
                    "content": turn.agent_response
                })

        return messages

    async def _enhance_with_kraken(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Apply Kraken LNN enhancement to messages."""
        if not self.kraken_lnn:
            return messages

        try:
            # Extract recent user message for processing
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            if user_messages:
                # Convert text to numerical sequence for Kraken
                text_to_process = user_messages[-1]["content"][:200]  # Limit for processing
                sequence = [float(ord(c)) / 255.0 for c in text_to_process]

                if len(sequence) >= 5:  # Minimum sequence length
                    result = await self.kraken_lnn.process_sequence(sequence)

                    # Add neural processing insights to system message
                    if messages and messages[0]["role"] == "system":
                        neural_insights = f"\nNeural processing: {result.get('liquid_outputs', [0])[:3]}"
                        messages[0]["content"] += neural_insights

            return messages

        except Exception as e:
            self.logger.warning("Kraken enhancement failed, proceeding without it",
                              error=str(e))
            return messages

    def _truncate_context(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Truncate conversation context to stay within token limits using proper tokenization."""
        # Keep system message always
        if not messages or messages[0]["role"] != "system":
            return messages

        system_msg = messages[0]
        other_messages = messages[1:]

        # Estimate tokens properly if available, fallback to character-based
        max_tokens = self._estimate_tokens([system_msg])
        estimated_tokens = max_tokens

        truncated_messages = [system_msg]

        for msg in reversed(other_messages):  # Add most recent first
            msg_tokens = self._estimate_tokens([msg])
            if estimated_tokens + msg_tokens <= self.config.max_context_length:
                truncated_messages.insert(1, msg)  # Insert after system message
                estimated_tokens += msg_tokens
            else:
                # Add truncation notice
                if len(truncated_messages) == 1:  # Only system message
                    truncated_messages.append({
                        "role": "system",
                        "content": "... [conversation history truncated due to length limits] ..."
                    })
                break

        return truncated_messages

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate token count using available tokenization methods."""
        try:
            # Try to use LLM client's token estimation if available
            if self.llm_client and hasattr(self.llm_client, '_estimate_token_count'):
                client = self.llm_client
                # `_estimate_token_count` is provider-specific (OpenAI/others) and
                # not part of the abstract `LLMClient` interface, so narrow safely
                # and coerce the result to int for callers.
                estimate = client._estimate_token_count(messages)
                return int(estimate)
            else:
                # Fallback to character-based approximation
                total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
                # Conservative estimate: ~3.5 characters per token
                return int(max(1, total_chars / 3.5))
        except Exception:
            # Ultimate fallback
            total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
            return int(max(1, total_chars / 4))

    async def _add_conversation_turn(
        self,
        role: str,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a conversation turn to the history with metadata."""
        turn = ConversationTurn(
            user_input=content if role == "user" else "",
            agent_response=content if role == "assistant" else "",
            timestamp=datetime.now(timezone.utc).isoformat(),
            context_embedding=None,  # Placeholder
            response_quality_score=0.0,  # Placeholder
            evolutionary_adaptations=None  # Placeholder
        )

        self.conversation_buffer.append(turn)
        self.conversation_metadata["total_turns"] += 1
        self.conversation_metadata["last_interaction"] = turn.timestamp

        # Trim buffer to maintain memory limits
        excess_turns = len(self.conversation_buffer) - self.config.conversation_memory
        if excess_turns > 0:
            self.conversation_buffer = self.conversation_buffer[excess_turns:]

        # Update average response length
        if role == "assistant":
            total_turns = self.conversation_metadata["total_turns"]
            avg_length = self.conversation_metadata["average_response_length"]
            self.conversation_metadata["average_response_length"] = (
                avg_length * (total_turns - 1) + len(content)
            ) / total_turns

    async def _update_performance_metrics(
        self,
        start_time: float,
        response_length: int,
        success: bool
    ) -> None:
        """Update comprehensive performance tracking metrics."""
        duration_ms = (time.time() - start_time) * 1000

        self.performance_metrics["total_requests"] += 1

        if success:
            self.performance_metrics["successful_requests"] += 1
            self.performance_metrics["total_tokens_used"] += response_length // 4  # Rough estimate
        else:
            self.performance_metrics["failed_requests"] += 1

        # Update average latency (exponential moving average)
        alpha = 0.1  # Smoothing factor
        current_avg = self.performance_metrics["average_latency_ms"]
        self.performance_metrics["average_latency_ms"] = (
            alpha * duration_ms + (1 - alpha) * current_avg
        )

        # Update error rate
        total_reqs = self.performance_metrics["total_requests"]
        self.performance_metrics["error_rate"] = (
            self.performance_metrics["failed_requests"] / total_reqs
        )

        # Update LLM client metrics if available
        if self.llm_client and hasattr(self.llm_client, 'metrics'):
            self.performance_metrics["total_cost"] = self.llm_client.metrics.total_cost

    async def _generate_fallback_response(self, message: str, error: str) -> AsyncGenerator[str, None]:
        """Generate fallback response when LLM integration fails."""
        fallback_response = f"""**FALLBACK MODE** (LLM Error: {error[:50]}...)

Hello! I received your message: "{message[:100]}{'...' if len(message) > 100 else ''}"

I apologize, but I'm currently operating in fallback mode due to a technical issue with the language model integration. While I can't provide a full AI-powered response right now, here's what I know:

**Genome ID**: {self.genome.genome_id}
**Generation**: {self.genome.generation}
**Key Traits**:
{chr(10).join(f"• {k}: {v:.1f}/1.0" for k, v in list(self.genome.traits.items())[:4])}

Please check your API keys and network connection, then restart the agent.

**End fallback response.**"""

        for word in fallback_response.split():
            yield word + " "
            await asyncio.sleep(0.01)  # Slow streaming to show it's fallback

    async def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history in a readable format."""
        return [{
            "timestamp": turn.timestamp,
            "user_input": turn.user_input,
            "agent_response": turn.agent_response,
            "quality_score": turn.response_quality_score
        } for turn in self.conversation_buffer]

    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive agent and LLM metrics."""
        llm_metrics = {}
        if self.llm_client and hasattr(self.llm_client, 'metrics'):
            llm_metrics = {
                "llm_requests": self.llm_client.metrics.total_requests,
                "llm_errors": self.llm_client.metrics.error_rate,
                "llm_cost": round(self.llm_client.metrics.total_cost, 4),
                "rate_limit_hits": self.llm_client.metrics.rate_limit_hits
            }

        return {
            "agent": {
                "id": self.genome.genome_id,
                "generation": self.genome.generation,
                "traits": self.genome.traits,
                "uptime_seconds": round(time.time() - self.performance_metrics["uptime_start"], 2)
            },
            "performance": self.performance_metrics,
            "conversation": dict(self.conversation_metadata),
            "llm": llm_metrics,
            "config": {
                "provider": self.config.llm_provider,
                "model": self.config.model_name,
                "streaming": self.config.streaming,
                "temperature": self.config.temperature,
                "fallback_enabled": self.config.fallback_to_mock
            }
        }

    async def reset_conversation(self) -> None:
        """Reset conversation history and refresh metadata."""
        self.conversation_buffer.clear()
        self.conversation_metadata = {
            "total_turns": 0,
            "average_response_length": 0,
            "total_tokens_used": 0,
            "conversation_topics": set(),
            "last_interaction": None,
            "context_quality_score": 0.0
        }
        self.logger.info("Conversation history reset")

    async def close(self) -> None:
        """Comprehensive cleanup of agent resources."""
        try:
            self.logger.info("Closing agent and cleaning up resources")

            # Close LLM client
            if self.llm_client:
                await self.llm_client.close()

            # Additional cleanup would go here
            self.is_initialized = False
            self.logger.info("Agent closed successfully")

        except Exception as e:
            self.logger.error("Error during agent cleanup", error=str(e))

async def create_agent(
    genome: ConversationalGenome,
    config: Optional[AgentConfig] = None
) -> NLPAgent:
    """Create and initialize an NLP agent.

    Args:
        genome: Conversational genome
        config: Optional agent configuration

    Returns:
        Initialized NLP agent

    Example:
        >>> genome = ConversationalGenome("agent_001")
        >>> agent = await create_agent(genome)
    """
    if config is None:
        config = AgentConfig()

    agent = NLPAgent(genome, config)
    await agent.initialize()

    return agent
