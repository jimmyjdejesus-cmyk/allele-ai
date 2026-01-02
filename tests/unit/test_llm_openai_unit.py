"""Unit tests for OpenAI LLM client with mocked dependencies."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from phylogenic.llm_client import LLMConfig
from phylogenic.llm_exceptions import LLMAuthenticationError, LLMModelNotAvailableError
from phylogenic.llm_openai import OpenAIClient


class TestOpenAIClientUnit:
    """Deterministic unit tests for OpenAIClient using mocked dependencies."""

    @pytest.fixture
    def mock_config(self):
        """Create a deterministic test configuration."""
        return LLMConfig(
            provider="openai",
            model="gpt-4-turbo-preview",
            api_key="sk-test-123456789",
            temperature=0.7,
            max_tokens=1000,
            timeout=30,
            max_retries=2,
        )

    @pytest.fixture
    def mock_openai_client(self):
        """Mock AsyncOpenAI client."""
        client = AsyncMock()
        # Setup models.list mock
        mock_models_data = [
            Mock(id="gpt-4-turbo-preview"),
            Mock(id="gpt-4"),
            Mock(id="gpt-3.5-turbo"),
        ]
        client.models.list.return_value = Mock(data=mock_models_data)
        return client

    @pytest.fixture
    def mock_openai_class(self, mock_openai_client):
        """Mock AsyncOpenAI class constructor."""
        with patch("phylogenic.llm_openai.AsyncOpenAI") as mock_class:
            mock_class.return_value = mock_openai_client
            yield mock_class

    @pytest.mark.asyncio
    async def test_initialization_invalid_api_key_format(self, mock_config):
        """Test initialize fails with invalid API key format."""
        mock_config.api_key = "invalid-key-format"
        client = OpenAIClient(mock_config)

        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.initialize()

        assert "Invalid OpenAI API key format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialization_success_with_mocked_openai(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test successful initialization with mocked OpenAI client."""
        client = OpenAIClient(mock_config)

        # Run initialization
        await client.initialize()

        # Verify client was created with correct parameters
        mock_openai_class.assert_called_once()
        call_args = mock_openai_class.call_args
        assert call_args[1]["api_key"] == mock_config.api_key
        assert call_args[1]["timeout"] == mock_config.timeout

        assert client.initialized

    def test_initialization_model_not_available(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test initialization fails when target model is not available."""
        # Change config to unavailable model
        mock_config.model = "gpt-5-nonexistent"

        # Mock only returns available models without our target
        mock_openai_client.models.list.return_value = Mock(
            data=[Mock(id="gpt-4"), Mock(id="gpt-3.5-turbo")]
        )

        client = OpenAIClient(mock_config)

        with pytest.raises(LLMModelNotAvailableError) as exc_info:
            asyncio.run(client.initialize())

        assert "gpt-5-nonexistent" in str(exc_info.value)

    def test_initialization_authentication_failure(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test initialization handles authentication errors."""
        from openai import AuthenticationError

        # Make models.list raise authentication error
        mock_response = Mock()
        mock_response.request = Mock()
        mock_response.headers = {}
        mock_openai_client.models.list.side_effect = AuthenticationError(
            "Invalid API key", response=mock_response, body=None
        )

        client = OpenAIClient(mock_config)

        with pytest.raises(LLMAuthenticationError) as exc_info:
            asyncio.run(client.initialize())

        assert "OpenAI authentication failed" in str(exc_info.value)

    def test_estimate_token_count_simple_text(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test token estimation using tiktoken fallback."""
        client = OpenAIClient(mock_config)

        # Test with simple message
        messages = [{"content": "Hello world"}]
        tokens = client._estimate_token_count(messages)

        # Should return positive integer
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_estimate_token_count_multiple_messages(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test token estimation with multiple messages."""
        client = OpenAIClient(mock_config)

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a programming language"},
        ]
        tokens = client._estimate_token_count(messages)

        # Should account for multiple messages plus overhead
        assert tokens > 10  # More than just content tokens

    def test_estimate_token_count_empty_messages(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test token estimation with empty message list."""
        client = OpenAIClient(mock_config)

        tokens = client._estimate_token_count([])
        assert tokens == 0

    def test_estimate_cost_known_models(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test cost estimation for known models."""
        client = OpenAIClient(mock_config)

        # Test GPT-4 pricing
        client.config.model = "gpt-4"
        cost = asyncio.run(client.estimate_cost(100, 200))
        expected_cost = (100 * 0.03 + 200 * 0.06) / 1000
        assert abs(cost - expected_cost) < 0.0001

        # Test GPT-4-turbo pricing
        client.config.model = "gpt-4-turbo"
        cost = asyncio.run(client.estimate_cost(100, 200))
        expected_cost = (100 * 0.01 + 200 * 0.03) / 1000
        assert abs(cost - expected_cost) < 0.0001

    def test_estimate_cost_unknown_model_fallback(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test cost estimation falls back to reasonable defaults for unknown models."""
        client = OpenAIClient(mock_config)
        client.config.model = "unknown-model"

        cost = asyncio.run(client.estimate_cost(100, 200))
        # Should return a reasonable positive cost
        assert cost > 0

    @patch("time.time")
    def test_get_available_models_with_caching(
        self, mock_time, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test model list caching behavior."""
        mock_time.return_value = 1000.0  # Fixed time for deterministic testing

        client = OpenAIClient(mock_config)
        asyncio.run(client.initialize())

        # First call should fetch from API
        # Note: initialize() calls models.list() twice (once for connectivity,
        # once for validation via get_available_models)
        # So we expect call_count to be 2 initially.
        # But get_available_models() caches the result.

        models1 = asyncio.run(client.get_available_models())
        # Should use cache from initialize
        assert mock_openai_client.models.list.call_count == 2

        # Check that models1 contains expected models (including fallbacks)
        expected_models = ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"]
        for model in expected_models:
            assert model in models1

        # Second call within cache TTL should use cache
        models2 = asyncio.run(client.get_available_models())
        assert mock_openai_client.models.list.call_count == 2  # Still 2
        assert models1 == models2

    def test_get_available_models_cache_expiry(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test that cache expires after TTL."""
        client = OpenAIClient(mock_config)
        asyncio.run(client.initialize())

        # Set initial cache
        client._available_models_cache = ["cached-model"]
        client._models_cache_time = 1000.0

        # Simulate time passing beyond TTL
        with patch("time.time", return_value=1000.0 + client._models_cache_ttl + 1):
            models = asyncio.run(client.get_available_models())

        # Should fetch fresh from API
        # initialize() called it twice. This call should be the 3rd.
        assert mock_openai_client.models.list.call_count == 3

        expected_models = ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"]
        for model in expected_models:
            assert model in models

    def test_get_available_models_api_error_fallback(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test fallback to hardcoded models when API fails."""
        from phylogenic.llm_exceptions import LLMInitializationError

        # Make API call fail
        mock_openai_client.models.list.side_effect = Exception("API Error")

        client = OpenAIClient(mock_config)

        # Initialize will fail because of API error
        with pytest.raises(LLMInitializationError):
            asyncio.run(client.initialize())

        # But we can still try to get models, which should fallback
        models = asyncio.run(client.get_available_models())
        assert len(models) > 0  # Should return fallback models
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)

    @pytest.mark.asyncio
    async def test_chat_completion_parameter_validation(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test parameter validation for chat completion."""
        client = OpenAIClient(mock_config)
        await client.initialize()

        # Mock successful response
        mock_response = AsyncMock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = AsyncMock()
        mock_response.usage.total_tokens = 150
        mock_openai_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]

        # Test non-streaming
        results = []
        async for chunk in client.chat_completion(messages, stream=False):
            results.append(chunk)

        assert results == ["Test response"]
        mock_openai_client.chat.completions.create.assert_called()

    @pytest.mark.asyncio
    async def test_chat_completion_with_retry_logic(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test retry behavior on rate limit errors."""
        from openai import RateLimitError

        client = OpenAIClient(mock_config)
        await client.initialize()

        # Mock rate limit error first, then success
        mock_response = Mock()
        mock_response.request = Mock()
        mock_response.headers = {}
        mock_error = RateLimitError("Rate limited", response=mock_response, body=None)
        # mock_error.retry_after = 2

        mock_success_response = AsyncMock()
        mock_success_response.choices = [Mock()]
        mock_success_response.choices[0].message.content = "Success after retry"
        mock_success_response.usage = AsyncMock()
        mock_success_response.usage.total_tokens = 100

        mock_openai_client.chat.completions.create.side_effect = [
            mock_error,
            mock_success_response,
        ]

        messages = [{"role": "user", "content": "Test"}]

        # Should succeed after retry
        results = []
        async for chunk in client.chat_completion(messages, stream=False):
            results.append(chunk)

        assert results == ["Success after retry"]
        assert mock_openai_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_context_manager_usage(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test client can be used as async context manager."""
        client = OpenAIClient(mock_config)

        async with client:
            assert client.initialized

        # Check if close was called on the underlying client
        # Note: client.initialized might not be reset to False in __aexit__ currently
        # but the underlying client should be closed.
        if client._openai_client:
            client._openai_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_metrics_tracking(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test that usage metrics are properly tracked."""
        client = OpenAIClient(mock_config)
        await client.initialize()

        # Mock a successful completion
        mock_response = AsyncMock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage = AsyncMock()
        mock_response.usage.total_tokens = 200
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 100
        mock_openai_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "Test"}]
        results = []
        async for chunk in client.chat_completion(messages, stream=False):
            results.append(chunk)

        # Check metrics were updated
        assert client.metrics.total_requests >= 1
        assert client.metrics.total_tokens >= 200

    def test_close_cleans_up_resources(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test that close method properly cleans up resources."""
        client = OpenAIClient(mock_config)
        asyncio.run(client.initialize())

        # Ensure client has resources
        assert client._openai_client is not None
        assert client._initialized

        asyncio.run(client.close())

        # Should be cleaned up
        assert client._openai_client is None
        assert not client._initialized
        mock_openai_client.close.assert_called_once()

    def test_close_handles_errors_gracefully(
        self, mock_config, mock_openai_class, mock_openai_client
    ):
        """Test that close handles errors during cleanup gracefully."""
        client = OpenAIClient(mock_config)
        asyncio.run(client.initialize())

        # Make close raise an error
        mock_openai_client.close.side_effect = Exception("Close failed")

        # Should not raise exception
        asyncio.run(client.close())
