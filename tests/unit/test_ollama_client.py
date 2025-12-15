"""Unit tests for Ollama LLM client with mocked HTTP dependencies."""

from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from allele.llm_client import LLMConfig
from allele.llm_exceptions import LLMGenerationError, LLMModelNotAvailableError
from allele.llm_ollama import OllamaClient


class TestOllamaClientUnit:
    """Deterministic unit tests for OllamaClient using mocked HTTP dependencies."""

    @pytest.fixture
    def mock_config(self):
        """Create a deterministic test configuration."""
        return LLMConfig(
            provider="ollama",
            model="llama2:latest",
            base_url="http://localhost:11434",
            temperature=0.7,
            max_tokens=1000,
            timeout=30,
            max_retries=2
        )

    @pytest.fixture
    def mock_ollama_config(self):
        """Create Ollama-specific test configuration."""
        return LLMConfig(
            provider="ollama",
            model="mistral:latest",
            base_url="https://ollama.com",
            api_key="test-api-key",
            temperature=0.1,
            max_tokens=500,
            timeout=60,
            max_retries=1
        )

    @pytest.fixture
    def mock_http_client(self):
        """Mock aiohttp ClientSession."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def mock_http_class(self, mock_http_client):
        """Mock aiohttp ClientSession class constructor."""
        with patch('aiohttp.ClientSession') as mock_class:
            mock_class.return_value.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_class.return_value.__aexit__ = AsyncMock(return_value=None)
            yield mock_class

    @pytest.mark.asyncio
    async def test_initialization_success_local(self, mock_config, mock_http_class, mock_http_client):
        """Test successful initialization with local Ollama."""
        client = OllamaClient(mock_config)

        # Mock successful model listing
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "models": [
                {"name": "llama2:latest"},
                {"name": "codellama:latest"}
            ]
        })
        mock_http_client.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_http_client.get.return_value.__aexit__ = AsyncMock(return_value=None)

        await client.initialize()

        assert client.initialized
        assert client._http_client is not None
        mock_http_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialization_success_cloud(self, mock_ollama_config, mock_http_class, mock_http_client):
        """Test successful initialization with Ollama Cloud."""
        client = OllamaClient(mock_ollama_config)

        # Mock successful model listing with authentication
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "models": [
                {"name": "mistral:latest"},
                {"name": "llama2:13b"}
            ]
        })
        mock_http_client.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_http_client.get.return_value.__aexit__ = AsyncMock(return_value=None)

        await client.initialize()

        assert client.initialized

    @pytest.mark.asyncio
    async def test_initialization_model_not_available(self, mock_config, mock_http_class, mock_http_client):
        """Test initialization fails when target model is not available."""
        mock_config.model = "nonexistent-model:latest"
        client = OllamaClient(mock_config)

        # Mock response with different available models
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "models": [{"name": "llama2:latest"}]
        })
        mock_http_client.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)

        with pytest.raises(LLMModelNotAvailableError) as exc_info:
            await client.initialize()

        assert "nonexistent-model:latest" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialization_http_error(self, mock_config, mock_http_class, mock_http_client):
        """Test initialization handles HTTP errors."""
        client = OllamaClient(mock_config)

        # Mock HTTP error during model listing
        mock_http_client.get.side_effect = aiohttp.ClientError("Connection failed")

        with pytest.raises(Exception):  # Should be LLMInitializationError
            await client.initialize()

    @pytest.mark.asyncio
    async def test_get_available_models_success(self, mock_config, mock_http_class, mock_http_client):
        """Test successful model listing."""
        client = OllamaClient(mock_config)

        # Mock successful model listing
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "models": [
                {"name": "llama2:latest"},
                {"name": "mistral:latest"},
                {"name": "codellama:13b"}
            ]
        })
        mock_http_client.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)

        await client.initialize()
        models = await client.get_available_models()

        expected_models = ["llama2:latest", "mistral:latest", "codellama:13b"]
        assert models == expected_models
        assert mock_http_client.get.called

    @pytest.mark.asyncio
    async def test_chat_completion_basic_success(self, mock_config, mock_http_class, mock_http_client):
        """Test basic chat completion."""
        client = OllamaClient(mock_config)

        # Mock successful completion
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "response": "Hello! How can I help you today?",
            "done": True
        })
        mock_http_client.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)

        await client.initialize()  # Mock successful init
        client._initialized = True
        client._http_client = mock_http_client

        messages = [{"role": "user", "content": "Hello"}]

        chunks = []
        async for chunk in client.chat_completion(messages, stream=False):
            chunks.append(chunk)

        assert chunks == ["Hello! How can I help you today?"]
        mock_http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_completion_streaming_success(self, mock_config, mock_http_class, mock_http_client):
        """Test streaming chat completion."""
        client = OllamaClient(mock_config)

        # Mock streaming response
        stream_chunks = [
            {"response": "Hello", "done": False},
            {"response": "!", "done": False},
            {"response": " How are you?", "done": True}
        ]

        async def mock_stream():
            for chunk in stream_chunks:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value=chunk)
                yield mock_response

        # Set up the mock to return an async iterator
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aiter__ = AsyncMock(return_value=iter([
            Mock(json=AsyncMock(return_value=chunk)) for chunk in stream_chunks
        ]))
        mock_http_client.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)

        await client.initialize()
        client._initialized = True
        client._http_client = mock_http_client

        messages = [{"role": "user", "content": "Hello"}]

        chunks = []
        async for chunk in client.chat_completion(messages, stream=True):
            chunks.append(chunk)

        # Verify we got the streaming chunks
        assert len(chunks) >= 2  # At least some chunks
        mock_http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_completion_request_parameters(self, mock_config, mock_http_class, mock_http_client):
        """Test that chat completion uses correct request parameters."""
        mock_config.temperature = 0.5
        mock_config.max_tokens = 200
        client = OllamaClient(mock_config)

        # Mock successful completion
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "response": "Test response",
            "done": True
        })
        mock_http_client.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)

        await client.initialize()
        client._initialized = True
        client._http_client = mock_http_client

        messages = [{"role": "user", "content": "Test message"}]

        chunks = []
        async for chunk in client.chat_completion(messages, stream=False):
            chunks.append(chunk)

        # Verify request was made with correct parameters
        call_args = mock_http_client.post.call_args
        request_data = call_args[1]['json']

        assert request_data['model'] == mock_config.model
        assert request_data['prompt'] == "Test message"  # Simplified message conversion
        assert request_data['stream'] is False
        assert 'options' in request_data
        assert request_data['options']['temperature'] == mock_config.temperature
        assert request_data['options']['num_predict'] == mock_config.max_tokens

    @pytest.mark.asyncio
    async def test_estimate_cost_mock_client(self, mock_config, mock_http_class, mock_http_client):
        """Test cost estimation for Ollama (should be zero as it's local/cloud)."""
        client = OllamaClient(mock_config)

        cost = await client.estimate_cost(100, 200)
        assert cost == 0.0  # Ollama doesn't charge

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, mock_config, mock_http_class, mock_http_client):
        """Test client cleanup via context manager."""
        client = OllamaClient(mock_config)

        # Mock initialization
        client._initialized = True
        client._http_client = mock_http_client

        # Test context manager
        async with client:
            assert client.initialized

        # Should be closed
        assert not client.initialized
        assert client._http_client is None

    def test_config_validation_temperature(self):
        """Test config validation for temperature."""
        # Valid temperature
        config = LLMConfig(provider="ollama", model="llama2", api_key="", temperature=0.8)
        assert config.temperature == 0.8

        # Invalid temperature
        with pytest.raises(ValueError):
            LLMConfig(provider="ollama", model="llama2", api_key="", temperature=2.5)

    def test_config_validation_max_tokens(self):
        """Test config validation for max_tokens."""
        # Valid max_tokens
        config = LLMConfig(provider="ollama", model="llama2", api_key="", max_tokens=500)
        assert config.max_tokens == 500

        # Invalid max_tokens
        with pytest.raises(ValueError):
            LLMConfig(provider="ollama", model="llama2", api_key="", max_tokens=0)

    @pytest.mark.asyncio
    async def test_error_handling_generation_failure(self, mock_config, mock_http_class, mock_http_client):
        """Test error handling during generation."""
        client = OllamaClient(mock_config)

        # Mock HTTP error during generation
        mock_http_client.post.side_effect = aiohttp.ClientError("Request failed")

        await client.initialize()
        client._initialized = True
        client._http_client = mock_http_client

        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(LLMGenerationError) as exc_info:
            chunks = []
            async for chunk in client.chat_completion(messages, stream=False):
                chunks.append(chunk)

        assert "Request failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_message_formatting_system_and_user(self, mock_config, mock_http_class, mock_http_client):
        """Test proper message formatting for system and user messages."""
        client = OllamaClient(mock_config)

        # Mock successful completion
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "response": "Understood",
            "done": True
        })
        mock_http_client.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)

        await client.initialize()
        client._initialized = True
        client._http_client = mock_http_client

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is AI?"}
        ]

        chunks = []
        async for chunk in client.chat_completion(messages, stream=False):
            chunks.append(chunk)

        # Verify request included system message
        call_args = mock_http_client.post.call_args
        request_payload = call_args[1]['json']
        prompt = request_payload['prompt']

        # Check that system context is included
        assert "helpful assistant" in prompt.lower()
        assert "What is AI?" in prompt

    @pytest.mark.asyncio
    async def test_initialization_retries_on_failure(self, mock_config, mock_http_class, mock_http_client):
        """Test that initialization retries on temporary failures."""
        client = OllamaClient(mock_config)

        # Mock first request fails, second succeeds
        failure_response = AsyncMock()
        failure_response.status = 500
        failure_response.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=None, history=None, status=500
        )

        success_response = AsyncMock()
        success_response.status = 200
        success_response.json = AsyncMock(return_value={
            "models": [{"name": "llama2:latest"}]
        })

        mock_http_client.get.side_effect = [
            failure_response,
            success_response
        ]

        await client.initialize()

        # Should have made 2 requests
        assert mock_http_client.get.call_count == 2
        assert client.initialized

    def test_base_url_default_handling(self, mock_config):
        """Test default base URL handling."""
        client = OllamaClient(mock_config)
        assert client.config.base_url == "http://localhost:11434"

        # Test with explicit URL
        mock_config.base_url = "https://custom-ollama.com"
        client_custom = OllamaClient(mock_config)
        assert client_custom.config.base_url == "https://custom-ollama.com"
