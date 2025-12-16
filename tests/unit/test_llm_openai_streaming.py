"""Unit tests for OpenAI streaming functionality with mocked dependencies."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from allele.llm_client import LLMConfig
from allele.llm_openai import OpenAIClient


class TestOpenAIStreamingUnit:
    """Deterministic unit tests for OpenAI streaming functionality."""

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
            max_retries=2
        )

    @pytest.fixture
    def mock_openai_client(self):
        """Mock AsyncOpenAI client."""
        client = AsyncMock()
        mock_models_data = [Mock(id="gpt-4-turbo-preview")]
        client.models.list.return_value = Mock(data=mock_models_data)
        return client

    @pytest.fixture
    def mock_openai_class(self, mock_openai_client):
        """Mock AsyncOpenAI class constructor."""
        with patch('allele.llm_openai.AsyncOpenAI') as mock_class:
            mock_class.return_value = mock_openai_client
            yield mock_class

    @pytest.mark.asyncio
    async def test_streaming_basic_functionality(self, mock_config, mock_openai_class, mock_openai_client):
        """Test basic streaming functionality with mocked responses."""
        client = OpenAIClient(mock_config)
        await client.initialize()

        # Mock streaming response with multiple chunks
        async def mock_stream():
            chunks = [
                MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))])
            ]
            for chunk in chunks:
                yield chunk
            # Final chunk with usage
            final_chunk = MagicMock(usage=MagicMock(total_tokens=50))
            final_chunk.choices = []
            yield final_chunk

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        messages = [{"role": "user", "content": "Hello"}]

        # Test streaming
        chunks = []
        async for chunk in client.chat_completion(messages, stream=True):
            chunks.append(chunk)

        assert chunks == ["Hello", " world", "!"]
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_with_empty_content(self, mock_config, mock_openai_class, mock_openai_client):
        """Test streaming handles chunks with no content."""
        client = OpenAIClient(mock_config)
        await client.initialize()

        async def mock_stream():
            chunks = [
                Mock(choices=[Mock(delta=Mock(content="Hello"))]),
                Mock(choices=[Mock(delta=Mock(content=""))]),  # Empty content
                Mock(choices=[Mock(delta=Mock(content=" world"))]),
                Mock(choices=[Mock(delta=Mock(content=None))]),  # No content at all
            ]
            for chunk in chunks:
                yield chunk

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        messages = [{"role": "user", "content": "Test"}]

        chunks = []
        async for chunk in client.chat_completion(messages, stream=True):
            chunks.append(chunk)

        # Should skip empty content and None
        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_non_streaming_response(self, mock_config, mock_openai_class, mock_openai_client):
        """Test non-streaming response format."""
        client = OpenAIClient(mock_config)
        await client.initialize()

        # Mock non-streaming response
        mock_response = AsyncMock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Complete response"
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 75
        mock_openai_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]

        # Test non-streaming
        chunks = []
        async for chunk in client.chat_completion(messages, stream=False):
            chunks.append(chunk)

        assert chunks == ["Complete response"]
        # Verify non-streaming parameters were passed
        call_args = mock_openai_client.chat.completions.create.call_args[1]
        assert call_args['stream'] is False

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, mock_config, mock_openai_class, mock_openai_client):
        """Test streaming handles errors gracefully."""
        client = OpenAIClient(mock_config)
        await client.initialize()

        # Mock streaming that raises an error
        async def mock_stream():
            yield Mock(choices=[Mock(delta=Mock(content="Hello"))])
            raise Exception("Stream error")

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        messages = [{"role": "user", "content": "Test"}]

        # Should not raise during iteration but may raise depending on implementation
        # This tests that at least the first chunk is yielded
        chunks = []
        try:
            async for chunk in client.chat_completion(messages, stream=True):
                chunks.append(chunk)
        except Exception:
            pass  # Expected for this test

        assert "Hello" in chunks  # At least got the first chunk

    @pytest.mark.asyncio
    async def test_streaming_request_parameters(self, mock_config, mock_openai_class, mock_openai_client):
        """Test that streaming requests include correct parameters."""
        client = OpenAIClient(mock_config)
        await client.initialize()

        # Mock empty stream
        async def mock_stream():
            final_chunk = MagicMock(usage=MagicMock(total_tokens=25))
            final_chunk.choices = []
            yield final_chunk

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        messages = [{"role": "user", "content": "Test streaming"}]

        chunks = []
        async for chunk in client.chat_completion(messages, stream=True):
            chunks.append(chunk)

        # Verify request parameters
        call_args = mock_openai_client.chat.completions.create.call_args[1]
        assert call_args['model'] == mock_config.model
        assert call_args['messages'] == messages
        assert call_args['temperature'] == mock_config.temperature
        assert call_args['max_tokens'] == mock_config.max_tokens
        assert call_args['stream'] is True
        assert 'stream_options' in call_args
        assert call_args['stream_options']['include_usage'] is True

    @pytest.mark.asyncio
    async def test_streaming_token_usage_tracking(self, mock_config, mock_openai_class, mock_openai_client):
        """Test that token usage is tracked during streaming."""
        client = OpenAIClient(mock_config)
        await client.initialize()

        async def mock_stream():
            for i in range(3):
                yield MagicMock(choices=[MagicMock(delta=MagicMock(content=f"Chunk{i}"))])
            # Final usage chunk
            final_chunk = MagicMock(usage=MagicMock(total_tokens=150, prompt_tokens=50, completion_tokens=100))
            final_chunk.choices = []
            yield final_chunk

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        messages = [{"role": "user", "content": "Test"}]

        chunks = []
        async for chunk in client.chat_completion(messages, stream=True):
            chunks.append(chunk)

        # Check that usage was recorded
        assert client.metrics.total_tokens >= 150

    @pytest.mark.asyncio
    async def test_streaming_with_system_messages(self, mock_config, mock_openai_class, mock_openai_client):
        """Test streaming with system and user messages."""
        client = OpenAIClient(mock_config)
        await client.initialize()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about AI."}
        ]

        async def mock_stream():
            yield Mock(choices=[Mock(delta=Mock(content="AI "))])
            yield Mock(choices=[Mock(delta=Mock(content="is "))])
            yield Mock(choices=[Mock(delta=Mock(content="awesome"))])

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        chunks = []
        async for chunk in client.chat_completion(messages, stream=True):
            chunks.append(chunk)

        assert chunks == ["AI ", "is ", "awesome"]
        # Verify the messages were passed correctly
        call_args = mock_openai_client.chat.completions.create.call_args[1]
        assert call_args['messages'] == messages

    @pytest.mark.asyncio
    async def test_streaming_preserves_message_order(self, mock_config, mock_openai_class, mock_openai_client):
        """Test that streaming preserves message order and content."""
        client = OpenAIClient(mock_config)
        await client.initialize()

        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2?"}
        ]

        response_parts = ["The ", "answer ", "is ", "4."]
        async def mock_stream():
            for part in response_parts:
                yield Mock(choices=[Mock(delta=Mock(content=part))])

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        chunks = []
        async for chunk in client.chat_completion(messages, stream=True):
            chunks.append(chunk)

        assert chunks == response_parts

    @pytest.mark.asyncio
    async def test_streaming_handles_large_content(self, mock_config, mock_openai_class, mock_openai_client):
        """Test streaming with larger content chunks."""
        client = OpenAIClient(mock_config)
        await client.initialize()

        # Simulate large response in multiple chunks
        large_content = "This is a much longer response that would typically come in multiple streaming chunks from the API."

        async def mock_stream():
            words = large_content.split()
            for word in words:
                yield Mock(choices=[Mock(delta=Mock(content=word + " "))])

        mock_openai_client.chat.completions.create.return_value = mock_stream()

        chunks = []
        async for chunk in client.chat_completion(messages=[{"role": "user", "content": "Test"}], stream=True):
            chunks.append(chunk)

        # Verify we got all the content
        received_content = "".join(chunks).strip()
        assert received_content == large_content
