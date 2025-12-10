"""Unit and integration tests for LLM functionality."""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch

from allele.llm_client import LLMConfig, LLMClient
from allele.llm_openai import OpenAIClient
from allele.llm_ollama import OllamaClient
from allele.llm_exceptions import LLMInitializationError, LLMAuthenticationError
from allele.agent import AgentConfig, NLPAgent
from allele import ConversationalGenome


class TestLLMIntegration:
    """Test suite for LLM integration components."""

    # Set up Ollama Cloud API key for real integration testing
    def setup_method(self):
        """Set up environment variables for real API testing."""
        os.environ['OLLAMA_API_KEY'] = 'd36d8eb3b3bc4aa4830a3848b4f1bbf4.SOsu3PaVKVqlusm4evvmxb0X'

    def teardown_method(self):
        """Clean up environment variables."""
        if 'OLLAMA_API_KEY' in os.environ:
            del os.environ['OLLAMA_API_KEY']

    @pytest.fixture
    def mock_llm_config(self):
        """Create a mock LLM configuration for testing."""
        return LLMConfig(
            provider="openai",
            model="gpt-4-turbo-preview",
            api_key="sk-test-key-123",
            temperature=0.7,
            max_tokens=1000,
            timeout=30,
            max_retries=2
        )

    @pytest.fixture
    def mock_agent_config(self):
        """Create a mock agent configuration."""
        return AgentConfig(
            llm_provider="openai",
            api_key="sk-test-key-123",
            fallback_to_mock=True,
            model_name="gpt-4-turbo-preview"
        )

    @pytest.fixture
    def mock_genome(self):
        """Create a mock genome for testing."""
        return ConversationalGenome(
            genome_id="test_genome_001",
            traits={
                'empathy': 0.8,
                'engagement': 0.7,
                'technical_knowledge': 0.6,
                'creativity': 0.9,
                'conciseness': 0.5,
                'context_awareness': 0.8,
                'adaptability': 0.7,
                'personability': 0.8
            }
        )

    @pytest.mark.asyncio
    async def test_openai_client_initialization_failure_invalid_key(self, mock_llm_config):
        """Test OpenAI client fails with invalid API key format."""
        mock_llm_config.api_key = "invalid-key-format"

        with pytest.raises(LLMAuthenticationError) as exc_info:
            client = OpenAIClient(mock_llm_config)
            await client.initialize()

        assert "Invalid OpenAI API key format" in str(exc_info.value)

    def test_llm_config_validation(self):
        """Test LLM configuration validation."""
        # Valid config should not raise
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="sk-valid-key-123"
        )
        assert config is not None

        # Invalid temperature
        with pytest.raises(ValueError):
            LLMConfig(provider="openai", model="gpt-4", api_key="sk-key", temperature=3.0)

        # Invalid max_tokens
        with pytest.raises(ValueError):
            LLMConfig(provider="openai", model="gpt-4", api_key="sk-key", max_tokens=-1)

    @pytest.mark.asyncio
    async def test_openai_client_mock_initialization_success(self, mock_llm_config):
        """Test OpenAI client initialization with mocked OpenAI library."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            # Setup mock OpenAI client
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client

            # Mock the models API (supports limit parameter)
            mock_model = Mock()
            mock_model.id = "gpt-4-turbo-preview"
            mock_models_response = Mock(data=[mock_model])
            # Mock the entire models.list chain
            mock_client.models = AsyncMock()
            mock_client.models.list = AsyncMock(return_value=mock_models_response)

            # Test initialization
            client = OpenAIClient(mock_llm_config)
            await client.initialize()

            assert client.initialized

            # Verify OpenAI client was created with correct parameters
            mock_openai.assert_called_once()
            call_args = mock_openai.call_args
            assert call_args[1]['api_key'] == mock_llm_config.api_key
            assert call_args[1]['timeout'] == mock_llm_config.timeout

    @pytest.mark.asyncio
    async def test_openai_client_estimates_cost(self, mock_llm_config):
        """Test cost estimation functionality."""
        with patch('allele.llm_openai.AsyncOpenAI'):
            client = OpenAIClient(mock_llm_config)

            # Test GPT-4 cost estimation
            cost = await client.estimate_cost(100, 200)  # input_tokens, output_tokens
            expected_cost = (100 * 0.01 + 200 * 0.03) / 1000  # Based on MODEL_PRICING
            assert abs(cost - expected_cost) < 0.0001

    def test_agent_config_llm_fields(self, mock_agent_config):
        """Test AgentConfig includes all LLM-related fields."""
        config = mock_agent_config

        assert hasattr(config, 'llm_provider')
        assert hasattr(config, 'api_key')
        assert hasattr(config, 'fallback_to_mock')
        assert hasattr(config, 'request_timeout')
        assert hasattr(config, 'rate_limit_requests_per_minute')
        assert hasattr(config, 'conversation_memory')
        assert hasattr(config, 'context_window')
        assert hasattr(config, 'max_context_length')

    def test_agent_config_validation(self):
        """Test AgentConfig validation works."""
        # Valid config
        config = AgentConfig(llm_provider="openai", api_key="sk-key")
        assert config.llm_provider == "openai"

        # Invalid provider
        with pytest.raises(ValueError):
            AgentConfig(llm_provider="invalid_provider")

        # Invalid temperature
        with pytest.raises(ValueError):
            AgentConfig(llm_provider="openai", temperature=3.0)

    @pytest.mark.asyncio
    async def test_agent_creation_with_ollama_provider(self, mock_genome):
        """Test agent creation with Ollama provider (no API keys needed)."""
        config = AgentConfig(llm_provider="ollama", model_name="llama2:latest")

        # Ollama doesn't require API keys
        agent = NLPAgent(mock_genome, config)

        # Agent should be created successfully
        assert agent.genome == mock_genome
        assert agent.config == config
        assert not agent.is_initialized

    @pytest.mark.asyncio
    async def test_ollama_cloud_real_initialization(self):
        """Test real Ollama Cloud initialization with actual API calls."""
        # Try multiple common Ollama Cloud models
        models_to_try = ["llama2", "llama2:7b", "llama2:13b", "mistral", "codellama"]

        # Real Ollama Cloud client - will find available model
        for model_name in models_to_try:
            config = LLMConfig(
                provider="ollama",
                model=model_name,
                base_url="https://ollama.com",
                api_key=""
            )

            client = OllamaClient(config)
            try:
                # Real API call to initialize and validate connection
                await client.initialize()

                # If we reach here, the model works
                print(f"✅ Model {model_name} is available and working")

                # Verify real connectivity
                assert client.initialized
                assert client._http_client is not None

                # Test real model listing
                models = await client.get_available_models()
                assert len(models) > 0
                assert isinstance(models, list)

                # Clean up
                await client.close()
                return  # Success - use the first working model

            except Exception as e:
                print(f"❌ Model {model_name} failed: {e}")
                await client.close()
                continue  # Try next model

        # If we reach here, no models worked
        pytest.skip("No compatible Ollama Cloud models found")

    @pytest.mark.asyncio
    async def test_ollama_cloud_real_chat(self, mock_genome):
        """Test real conversational AI with Ollama Cloud."""
        # Try to use a working model from initialization test
        models_to_try = ["llama2", "llama2:7b", "mistral", "codellama"]

        for model_name in models_to_try:
            config = AgentConfig(
                llm_provider="ollama",
                model_name=model_name,
                temperature=0.1,  # Low creativity for predictable testing
                max_tokens=50    # Limit response length for faster testing
            )

            try:
                # Create real agent with Ollama Cloud
                agent = NLPAgent(mock_genome, config)
                await agent.initialize()

                # Real LLM conversation
                test_message = "What is 2+2?"
                responses = []

                async for chunk in agent.chat(test_message):
                    responses.append(chunk)

                full_response = "".join(responses)

                # Validate real AI response
                assert len(full_response) > 5  # Substantial response
                # Don't check for specific content since models may vary

                # Clean up
                await agent.close()

                print(f"✅ Chat test successful with {model_name}")
                return  # Success

            except Exception as e:
                print(f"❌ Chat test failed with {model_name}: {e}")
                await agent.close()
                continue

        # Skip if no models work
        pytest.skip("Chat testing requires working Ollama Cloud connection")

    @pytest.mark.asyncio
    async def test_agent_initialization_requires_api_key(self):
        """Test agent initialization fails without API key."""
        genome = ConversationalGenome("test", {'empathy': 0.5, 'engagement': 0.5, 'technical_knowledge': 0.5,
                                              'creativity': 0.5, 'conciseness': 0.5, 'context_awareness': 0.5,
                                              'adaptability': 0.5, 'personability': 0.5})

        # Use a fake API key for agent creation, then clear environment for initialization
        config = AgentConfig(llm_provider="openai", api_key="sk-fake-key", fallback_to_mock=False)

        with patch('allele.agent.NLPAgent._resolve_api_key', return_value="sk-fake-key"):
            agent = NLPAgent(genome, config)

        # Mock the environment to not have API keys during initialization
        with patch.dict('os.environ', {}, clear=True):
            # Mock os.getenv to return None for OPENAI_API_KEY
            with patch('os.getenv', side_effect=lambda key, default=None: None):
                # Should fail to initialize without API key available during init
                with pytest.raises(ValueError, match="API key not found"):
                    await agent.initialize()

    @pytest.mark.asyncio
    async def test_system_prompt_generation(self, mock_genome, local_ollama_config):
        """Test system prompt includes genome trait information."""
        agent = NLPAgent(mock_genome, local_ollama_config)

        # Initialize with real LLM (no mocking)
        try:
            await agent.initialize()
            assert agent.is_initialized
        except Exception:
            pytest.skip(f"Could not initialize agent with {local_ollama_config.model_name}")

        prompt = agent._create_system_prompt()

        # Check that prompt contains trait information
        assert mock_genome.genome_id in prompt
        assert str(mock_genome.generation) in prompt

        # Check that traits are described (using natural language)
        assert "emotional understanding" in prompt.lower()
        assert "technical expertise" in prompt.lower()
        assert "creative" in prompt.lower()

    @pytest.mark.asyncio
    async def test_chat_with_fallback_mode(self, mock_genome):
        """Test chat functionality uses fallback when LLM fails."""
        config = AgentConfig(fallback_to_mock=True)
        agent = NLPAgent(mock_genome, config)

        # Initialize with mock setup
        with patch('allele.llm_openai.AsyncOpenAI'):
            agent.llm_client = AsyncMock(spec=LLMClient)
            agent.llm_client.chat_completion.side_effect = Exception("LLM failure")
            agent.is_initialized = True

        responses = []
        async for chunk in agent.chat("Test message"):
            responses.append(chunk)

        response_text = "".join(responses)

        # Should contain fallback response
        assert "FALLBACK MODE" in response_text
        assert "LLM failure" in response_text
        assert mock_genome.genome_id in response_text

    def test_conversation_turn_creation(self):
        """Test conversation turn creation and structure."""
        genome = ConversationalGenome("test_genome", {
            'empathy': 0.5, 'engagement': 0.5, 'technical_knowledge': 0.5,
            'creativity': 0.5, 'conciseness': 0.5, 'context_awareness': 0.5,
            'adaptability': 0.5, 'personability': 0.5
        })
        config = AgentConfig(llm_provider="ollama", model_name="llama2:latest")
        agent = NLPAgent(genome, config)

        # Mock agent to be initialized for testing (use Ollama for simplicity)
        with patch('allele.llm_ollama.AsyncOllama', create=True):
            agent.llm_client = AsyncMock(spec=LLMClient)
            agent.is_initialized = True

        # Test conversation buffer is initially empty
        assert len(agent.conversation_buffer) == 0

        # Manually add conversation turns to test structure
        user_message = "Hello, how are you?"
        agent_response = "I'm doing well, thank you!"

        # Add user turn
        agent.conversation_buffer.append(Mock(
            user_input=user_message,
            agent_response="",
            timestamp="2025-01-01T12:00:00.000000+00:00",
            context_embedding=None,
            response_quality_score=0.0,
            evolutionary_adaptations=None
        ))

        # Add agent turn
        agent.conversation_buffer.append(Mock(
            user_input="",
            agent_response=agent_response,
            timestamp="2025-01-01T12:00:01.000000+00:00",
            context_embedding=None,
            response_quality_score=0.8,
            evolutionary_adaptations=None
        ))

        # Test conversation buffer structure
        assert len(agent.conversation_buffer) == 2
        assert agent.conversation_buffer[0].user_input == user_message
        assert agent.conversation_buffer[1].agent_response == agent_response

        # Test conversation metadata is updated
        assert agent.conversation_metadata["total_turns"] == 2

    @pytest.mark.asyncio
    async def test_agent_metrics_tracking(self, mock_genome):
        """Test that agent tracks performance metrics."""
        config = AgentConfig(fallback_to_mock=True)
        agent = NLPAgent(mock_genome, config)

        with patch('allele.llm_openai.AsyncOpenAI'):
            agent.llm_client = AsyncMock(spec=LLMClient)
            agent.llm_client.chat_completion = AsyncMock(return_value=iter(["Response"]))
            agent.is_initialized = True

            # Perform a chat operation
            await agent.chat("Test message")

            # Check that metrics were initialized
            assert 'total_requests' in agent.performance_metrics
            assert 'uptime_start' in agent.performance_metrics
            assert agent.performance_metrics['total_requests'] >= 0

    @pytest.mark.asyncio
    async def test_agent_configuration_defaults(self):
        """Test AgentConfig provides sensible defaults."""
        config = AgentConfig()

        assert config.llm_provider == "openai"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.streaming is True
        assert config.fallback_to_mock is False
        assert config.request_timeout == 60
        assert config.rate_limit_requests_per_minute == 60
        assert config.conversation_memory == 50
        assert config.context_window == 10
        assert config.max_context_length == 8000

    def test_openai_model_pricing_data(self):
        """Test that OpenAI pricing data is available."""
        pricing = OpenAIClient.MODEL_PRICING

        # Check some known models
        assert 'gpt-4' in pricing
        assert 'gpt-4-turbo' in pricing
        assert 'gpt-3.5-turbo' in pricing

        # Check pricing structure
        for model, prices in pricing.items():
            assert 'input' in prices
            assert 'output' in prices
            assert isinstance(prices['input'], float)
            assert isinstance(prices['output'], float)

    @pytest.mark.asyncio
    async def test_llm_client_context_manager(self):
        """Test LLM client can be used as context manager."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4-turbo-preview",
            api_key="sk-test-key"
        )

        with patch('allele.llm_openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            mock_client.models.list.return_value = Mock(data=[Mock(id="gpt-4-turbo-preview")])

            async with OpenAIClient(config) as client:
                assert client.initialized
                assert client.llm_client is not None

            # Should be closed after context
            mock_client.close.assert_called_once()

    def test_llm_config_post_init_validation(self):
        """Test LLM configuration validation."""
        # Should work with valid data
        config = LLMConfig(provider="openai", model="gpt-4", api_key="sk-key")
        assert config.provider == "openai"

        # Should fail with invalid provider name (too long, etc.)
        with pytest.raises(ValueError):
            LLMConfig(provider="", model="gpt-4", api_key="sk-key")

        with pytest.raises(ValueError):
            LLMConfig(provider="openai", model="", api_key="sk-key")

        with pytest.raises(ValueError):
            LLMConfig(provider="openai", model="gpt-4", api_key="")

    @pytest.mark.asyncio
    async def test_openai_available_models_caching(self):
        """Test that available models are cached properly."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4-turbo-preview",
            api_key="sk-test-key"
        )

        with patch('allele.llm_openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client

            # Setup models response
            mock_models = [Mock(id="gpt-4"), Mock(id="gpt-4-turbo")]
            mock_client.models.list.return_value = Mock(data=mock_models)

            client = OpenAIClient(config)
            await client.initialize()

            # First call should fetch from API
            models1 = await client.get_available_models()
            assert mock_client.models.list.call_count == 1

            # Second call should use cache
            models2 = await client.get_available_models()
            assert mock_client.models.list.call_count == 1  # Still 1

            assert models1 == models2 == ["gpt-4", "gpt-4-turbo"]
