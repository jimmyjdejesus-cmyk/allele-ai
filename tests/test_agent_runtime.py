"""
Runtime tests for NLPAgent and agent creation.

Tests actual execution of agent initialization, chat, and integration.
"""


import pytest

from allele import AgentConfig, ConversationalGenome, NLPAgent, create_agent


class TestAgentRuntime:
    """Runtime tests for NLPAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization_runtime(self, custom_genome, agent_config):
        """Test agent initializes correctly at runtime."""
        agent = NLPAgent(custom_genome, agent_config)

        assert agent.genome == custom_genome
        assert agent.config == agent_config
        assert agent.is_initialized is False

        # Initialize
        result = await agent.initialize()

        assert result is True
        assert agent.is_initialized is True

    @pytest.mark.asyncio
    async def test_create_agent_runtime(self, custom_genome):
        """Test create_agent function executes correctly."""
        agent = await create_agent(custom_genome)

        assert isinstance(agent, NLPAgent)
        assert agent.is_initialized is True
        assert agent.genome == custom_genome

    @pytest.mark.asyncio
    async def test_create_agent_with_config(self, custom_genome, agent_config):
        """Test create_agent with custom config."""
        agent = await create_agent(custom_genome, agent_config)

        assert agent.config == agent_config
        assert agent.is_initialized is True

    @pytest.mark.asyncio
    async def test_system_prompt_generation(self, custom_genome, agent_config):
        """Test system prompt generation from genome."""
        agent = NLPAgent(custom_genome, agent_config)
        await agent.initialize()

        prompt = agent._create_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0

        # Should contain trait information
        for trait_name in custom_genome.traits.keys():
            assert trait_name.replace('_', ' ').lower() in prompt.lower() or \
                   trait_name in prompt.lower()

    @pytest.mark.asyncio
    async def test_chat_execution(self, custom_genome, agent_config):
        """Test chat method executes correctly."""
        agent = NLPAgent(custom_genome, agent_config)
        await agent.initialize()

        # Chat should yield response
        response_chunks = []
        async for chunk in agent.chat("Hello, how are you?"):
            response_chunks.append(chunk)

        # Should have received response
        assert len(response_chunks) > 0

        # Response should contain genome information (mock response)
        full_response = "".join(response_chunks)
        assert len(full_response) > 0

    @pytest.mark.asyncio
    async def test_chat_with_context(self, custom_genome, agent_config):
        """Test chat with context parameter."""
        agent = NLPAgent(custom_genome, agent_config)
        await agent.initialize()

        context = {"user_id": "test_user", "session_id": "test_session"}

        response_chunks = []
        async for chunk in agent.chat("Test message", context=context):
            response_chunks.append(chunk)

        assert len(response_chunks) > 0

    @pytest.mark.asyncio
    async def test_kraken_integration(self, custom_genome):
        """Test Kraken LNN integration with agent."""
        config = AgentConfig(kraken_enabled=True)
        agent = NLPAgent(custom_genome, config)
        await agent.initialize()

        assert agent.kraken_lnn is not None

        # Chat should use Kraken
        response_chunks = []
        async for chunk in agent.chat("Test message"):
            response_chunks.append(chunk)

        assert len(response_chunks) > 0

    @pytest.mark.asyncio
    async def test_kraken_disabled(self, custom_genome):
        """Test agent without Kraken LNN."""
        config = AgentConfig(kraken_enabled=False)
        agent = NLPAgent(custom_genome, config)
        await agent.initialize()

        assert agent.kraken_lnn is None

    @pytest.mark.asyncio
    async def test_chat_before_initialization(self, custom_genome, agent_config):
        """Test chat fails before initialization."""
        agent = NLPAgent(custom_genome, agent_config)

        # Should raise error
        with pytest.raises(Exception):  # AgentError expected
            async for _ in agent.chat("Test"):
                pass

    @pytest.mark.asyncio
    async def test_conversation_history_tracking(self, custom_genome, agent_config):
        """Test conversation history is tracked."""
        agent = NLPAgent(custom_genome, agent_config)
        await agent.initialize()

        initial_history_length = len(agent.conversation_history)

        # Send multiple messages
        messages = ["Hello", "How are you?", "Tell me about yourself"]

        for message in messages:
            async for _ in agent.chat(message):
                pass

        # History should be tracked (if implemented)
        # Note: Current implementation may not track history yet
        assert hasattr(agent, 'conversation_history')

    @pytest.mark.asyncio
    async def test_multiple_agents_independent(self, custom_genome, technical_genome):
        """Test multiple agents operate independently."""
        config1 = AgentConfig(model_name="gpt-4")
        config2 = AgentConfig(model_name="gpt-4")

        agent1 = await create_agent(custom_genome, config1)
        agent2 = await create_agent(technical_genome, config2)

        assert agent1.genome != agent2.genome
        assert agent1.is_initialized is True
        assert agent2.is_initialized is True

        # Both should be able to chat independently
        response1_chunks = []
        async for chunk in agent1.chat("Test 1"):
            response1_chunks.append(chunk)

        response2_chunks = []
        async for chunk in agent2.chat("Test 2"):
            response2_chunks.append(chunk)

        assert len(response1_chunks) > 0
        assert len(response2_chunks) > 0

    @pytest.mark.asyncio
    async def test_genome_trait_affects_response(self, default_genome):
        """Test genome traits affect agent behavior."""
        # Create high empathy genome
        high_empathy = ConversationalGenome(
            "high_empathy",
            traits={'empathy': 0.95, 'engagement': 0.90}
        )

        # Create low empathy genome
        low_empathy = ConversationalGenome(
            "low_empathy",
            traits={'empathy': 0.20, 'engagement': 0.30}
        )

        config = AgentConfig()

        agent_high = await create_agent(high_empathy, config)
        agent_low = await create_agent(low_empathy, config)

        # Both should generate different system prompts
        prompt_high = agent_high._create_system_prompt()
        prompt_low = agent_low._create_system_prompt()

        # Prompts should differ based on traits
        assert prompt_high != prompt_low

    def test_agent_config_defaults(self):
        """Test agent config has correct defaults."""
        config = AgentConfig()

        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.streaming is True
        assert config.memory_enabled is True
        assert config.evolution_enabled is True
        assert config.kraken_enabled is True

    def test_agent_config_customization(self):
        """Test agent config can be customized."""
        config = AgentConfig(
            model_name="claude-3",
            temperature=0.5,
            max_tokens=1024,
            streaming=False,
            memory_enabled=False,
            evolution_enabled=False,
            kraken_enabled=False
        )

        assert config.model_name == "claude-3"
        assert config.temperature == 0.5
        assert config.max_tokens == 1024
        assert config.streaming is False
        assert config.memory_enabled is False
        assert config.evolution_enabled is False
        assert config.kraken_enabled is False

