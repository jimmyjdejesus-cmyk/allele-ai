"""
NLP Agent creation and management for Allele.

This module provides high-level agent creation using conversational genomes.

Author: Bravetto AI Systems
Version: 1.0.0
"""

from typing import Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass

from .genome import ConversationalGenome
from .kraken_lnn import KrakenLNN
from .types import AgentResponse, ConversationTurn
from .exceptions import AgentError


@dataclass
class AgentConfig:
    """Configuration for NLP agents.

    Attributes:
        model_name: Name of the LLM model to use (e.g., "gpt-4", "claude-3")
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        streaming: Whether to enable streaming responses
        memory_enabled: Whether to enable conversation memory
        evolution_enabled: Whether to enable evolutionary adaptation
        kraken_enabled: Whether to use Kraken LNN processing
    """
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048
    streaming: bool = True
    memory_enabled: bool = True
    evolution_enabled: bool = True
    kraken_enabled: bool = True


class NLPAgent:
    """NLP Agent powered by conversational genome.

    Example:
        >>> genome = ConversationalGenome("agent_001", {'empathy': 0.9})
        >>> config = AgentConfig(model_name="gpt-4")
        >>> agent = NLPAgent(genome, config)
        >>> await agent.initialize()
        >>> async for response in agent.chat("Hello!"):
        ...     print(response)
    """

    def __init__(
        self,
        genome: ConversationalGenome,
        config: AgentConfig
    ):
        """Initialize NLP agent.

        Args:
            genome: Conversational genome defining agent personality
            config: Agent configuration
        """
        self.genome = genome
        self.config = config
        self.conversation_history: list[ConversationTurn] = []
        self.kraken_lnn: Optional[KrakenLNN] = None
        self.is_initialized = False

        # Initialize Kraken LNN if enabled
        if config.kraken_enabled:
            self.kraken_lnn = KrakenLNN(reservoir_size=100)

    async def initialize(self) -> bool:
        """Initialize the agent.

        Returns:
            True if initialization successful
        """
        # Placeholder for LLM client initialization
        # In actual implementation, this would initialize OpenAI/Anthropic/etc client
        self.is_initialized = True
        return True

    async def chat(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Chat with the agent.

        Args:
            message: User message
            context: Optional context dictionary

        Yields:
            Response chunks if streaming enabled, else full response

        Raises:
            AgentError: If agent not initialized
        """
        if not self.is_initialized:
            raise AgentError("Agent not initialized. Call initialize() first.")

        # Create system prompt based on genome
        system_prompt = self._create_system_prompt()

        # Placeholder for actual LLM integration
        # In production, this would call OpenAI/Anthropic/etc API
        response_text = f"[Agent {self.genome.genome_id}] Response based on traits: {self.genome.traits}"

        # Process through Kraken if enabled
        if self.kraken_lnn:
            # Convert text to sequence for processing
            sequence = [float(ord(c)) / 255.0 for c in message[:10]]
            result = await self.kraken_lnn.process_sequence(sequence)

        yield response_text

    def _create_system_prompt(self) -> str:
        """Create system prompt from genome traits.

        Returns:
            System prompt string
        """
        traits = self.genome.traits
        prompt_parts = [
            "You are an AI assistant with the following characteristics:",
            "",
            f"🧠 Empathy: {traits['empathy']:.1f}/1.0",
            f"💬 Engagement: {traits['engagement']:.1f}/1.0",
            f"🔬 Technical Knowledge: {traits['technical_knowledge']:.1f}/1.0",
            f"🎨 Creativity: {traits['creativity']:.1f}/1.0",
            f"📝 Conciseness: {traits['conciseness']:.1f}/1.0",
            f"🧩 Context Awareness: {traits['context_awareness']:.1f}/1.0",
            "",
            "Adapt your responses according to these trait levels."
        ]

        return "\n".join(prompt_parts)


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

