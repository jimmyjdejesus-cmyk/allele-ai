"""
Allele: Genome-Based Conversational AI SDK

A production-ready SDK for creating genome-based conversational AI agents with
8 evolved traits, Kraken Liquid Neural Networks, and evolutionary optimization.

Official Python library for the Allele platform.

Example:
    >>> from allele import ConversationalGenome, create_agent
    >>> 
    >>> # Create genome with desired traits
    >>> genome = ConversationalGenome(
    ...     genome_id="my_agent",
    ...     traits={
    ...         'empathy': 0.9,
    ...         'technical_knowledge': 0.95,
    ...         'creativity': 0.7
    ...     }
    ... )
    >>> 
    >>> # Create agent
    >>> agent = await create_agent(genome, model="gpt-4")
    >>> 
    >>> # Start conversation
    >>> async for response in agent.chat("Explain quantum computing"):
    ...     print(response, end='')

Author: Bravetto AI Systems
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.1"
__author__ = "Jimmy De Jesus"
__license__ = "MIT"

# Core genome classes
from .genome import (
    ConversationalGenome,
    Gene,
    GenomeBase,
)

# Neural network components
from .kraken_lnn import (
    KrakenLNN,
    LiquidStateMachine,
    LiquidDynamics,
    AdaptiveWeightMatrix,
    TemporalMemoryBuffer,
)

# Evolution engine
from .evolution import (
    EvolutionEngine,
    EvolutionConfig,
    GeneticOperators,
)

# Agent creation and management
from .agent import (
    NLPAgent,
    create_agent,
    AgentConfig,
)

# Type definitions and exceptions
from .types import (
    TraitDict,
    ConversationTurn,
    AgentResponse,
)

from .exceptions import (
    AbeNLPError,
    GenomeError,
    EvolutionError,
    AgentError,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    
    # Genome classes
    "ConversationalGenome",
    "Gene",
    "GenomeBase",
    
    # Neural network
    "KrakenLNN",
    "LiquidStateMachine",
    "LiquidDynamics",
    "AdaptiveWeightMatrix",
    "TemporalMemoryBuffer",
    
    # Evolution
    "EvolutionEngine",
    "EvolutionConfig",
    "GeneticOperators",
    
    # Agent
    "NLPAgent",
    "create_agent",
    "AgentConfig",
    
    # Types
    "TraitDict",
    "ConversationTurn",
    "AgentResponse",
    
    # Exceptions
    "AbeNLPError",
    "GenomeError",
    "EvolutionError",
    "AgentError",
]

