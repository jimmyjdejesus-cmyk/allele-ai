# Copyright (C) 2025 Phylogenic AI Labs & Jimmy De Jesus
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

"""Allele: Genome-Based Conversational AI SDK

A production-ready SDK for creating genome-based conversational AI agents with
8 evolved traits, Kraken Liquid Neural Networks, and evolutionary optimization.

Official Python library for the Phylogenic platform.

Example:
    >>> from phylogenic import ConversationalGenome, create_agent
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
License: AGPL-3.0
"""

__version__ = "1.0.1"
__author__ = "Jimmy De Jesus & Bravetto AI Systems"
__license__ = "AGPL-3.0"

# Core genome classes
# Agent creation and management
from .agent import (
    AgentConfig,
    NLPAgent,
    create_agent,
)
from .config import settings as settings

# Evolution engine
from .evolution import (
    EvolutionConfig,
    EvolutionEngine,
    GeneticOperators,
)
from .exceptions import (
    AbeNLPError,
    AgentError,
    EvolutionError,
    GenomeError,
)
from .genome import (
    ConversationalGenome,
    Gene,
    GenomeBase,
)

# Neural network components
from .kraken_lnn import (
    AdaptiveWeightMatrix,
    KrakenLNN,
    LiquidDynamics,
    LiquidStateMachine,
    TemporalMemoryBuffer,
)

# Type definitions and exceptions
from .types import (
    AgentResponse,
    ConversationTurn,
    TraitDict,
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
    "settings",
]
