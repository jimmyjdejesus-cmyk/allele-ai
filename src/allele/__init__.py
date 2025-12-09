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

"""Allele: Genome-Based Conversational AI SDK

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
License: AGPL-3.0
"""

__version__ = "1.0.0"
__author__ = "Jimmy De Jesus & Bravetto AI Systems"
__license__ = "AGPL-3.0"

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
from .config import settings as settings

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
