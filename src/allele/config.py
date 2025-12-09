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

from typing import Dict, Optional
try:
    # Pydantic v1 still exposes BaseSettings; in v2 it is moved to pydantic-settings
    from pydantic import BaseSettings as _BaseSettings, BaseModel, Field
except Exception:
    # Fallback for pydantic v2 where BaseSettings comes from pydantic-settings
    from pydantic import BaseModel, Field
    try:
        from pydantic_settings import BaseSettings as _BaseSettings
    except Exception:
        # Last resort: use BaseModel as a simple replacement (no env loading)
        _BaseSettings = BaseModel

try:
    # Pydantic v2 uses ConfigDict
    from pydantic import ConfigDict  # type: ignore
    _HAS_CONFIGDICT = True
except Exception:
    _HAS_CONFIGDICT = False

# Central configuration definitions using pydantic

DEFAULT_TRAITS: Dict[str, float] = {
    'empathy': 0.5,
    'engagement': 0.5,
    'technical_knowledge': 0.5,
    'creativity': 0.5,
    'conciseness': 0.5,
    'context_awareness': 0.5,
    'adaptability': 0.5,
    'personability': 0.5
}

class AgentSettings(BaseModel):
    model_name: str = Field("gpt-4", description="Name of the LLM to use")
    temperature: float = 0.7
    max_tokens: int = 2048
    streaming: bool = True
    memory_enabled: bool = True
    evolution_enabled: bool = True
    kraken_enabled: bool = True

class EvolutionSettings(BaseModel):
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_pressure: float = 0.2
    elitism_enabled: bool = True
    tournament_size: int = 3
    # Controls whether evolution modifies objects in-place (low memory, faster)
    # or creates new Genome instances each generation (immutable, functional).
    immutable_evolution: bool = False
    # Convenience flag that enables HPC-oriented optimizations (reduced logging,
    # in-place mutation, etc.). Defaults to True for high performance.
    hpc_mode: bool = True
    # If you set immutable_evolution=True, it's recommended to also set
    # hpc_mode=False to avoid conflicting intentions. However the engine will
    # respect the immutable flag first and fall back to the hpc_mode setting
    # only when immutable_evolution is False.

class KrakenSettings(BaseModel):
    reservoir_size: int = 100
    connectivity: float = 0.1
    memory_buffer_size: int = 1000

class LiquidDynamicsSettings(BaseModel):
    viscosity: float = 0.1
    temperature: float = 1.0
    pressure: float = 1.0
    flow_rate: float = 0.5
    turbulence: float = 0.05

class AlleleSettings(_BaseSettings):
    """Application settings loaded from environment variables or .env files.

    Naming convention for env vars is uppercase with underscores; nested fields
    will be prefixed (e.g., AGENT_MODEL_NAME, EVOLUTION_POPULATION_SIZE).
    """

    agent: AgentSettings = AgentSettings()
    evolution: EvolutionSettings = EvolutionSettings()
    kraken: KrakenSettings = KrakenSettings()
    liquid_dynamics: LiquidDynamicsSettings = LiquidDynamicsSettings()
    default_traits: Dict[str, float] = DEFAULT_TRAITS

    if _HAS_CONFIGDICT:
        model_config = ConfigDict(env_nested_delimiter="__")
    else:
        class Config:
            env_nested_delimiter = "__"

# Singleton instance to use across the package
settings = AlleleSettings()

