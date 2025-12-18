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

import importlib
from typing import Any, Dict

# Alias used to import a BaseSettings implementation (pydantic v2/v1)
PydanticBaseSettingsImpl: Any

try:
    # Pydantic v1 or v2 - use BaseModel for compatibility
    from pydantic import BaseModel, Field
except Exception:
    # Fallback (unlikely to be needed)
    from pydantic import BaseModel, Field

try:
    # Pydantic v2 settings
    from pydantic_settings import BaseSettings as _ImportedPydanticBaseSettingsV2
    from pydantic_settings import SettingsConfigDict
    PydanticBaseSettingsImpl = _ImportedPydanticBaseSettingsV2
    _HAS_SETTINGS = True
except Exception:
    # Fallback to pydantic v1 BaseSettings if available
    try:
        from pydantic import BaseSettings as _ImportedPydanticBaseSettingsV1

        PydanticBaseSettingsImpl = _ImportedPydanticBaseSettingsV1
        _HAS_SETTINGS = True
    except Exception:
        _HAS_SETTINGS = False
        PydanticBaseSettingsImpl = None

# Central configuration definitions using pydantic

DEFAULT_TRAITS: Dict[str, float] = {
    "empathy": 0.5,
    "engagement": 0.5,
    "technical_knowledge": 0.5,
    "creativity": 0.5,
    "conciseness": 0.5,
    "context_awareness": 0.5,
    "adaptability": 0.5,
    "personability": 0.5,
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


# Create a single runtime BaseSettingsImpl class that subclasses the
# available pydantic BaseSettings implementation when present, otherwise
# fall back to BaseModel. Defining it once avoids duplicate-definition
# errors from mypy while preserving runtime behavior.

_PydanticRuntimeBase: type
if importlib.util.find_spec("pydantic_settings") is not None:
    _mod = importlib.import_module("pydantic_settings")
    _PydanticRuntimeBase = _mod.BaseSettings
elif importlib.util.find_spec("pydantic") is not None:
    _mod = importlib.import_module("pydantic")
    _PydanticRuntimeBase = _mod.BaseSettings
else:
    _PydanticRuntimeBase = BaseModel


class AlleleSettings(BaseModel):
    """Application settings loaded from environment variables or .env files.

    Naming convention for env vars is uppercase with underscores; nested fields
    will be prefixed (e.g., AGENT__MODEL_NAME, EVOLUTION__POPULATION_SIZE).
    """

    agent: AgentSettings = AgentSettings(model_name="gpt-4")
    evolution: EvolutionSettings = EvolutionSettings()
    kraken: KrakenSettings = KrakenSettings()
    liquid_dynamics: LiquidDynamicsSettings = LiquidDynamicsSettings()
    default_traits: Dict[str, float] = DEFAULT_TRAITS

    if _HAS_SETTINGS and "SettingsConfigDict" in globals():
        # Attach pydantic v2 model_config dynamically when available so
        # environment variable names like AGENT__MODEL_NAME are supported.
        model_config = SettingsConfigDict(
            env_nested_delimiter="__",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )
    else:

        class Config:
            env_nested_delimiter = "__"
            env_file = ".env"
            env_file_encoding = "utf-8"

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """If a pydantic BaseSettings implementation is available at
        runtime, instantiate and return a BaseSettings-backed instance so
        environment variable overrides are respected when callers use
        ``AlleleSettings()`` directly in tests or runtime code.
        """
        if PydanticBaseSettingsImpl is not None and cls is AlleleSettings:
            # Build a runtime class dict without the `__new__` hook to avoid
            # recursive instantiation issues when the runtime class is
            # created from the BaseModel-defined class.
            class_dict = dict(cls.__dict__)
            class_dict.pop("__new__", None)
            class_dict.pop("__classcell__", None)
            RuntimeCls = type("AlleleSettingsRuntime", (PydanticBaseSettingsImpl,), class_dict)
            # Construct and return a pydantic-backed instance which will
            # automatically load from environment variables.
            return RuntimeCls(*args, **kwargs)

        return super().__new__(cls)


# Singleton instance to use across the package
if PydanticBaseSettingsImpl is not None:
    try:
        # Create a runtime subclass of the pydantic BaseSettings implementation
        # that contains the same attributes as our BaseModel-defined
        # `AlleleSettings`. We instantiate that class so pydantic's env-var
        # parsing behavior is used at runtime while preserving static typing.
        # Use the BaseSettings implementation directly if it exists; fall back
        # to BaseModel behavior when constructing the runtime class. We avoid
        # referencing internal temporary names to keep mypy happy.
        AlleleSettingsRuntime = type("AlleleSettingsRuntime", (PydanticBaseSettingsImpl,), dict(AlleleSettings.__dict__))
        settings = AlleleSettingsRuntime()
    except Exception:
        settings = AlleleSettings()
else:
    settings = AlleleSettings()
