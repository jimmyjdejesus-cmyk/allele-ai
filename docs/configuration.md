# Configuration Guide

Allele provides a flexible, centralized configuration system using **Pydantic Settings** that supports:
- Code-based defaults
- Environment variable overrides
- `.env` file loading
- Type validation and IDE autocomplete

---

## Quick Start

### Using Default Configuration

The simplest approach is to use the built-in defaults:

```python
from allele import ConversationalGenome, create_agent, AgentConfig

# Uses default settings automatically
genome = ConversationalGenome("agent_001")
config = AgentConfig()  # Uses hardcoded defaults
agent = await create_agent(genome, config)
```

### Using Central Settings

Load configuration from the central settings singleton:

```python
from allele import settings, AgentConfig, EvolutionConfig

# Inspect current settings
print(settings.agent.model_name)  # "gpt-4"
print(settings.evolution.population_size)  # 100

# Create configs from settings
agent_config = AgentConfig.from_settings()
evolution_config = EvolutionConfig.from_settings()
```

### Environment Variable Overrides

Set environment variables to override defaults:

```bash
# Bash/Linux/Mac
export AGENT__MODEL_NAME="gpt-4-turbo"
export AGENT__TEMPERATURE="0.9"
export EVOLUTION__POPULATION_SIZE="200"
export KRAKEN__RESERVOIR_SIZE="150"

# PowerShell (Windows)
$env:AGENT__MODEL_NAME = "gpt-4-turbo"
$env:AGENT__TEMPERATURE = "0.9"
$env:EVOLUTION__POPULATION_SIZE = "200"
```

**Note:** Use double underscores (`__`) to separate nested settings (e.g., `AGENT__MODEL_NAME`).

### Create a genome using settings default traits

```python
from allele import ConversationalGenome

genome = ConversationalGenome.from_settings('example_id')
```

### Create Kraken LNN from central settings

```python
from allele import KrakenLNN

kraken = KrakenLNN.from_settings()
```

---

## Override via environment variables

You can override nested keys using environment variables separated with `__` (double underscore) as a nested delimiter:

- Set agent model: `AGENT__MODEL_NAME` — e.g. `AGENT__MODEL_NAME=gpt-4` 
- Set population size: `EVOLUTION__POPULATION_SIZE=10`
- Set Kraken reservoir size: `KRAKEN__RESERVOIR_SIZE=200`

Example (Linux / macOS):

```bash
export AGENT__MODEL_NAME=test-model-env
export EVOLUTION__POPULATION_SIZE=10
python examples/config_env_override.py
```

Example (PowerShell on Windows):

```powershell
$env:AGENT__MODEL_NAME = 'test-model-env'
$env:EVOLUTION__POPULATION_SIZE = '10'
python .\examples\config_env_override.py
```

---

## Programmatic override

You can create a new `AlleleSettings` instance with overrides if you want a separate settings object and avoid mutating the singleton `allele.settings`:

```python
from allele.config import AlleleSettings, AgentSettings

custom_settings = AlleleSettings(agent=AgentSettings(model_name='my-model'))
agent_cfg = AgentConfig.from_settings(custom_settings)
```

This pattern is useful for per-run/sandboxed configurations.

---

## Notes and considerations

- Pydantic v2 introduces `pydantic-settings` and `ConfigDict`. The `config.py` uses compatibility logic so the library functions with both pydantic v1 and v2.
- We expose `settings` from `allele` package for convenience. If you want to centralize all runtime behavior in an application, create a single settings instance early and use the `from_settings(...)` factory functions to populate the dataclasses.
- If you use environment overrides, recreate the `AlleleSettings()` object (e.g., `AlleleSettings()` or restart the process) to pick up new environment variables.

---

If you'd like, we can also add a small `config` example file with `.env` or show programmatic merging of settings for different environments (e.g., development/prod).

---

## Options and tradeoffs

When designing configuration for libraries and applications, there are multiple approaches. Below we summarize what we implemented here (pydantic-based central settings) and alternatives with tradeoffs.

- Central pydantic-based settings (current approach):
	- Pros:
		- Strong typing, default values, validation integrated and convenient.
		- Supports environment variables and `.env` override semantics via BaseSettings.
		- `from_settings()` helpers make it easy to programmatically convert settings into dataclass configs used by the library.
	- Cons:
		- Adds `pydantic` dependency and subtle behavior changes between pydantic v1/v2 (handled via a compat shim).
		- If runtime mutability of `settings` is required, it is possible but usually not advisable; prefer creating overrides via `AlleleSettings(...)` objects.

- Per-class dataclass defaults (legacy approach):
	- Pros:
		- No dependency on an external library, defaults are plain Python dataclasses or module-level constants.
		- Simpler mental model for library authors.
	- Cons:
		- No built-in environment variable support; adding that requires code changes or a third-party helper.
		- Merging runtime overrides and consistent behavior across components becomes ad-hoc.

- YAML/JSON-based config files (external config file):
	- Pros:
		- Explicit file that can be part of deployment, easy to version control and manage different environment configs.
		- Structured and programmatic merging possible.
	- Cons:
		- Extra code to load and validate files and to map to dataclasses.
		- Might be redundant with environment variables in cloud deployments.

- Environment-only configuration (no central object):
	- Pros:
		- Declarative; integrates well with containerized deployments and 12-factor app patterns.
	- Cons:
		- Scattered access and no centralized documentation of defaults; harder for library users to discover default values.

Which to pick?

- For libraries (like Allele), a hybrid is often best: keep dataclass defaults for stable, documented behavior and provide a central settings object for application-level overrides. In this repository we expose both: dataclasses (AgentConfig/EvolutionConfig) and the central `settings` with `from_settings` helpers.

- For apps, a central pydantic-based settings object is recommended (as implemented), as it merges environment config, file-based overrides (if needed via `python-dotenv`), and typed validation.

Performance & immutability tradeoffs

- `AlleleSettings` is intended to be a central, read-only source of default values. If you mutate it in runtime you may cause inconsistent behavior; prefer creating new `AlleleSettings` objects for scoped overrides.
- The `EvolutionEngine` has historically created new objects for evolved genomes. For integration tests we opted to mutate genomes in-place during evolution so references to genome objects that existed before evolution can reflect changes. If you prefer immutable flow, you should switch to returning a new population list and update any code or tests accordingly.

---

## HPC Mode & Mutation Strategy ⚡️

For high-performance (HPC) use cases, Allele defaults to an in-place mutation strategy which reduces memory pressure and improves speed by reusing existing genome objects. This is most suitable for long-running, memory-sensitive workloads.

- Default behavior: `hpc_mode=True`, `immutable_evolution=False` (in-place mutation)
- To use immutable evolution for reproducible, functional-style behavior, set `EVOLUTION__IMMUTABLE_EVOLUTION=true` (or programmatically via `EvolutionConfig`).

Examples:

```python
from allele import EvolutionConfig, EvolutionEngine
# HPC mode (default)
cfg = EvolutionConfig.from_settings()
assert cfg.hpc_mode and not cfg.immutable_evolution

# Immutable flow
cfg2 = EvolutionConfig(immutable_evolution=True, hpc_mode=False)
engine2 = EvolutionEngine(cfg2)
```

When `immutable_evolution` is enabled, the engine will allocate new `ConversationalGenome` objects for offspring rather than modifying parents in-place. This makes debugging and reproducibility easier but will increase memory usage.

Need help choosing?

I can adapt the code for your preferred pattern:
	- Convert `EvolutionEngine` to purely immutable (new object creation) and update tests accordingly.
	- Adjust `ConversationalGenome` to only accept settings at construction time and remain immutable afterward.
	- Add a YAML loader to complement env var overrides.

Tell me which behavior you want (e.g. immutable vs in-place mutation, environment-only or programmatic overrides), and I will implement and document it fully.