from phylogenic import settings
from phylogenic.agent import AgentConfig
from phylogenic.evolution import EvolutionConfig


def test_settings_load_defaults():
    assert settings is not None
    # Ensure that nested settings are accessible
    assert settings.agent.model_name == "gpt-4"
    assert settings.evolution.population_size == 100


def test_agentconfig_from_settings():
    cfg = AgentConfig.from_settings()
    assert cfg.model_name == settings.agent.model_name
    assert cfg.kraken_enabled == settings.agent.kraken_enabled


def test_evolutionconfig_from_settings():
    cfg = EvolutionConfig.from_settings()
    assert cfg.population_size == settings.evolution.population_size
    assert cfg.generations == settings.evolution.generations
