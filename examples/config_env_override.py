"""
Example: Using central settings and environment overrides

This example shows two ways to configure: environment variables and programmatic override
"""

import os

from allele import settings, AgentConfig, EvolutionConfig, ConversationalGenome, KrakenLNN


def main():
    print("Default agent model:", settings.agent.model_name)
    print("Default evolution population size:", settings.evolution.population_size)

    print("\nCreate agent with default settings from central config")
    agent_cfg = AgentConfig.from_settings()
    print("  Agent model from AgentConfig:", agent_cfg.model_name)

    print("\nOverride via environment variable (AGENT__MODEL_NAME)")
    os.environ['AGENT__MODEL_NAME'] = 'environment-gpt'
    # Recreate runtime settings so pydantic BaseSettings picks up the new env var
    from allele.config import AlleleSettings
    new_settings = AlleleSettings()

    agent_cfg_env = AgentConfig.from_settings(new_settings)
    print("  Agent model from AgentConfig with env override:", agent_cfg_env.model_name)

    # Programmatic override
    print("\nProgrammatic override: create custom AlleleSettings instance")
    custom_settings = AlleleSettings()
    custom_settings.agent.model_name = 'custom-gpt'

    agent_cfg_custom = AgentConfig.from_settings(custom_settings)
    print("  Agent model from AgentConfig custom settings:", agent_cfg_custom.model_name)

    # Genome default traits
    g = ConversationalGenome.from_settings("example_settings_genome")
    print("\nGenome default traits used:", g.traits)

    # Kraken LNN from settings
    k = KrakenLNN.from_settings()
    print("\nKraken reservoir size:", k.reservoir_size)


if __name__ == '__main__':
    main()
