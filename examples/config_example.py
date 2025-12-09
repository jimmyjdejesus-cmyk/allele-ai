#!/usr/bin/env python3
"""
Configuration Usage Example for Allele.

This example demonstrates different ways to use the central configuration system:
- Reading from central settings
- Using from_settings() factory methods
- Environment variable overrides
- Programmatic configuration
"""

import asyncio
import os
from allele import (
    settings,
    AgentConfig,
    EvolutionConfig,
    EvolutionEngine,
    ConversationalGenome,
    KrakenLNN,
    create_agent
)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


async def main():
    """Run configuration examples."""
    print("🔧 Allele Configuration System Examples\n")
    
    # ==========================================================================
    # Example 1: Reading Central Settings
    # ==========================================================================
    print_section("1. Reading Central Settings")
    
    print("Agent Settings:")
    print(f"  Model: {settings.agent.model_name}")
    print(f"  Temperature: {settings.agent.temperature}")
    print(f"  Max Tokens: {settings.agent.max_tokens}")
    print(f"  Kraken Enabled: {settings.agent.kraken_enabled}")
    
    print("\nEvolution Settings:")
    print(f"  Population Size: {settings.evolution.population_size}")
    print(f"  Generations: {settings.evolution.generations}")
    print(f"  Mutation Rate: {settings.evolution.mutation_rate}")
    print(f"  Elitism: {settings.evolution.elitism_enabled}")
    
    print("\nKraken LNN Settings:")
    print(f"  Reservoir Size: {settings.kraken.reservoir_size}")
    print(f"  Connectivity: {settings.kraken.connectivity}")
    print(f"  Memory Buffer: {settings.kraken.memory_buffer_size}")
    
    print("\nLiquid Dynamics Settings:")
    print(f"  Viscosity: {settings.liquid_dynamics.viscosity}")
    print(f"  Temperature: {settings.liquid_dynamics.temperature}")
    print(f"  Pressure: {settings.liquid_dynamics.pressure}")
    
    print("\nDefault Traits:")
    for trait, value in settings.default_traits.items():
        print(f"  {trait}: {value}")
    
    # ==========================================================================
    # Example 2: Using from_settings() Factories
    # ==========================================================================
    print_section("2. Using from_settings() Factory Methods")
    
    # Create configs from central settings
    agent_config = AgentConfig.from_settings()
    evolution_config = EvolutionConfig.from_settings()
    
    print("AgentConfig created from settings:")
    print(f"  Model: {agent_config.model_name}")
    print(f"  Temperature: {agent_config.temperature}")
    
    print("\nEvolutionConfig created from settings:")
    print(f"  Population: {evolution_config.population_size}")
    print(f"  Generations: {evolution_config.generations}")
    
    # Create genome using settings defaults
    genome = ConversationalGenome.from_settings("example_agent")
    print(f"\nGenome created with default traits:")
    print(f"  Genome ID: {genome.genome_id}")
    print(f"  Empathy: {genome.traits['empathy']}")
    print(f"  Technical Knowledge: {genome.traits['technical_knowledge']}")
    
    # Create Kraken from settings
    kraken = KrakenLNN.from_settings()
    print(f"\nKraken LNN created from settings:")
    print(f"  Reservoir Size: {kraken.reservoir_size}")
    print(f"  Connectivity: {kraken.connectivity}")
    
    # ==========================================================================
    # Example 3: Environment Variable Overrides
    # ==========================================================================
    print_section("3. Environment Variable Overrides")
    
    # Show how to set environment variables
    print("To override settings via environment variables:")
    print("\nBash/Linux/Mac:")
    print("  export AGENT__MODEL_NAME='gpt-4-turbo'")
    print("  export AGENT__TEMPERATURE='0.9'")
    print("  export EVOLUTION__POPULATION_SIZE='200'")
    
    print("\nPowerShell (Windows):")
    print("  $env:AGENT__MODEL_NAME = 'gpt-4-turbo'")
    print("  $env:AGENT__TEMPERATURE = '0.9'")
    print("  $env:EVOLUTION__POPULATION_SIZE = '200'")
    
    print("\nOr create a .env file:")
    print("  AGENT__MODEL_NAME=gpt-4-turbo")
    print("  AGENT__TEMPERATURE=0.9")
    print("  EVOLUTION__POPULATION_SIZE=200")
    
    # Check if any env vars are set
    env_vars_set = []
    check_vars = [
        'AGENT__MODEL_NAME',
        'AGENT__TEMPERATURE',
        'EVOLUTION__POPULATION_SIZE',
        'EVOLUTION__IMMUTABLE_EVOLUTION',
        'EVOLUTION__HPC_MODE',
        'KRAKEN__RESERVOIR_SIZE'
    ]
    
    print("\nCurrently set environment variables:")
    for var in check_vars:
        value = os.getenv(var)
        if value:
            env_vars_set.append((var, value))
            print(f"  {var} = {value}")
    
    if not env_vars_set:
        print("  (None detected - using defaults)")
    
    # ==========================================================================
    # Example 4: Programmatic Configuration
    # ==========================================================================
    print_section("4. Programmatic Configuration")
    
    # Create custom settings instance
    from allele.config import AlleleSettings, AgentSettings, EvolutionSettings
    
    custom_settings = AlleleSettings(
        agent=AgentSettings(
            model_name="custom-model",
            temperature=0.95,
            max_tokens=4096
        ),
        evolution=EvolutionSettings(
            population_size=50,
            generations=20,
            mutation_rate=0.2
        )
    )
    
    print("Created custom settings instance:")
    print(f"  Agent Model: {custom_settings.agent.model_name}")
    print(f"  Agent Temperature: {custom_settings.agent.temperature}")
    print(f"  Evolution Population: {custom_settings.evolution.population_size}")
    
    # Use custom settings with configs
    custom_agent_config = AgentConfig.from_settings(custom_settings)
    print(f"\nAgentConfig from custom settings:")
    print(f"  Model: {custom_agent_config.model_name}")
    print(f"  Temperature: {custom_agent_config.temperature}")

    # Demonstrate immutable vs in-place behavior
    print_section("4b. Mutation Strategy Examples")
    # HPC/in-place (default)
    engine_hpc = EvolutionEngine(EvolutionConfig.from_settings())
    print(f"  HPC mode (in-place): {engine_hpc.hpc_mode}, immutable: {engine_hpc.immutable_evolution}")
    # Immutable
    engine_immutable = EvolutionEngine(EvolutionConfig(immutable_evolution=True, hpc_mode=False))
    print(f"  Immutable mode: {engine_immutable.immutable_evolution}, hpc: {engine_immutable.hpc_mode}")
    
    # ==========================================================================
    # Example 5: Hybrid Approach
    # ==========================================================================
    print_section("5. Hybrid Approach (Settings + Overrides)")
    
    # Start with settings, override specific values
    base_config = AgentConfig.from_settings()
    
    hybrid_config = AgentConfig(
        model_name=base_config.model_name,  # From settings
        temperature=0.95,  # Override
        max_tokens=base_config.max_tokens,  # From settings
        streaming=True,
        memory_enabled=base_config.memory_enabled,
        evolution_enabled=False,  # Override
        kraken_enabled=base_config.kraken_enabled
    )
    
    print("Hybrid configuration (settings + overrides):")
    print(f"  Model (from settings): {hybrid_config.model_name}")
    print(f"  Temperature (override): {hybrid_config.temperature}")
    print(f"  Evolution (override): {hybrid_config.evolution_enabled}")
    
    # ==========================================================================
    # Example 6: Using Configured Agent
    # ==========================================================================
    print_section("6. Using Configured Agent")
    
    # Create genome and agent using settings
    genome = ConversationalGenome.from_settings(
        "demo_agent",
        traits={'empathy': 0.9, 'technical_knowledge': 0.85}
    )
    
    config = AgentConfig.from_settings()
    agent = await create_agent(genome, config)
    
    print(f"Agent created with settings-based configuration:")
    print(f"  Genome ID: {agent.genome.genome_id}")
    print(f"  Model: {config.model_name}")
    print(f"  Initialized: {agent.is_initialized}")
    
    # Simple interaction
    print(f"\nAgent response:")
    async for chunk in agent.chat("Hello! Explain your configuration."):
        print(f"  {chunk}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print_section("Summary")
    
    print("Configuration Options:")
    print("\n1. Hardcoded Defaults:")
    print("   config = AgentConfig(model_name='gpt-4', temperature=0.7)")
    print("   ✓ Simple, explicit")
    print("   ✗ Config scattered, hard to change")
    
    print("\n2. Central Settings (Recommended):")
    print("   config = AgentConfig.from_settings()")
    print("   ✓ Centralized, type-safe")
    print("   ✓ Environment variable support")
    print("   ✓ Easy to test with different configs")
    
    print("\n3. .env Files:")
    print("   Create .env file, use from_settings()")
    print("   ✓ No code changes for environments")
    print("   ✓ Standard deployment practice")
    print("   ✓ Git-ignored for security")
    
    print("\n4. Hybrid:")
    print("   base = AgentConfig.from_settings()")
    print("   config = AgentConfig(..., temperature=0.9)")
    print("   ✓ Flexible for edge cases")
    print("   ✓ Override only what's needed")
    
    print("\n" + "="*60)
    print("\n✅ Configuration examples completed!")
    print("\nNext steps:")
    print("  - Copy .env.example to .env and customize")
    print("  - See docs/CONFIGURATION.md for full guide")
    print("  - Try setting AGENT__MODEL_NAME environment variable")


if __name__ == "__main__":
    asyncio.run(main())
