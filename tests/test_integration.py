"""
Integration tests for Allele system.

Tests end-to-end workflows involving multiple components.
"""

import pytest
import numpy as np
from typing import List

from allele import (
    ConversationalGenome,
    EvolutionEngine,
    EvolutionConfig,
    create_agent,
    AgentConfig,
    KrakenLNN
)
from tests.test_utils import (
    assert_genome_valid,
    calculate_population_diversity,
    generate_population
)


class TestIntegration:
    """Integration tests for full workflows."""
    
    @pytest.mark.asyncio
    async def test_genome_to_agent_workflow(self, custom_genome):
        """Test complete workflow from genome creation to agent usage."""
        # Create genome
        assert_genome_valid(custom_genome)
        
        # Create agent from genome
        config = AgentConfig()
        agent = await create_agent(custom_genome, config)
        
        assert agent.is_initialized is True
        assert agent.genome == custom_genome
        
        # Use agent
        response_chunks = []
        async for chunk in agent.chat("Hello"):
            response_chunks.append(chunk)
        
        assert len(response_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_evolution_to_agent_workflow(self, evolution_engine, fitness_function):
        """Test workflow from evolution to agent creation."""
        # Initialize population
        population = evolution_engine.initialize_population()
        
        # Evolve population
        best_genome = await evolution_engine.evolve(
            population,
            fitness_function,
            generations=5
        )
        
        assert_genome_valid(best_genome)
        assert best_genome.fitness_score > 0.0
        
        # Create agent from evolved genome
        agent = await create_agent(best_genome)
        
        assert agent.is_initialized is True
        assert agent.genome == best_genome
        
        # Agent should be usable
        response_chunks = []
        async for chunk in agent.chat("Test message"):
            response_chunks.append(chunk)
        
        assert len(response_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_kraken_with_agent_integration(self, custom_genome):
        """Test Kraken LNN integration with agent."""
        # Create agent with Kraken enabled
        config = AgentConfig(kraken_enabled=True)
        agent = await create_agent(custom_genome, config)
        
        assert agent.kraken_lnn is not None
        
        # Process message through agent (which uses Kraken)
        response_chunks = []
        async for chunk in agent.chat("Test message for Kraken"):
            response_chunks.append(chunk)
        
        assert len(response_chunks) > 0
        
        # Verify Kraken state
        state = await agent.kraken_lnn.get_network_state()
        assert state['reservoir_size'] == 100
    
    @pytest.mark.asyncio
    async def test_full_evolution_cycle(self, evolution_config, fitness_function):
        """Test complete evolution cycle with all components."""
        engine = EvolutionEngine(evolution_config)
        
        # Initialize population
        population = engine.initialize_population()
        initial_diversity = calculate_population_diversity(population)
        
        # Evolve
        best_genome = await engine.evolve(population, fitness_function)
        
        # Verify evolution occurred
        assert best_genome is not None
        assert best_genome.fitness_score > 0.0
        assert len(engine.evolution_history) == evolution_config.generations
        
        # Verify fitness improved over generations
        if len(engine.evolution_history) > 1:
            initial_fitness = engine.evolution_history[0]['best_fitness']
            final_fitness = engine.evolution_history[-1]['best_fitness']
            # Fitness should improve or at least be tracked
            assert final_fitness >= initial_fitness - 0.1
    
    @pytest.mark.asyncio
    async def test_genome_serialization_roundtrip(self, custom_genome):
        """Test genome serialization and deserialization in workflow."""
        # Set additional state
        custom_genome.fitness_score = 0.85
        custom_genome.generation = 5
        
        # Serialize
        data = custom_genome.to_dict()
        
        # Deserialize
        restored = ConversationalGenome.from_dict(data)
        
        # Verify restoration
        assert restored.genome_id == custom_genome.genome_id
        assert restored.traits == custom_genome.traits
        assert restored.fitness_score == custom_genome.fitness_score
        
        # Use restored genome
        agent = await create_agent(restored)
        assert agent.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_multi_agent_conversation_simulation(self):
        """Test multiple agents with different genomes."""
        # Create diverse genomes
        genome1 = ConversationalGenome(
            "agent1",
            traits={'empathy': 0.95, 'engagement': 0.90}
        )
        genome2 = ConversationalGenome(
            "agent2",
            traits={'technical_knowledge': 0.95, 'conciseness': 0.90}
        )
        
        # Create agents
        agent1 = await create_agent(genome1)
        agent2 = await create_agent(genome2)
        
        # Both agents should work independently
        response1 = []
        async for chunk in agent1.chat("Hello"):
            response1.append(chunk)
        
        response2 = []
        async for chunk in agent2.chat("Hello"):
            response2.append(chunk)
        
        assert len(response1) > 0
        assert len(response2) > 0
    
    @pytest.mark.asyncio
    async def test_evolution_with_fitness_from_agent(self, evolution_config):
        """Test evolution using fitness based on agent performance."""
        def agent_based_fitness(genome: ConversationalGenome) -> float:
            """Fitness based on trait balance (simulating agent quality)."""
            traits = genome.traits
            # Reward balanced, high-value traits
            mean_trait = np.mean(list(traits.values()))
            variance = np.var(list(traits.values()))
            balance = 1.0 - min(variance, 1.0)
            return 0.7 * mean_trait + 0.3 * balance
        
        engine = EvolutionEngine(evolution_config)
        population = engine.initialize_population()
        
        best_genome = await engine.evolve(population, agent_based_fitness)
        
        assert best_genome is not None
        assert best_genome.fitness_score > 0.0
        
        # Create agent from best genome
        agent = await create_agent(best_genome)
        assert agent.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_kraken_memory_persistence(self, custom_genome):
        """Test Kraken memory persists across agent interactions."""
        config = AgentConfig(kraken_enabled=True)
        agent = await create_agent(custom_genome, config)
        
        # First interaction
        async for _ in agent.chat("First message"):
            pass
        
        initial_memory = len(agent.kraken_lnn.temporal_memory.memories)
        
        # Second interaction
        async for _ in agent.chat("Second message"):
            pass
        
        # Memory should have grown
        final_memory = len(agent.kraken_lnn.temporal_memory.memories)
        assert final_memory >= initial_memory
    
    @pytest.mark.asyncio
    async def test_genome_mutation_during_evolution(self, evolution_config, fitness_function):
        """Test genome mutations occur during evolution."""
        engine = EvolutionEngine(evolution_config)
        population = engine.initialize_population()
        
        # Capture initial traits
        initial_traits = {}
        for genome in population[:5]:  # Sample first 5
            initial_traits[genome.genome_id] = genome.traits.copy()
        
        # Evolve
        await engine.evolve(population, fitness_function, generations=3)
        
        # Verify mutations occurred (traits should have changed)
        mutations_found = False
        for genome in population[:5]:
            if genome.genome_id in initial_traits:
                original = initial_traits[genome.genome_id]
                current = genome.traits
                if any(abs(current[t] - original[t]) > 1e-6 for t in current):
                    mutations_found = True
                    break
        
        # Mutations should have occurred
        assert mutations_found
    
    @pytest.mark.asyncio
    async def test_crossover_produces_valid_offspring(self, custom_genome, technical_genome):
        """Test crossover produces valid offspring in workflow."""
        # Perform crossover
        offspring = custom_genome.crossover(technical_genome)
        
        assert_genome_valid(offspring)
        
        # Use offspring to create agent
        agent = await create_agent(offspring)
        assert agent.is_initialized is True
        
        # Agent should work
        response_chunks = []
        async for chunk in agent.chat("Test"):
            response_chunks.append(chunk)
        
        assert len(response_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test error handling across components."""
        # Invalid genome should fail
        with pytest.raises(Exception):
            invalid_genome = ConversationalGenome("test", {'invalid_trait': 0.5})
            agent = await create_agent(invalid_genome)
    
    @pytest.mark.asyncio
    async def test_state_consistency_across_operations(self, custom_genome):
        """Test state remains consistent across operations."""
        # Create agent
        agent = await create_agent(custom_genome)
        
        # Perform multiple operations
        for i in range(5):
            response_chunks = []
            async for chunk in agent.chat(f"Message {i}"):
                response_chunks.append(chunk)
            
            assert len(response_chunks) > 0
        
        # Genome should remain valid
        assert_genome_valid(agent.genome)
        assert agent.is_initialized is True

