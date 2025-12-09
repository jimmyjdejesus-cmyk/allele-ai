"""
Stress tests for Allele system.

Tests edge cases, resource limits, and failure modes.
"""

import pytest
import numpy as np
from typing import List
import asyncio

from allele import (
    ConversationalGenome,
    EvolutionEngine,
    EvolutionConfig,
    KrakenLNN,
    create_agent,
    AgentConfig
)
from allele.exceptions import ValidationError
from tests.test_utils import (
    generate_population,
    generate_test_sequence,
    assert_genome_valid
)


class TestStress:
    """Stress tests for edge cases and limits."""
    
    def test_large_population_stress(self):
        """Test with very large population."""
        config = EvolutionConfig(
            population_size=1000,
            generations=1,
            mutation_rate=0.1
        )
        engine = EvolutionEngine(config)
        
        population = engine.initialize_population()
        
        assert len(population) == 1000
        
        # All genomes should be valid
        for genome in population:
            assert_genome_valid(genome)
    
    @pytest.mark.asyncio
    async def test_long_evolution_run(self, fitness_function):
        """Test evolution with many generations."""
        config = EvolutionConfig(
            population_size=50,
            generations=100,
            mutation_rate=0.1
        )
        engine = EvolutionEngine(config)
        
        population = engine.initialize_population()
        
        # Run long evolution
        best_genome = await engine.evolve(population, fitness_function)
        
        assert best_genome is not None
        assert len(engine.evolution_history) == 100
        assert_genome_valid(best_genome)
    
    def test_extreme_trait_values(self):
        """Test genomes with extreme trait values."""
        # Minimum values
        min_genome = ConversationalGenome(
            "min_traits",
            traits={trait: 0.0 for trait in ConversationalGenome.DEFAULT_TRAITS.keys()}
        )
        assert_genome_valid(min_genome)
        
        # Maximum values
        max_genome = ConversationalGenome(
            "max_traits",
            traits={trait: 1.0 for trait in ConversationalGenome.DEFAULT_TRAITS.keys()}
        )
        assert_genome_valid(max_genome)
        
        # Mixed extreme values
        mixed_genome = ConversationalGenome(
            "mixed_traits",
            traits={
                'empathy': 0.0,
                'engagement': 1.0,
                'technical_knowledge': 0.0,
                'creativity': 1.0,
                'conciseness': 0.0,
                'context_awareness': 1.0,
                'adaptability': 0.0,
                'personability': 1.0
            }
        )
        assert_genome_valid(mixed_genome)
    
    def test_many_mutations(self, default_genome):
        """Test many consecutive mutations."""
        original_traits = default_genome.traits.copy()
        
        # Perform many mutations
        for _ in range(1000):
            trait_name = np.random.choice(list(default_genome.traits.keys()))
            default_genome.mutate_trait(trait_name, mutation_strength=0.1)
        
        # Genome should still be valid
        assert_genome_valid(default_genome)
        
        # Traits should still be in valid range
        for trait_value in default_genome.traits.values():
            assert 0.0 <= trait_value <= 1.0
    
    def test_deep_crossover_chain(self):
        """Test crossover chain creating many generations."""
        parent1 = ConversationalGenome("p1")
        parent2 = ConversationalGenome("p2")
        
        current_gen = parent1
        for i in range(10):
            next_gen = ConversationalGenome(f"gen_{i}")
            current_gen = current_gen.crossover(next_gen)
            assert_genome_valid(current_gen)
            assert current_gen.generation == i + 1
    
    @pytest.mark.asyncio
    async def test_extended_sequence_processing(self, kraken_lnn):
        """Test processing very long sequences."""
        very_long_sequence = generate_test_sequence(10000)
        
        result = await kraken_lnn.process_sequence(very_long_sequence)
        
        assert result['success'] is True
        assert len(result['liquid_outputs']) == len(very_long_sequence)
    
    def test_memory_buffer_overflow(self, kraken_lnn):
        """Test memory buffer handling when overflow occurs."""
        buffer_size = kraken_lnn.temporal_memory.buffer_size
        
        # Fill buffer beyond capacity
        sequences = [generate_test_sequence(5, seed=i) for i in range(buffer_size + 100)]
        
        async def fill_buffer():
            for seq in sequences:
                await kraken_lnn.process_sequence(seq, memory_consolidation=False)
        
        asyncio.run(fill_buffer())
        
        # Memory should not exceed buffer size
        assert len(kraken_lnn.temporal_memory.memories) <= buffer_size
    
    def test_concurrent_mutations(self, default_genome):
        """Test concurrent mutation operations."""
        import threading
        
        def mutate_random_trait():
            for _ in range(100):
                trait_name = np.random.choice(list(default_genome.traits.keys()))
                default_genome.mutate_trait(trait_name, mutation_strength=0.1)
        
        threads = [threading.Thread(target=mutate_random_trait) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Genome should still be valid
        assert_genome_valid(default_genome)
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        # Invalid trait values should raise ValidationError
        with pytest.raises(ValidationError):
            ConversationalGenome("invalid", {'empathy': 1.5})
        
        with pytest.raises(ValidationError):
            ConversationalGenome("invalid", {'empathy': -0.1})
        
        with pytest.raises(ValidationError):
            ConversationalGenome("invalid", {'invalid_trait': 0.5})
    
    @pytest.mark.asyncio
    async def test_many_concurrent_agents(self):
        """Test creating many agents concurrently."""
        genomes = generate_population(50)
        config = AgentConfig()
        
        agents = await asyncio.gather(*[create_agent(g, config) for g in genomes])
        
        assert len(agents) == 50
        assert all(agent.is_initialized for agent in agents)
    
    def test_population_diversity_maintenance(self, evolution_config, fitness_function):
        """Test population diversity is maintained under stress."""
        config = evolution_config
        config.population_size = 200
        config.generations = 20
        
        engine = EvolutionEngine(config)
        population = engine.initialize_population()
        
        initial_diversity = engine._calculate_diversity(population)
        
        # Evolve
        import asyncio
        asyncio.run(engine.evolve(population, fitness_function))
        
        final_diversity = engine._calculate_diversity(population)
        
        # Diversity should be maintained (not collapse to zero)
        assert final_diversity > 0.0
    
    def test_rapid_serialization_deserialization(self, custom_genome):
        """Test rapid serialization/deserialization cycles."""
        current_genome = custom_genome
        for _ in range(100):
            data = current_genome.to_dict()
            restored = ConversationalGenome.from_dict(data)
            assert_genome_valid(restored)
            current_genome = restored  # Use restored for next iteration
    
    @pytest.mark.asyncio
    async def test_rapid_lnn_processing(self, kraken_lnn):
        """Test rapid LNN processing."""
        sequences = [generate_test_sequence(10, seed=i) for i in range(200)]
        
        # Process rapidly
        results = await asyncio.gather(*[kraken_lnn.process_sequence(s) for s in sequences])
        
        assert len(results) == 200
        assert all(r['success'] for r in results)
    
    def test_extreme_mutation_rates(self, default_genome):
        """Test with extreme mutation rates."""
        # Very high mutation rate
        default_genome.mutate_all_traits(mutation_rate=1.0)
        assert_genome_valid(default_genome)
        
        # Very low mutation rate (should still work)
        original_traits = default_genome.traits.copy()
        default_genome.mutate_all_traits(mutation_rate=0.001)
        # May or may not mutate, but should remain valid
        assert_genome_valid(default_genome)
    
    def test_empty_population_handling(self):
        """Test handling edge case of empty population."""
        config = EvolutionConfig(population_size=0)
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, AssertionError)):
            engine = EvolutionEngine(config)
            engine.initialize_population()
    
    def test_single_genome_population(self, fitness_function):
        """Test evolution with single genome population."""
        config = EvolutionConfig(population_size=1, generations=5)
        engine = EvolutionEngine(config)
        
        population = engine.initialize_population()
        
        import asyncio
        best_genome = asyncio.run(engine.evolve(population, fitness_function))
        
        assert best_genome is not None
        assert_genome_valid(best_genome)
    
    @pytest.mark.asyncio
    async def test_agent_with_extreme_genome(self):
        """Test agent creation with extreme trait values."""
        extreme_genome = ConversationalGenome(
            "extreme",
            traits={trait: 1.0 for trait in ConversationalGenome.DEFAULT_TRAITS.keys()}
        )
        
        agent = await create_agent(extreme_genome)
        assert agent.is_initialized is True
        
        # Should still work
        response_chunks = []
        async for chunk in agent.chat("Test"):
            response_chunks.append(chunk)
        
        assert len(response_chunks) > 0
    
    def test_very_small_reservoir(self):
        """Test LNN with very small reservoir."""
        lnn = KrakenLNN(reservoir_size=10, connectivity=0.5)
        sequence = generate_test_sequence(20)
        
        import asyncio
        result = asyncio.run(lnn.process_sequence(sequence))
        
        assert result['success'] is True
    
    def test_very_large_reservoir(self):
        """Test LNN with very large reservoir."""
        lnn = KrakenLNN(reservoir_size=1000, connectivity=0.05)
        sequence = generate_test_sequence(50)
        
        import asyncio
        result = asyncio.run(lnn.process_sequence(sequence))
        
        assert result['success'] is True
        assert len(result['reservoir_state']) == 1000

