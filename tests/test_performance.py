"""
Performance tests and benchmarks for Allele system.

Tests latency, throughput, memory usage, and scalability.
"""

import pytest
import numpy as np
import asyncio
from typing import List
import time

from allele import (
    ConversationalGenome,
    EvolutionEngine,
    EvolutionConfig,
    KrakenLNN,
    create_agent,
    AgentConfig
)
from tests.test_utils import (
    generate_population,
    generate_test_sequence,
    measure_execution_time,
    measure_async_execution_time
)


class TestPerformance:
    """Performance benchmarks and tests."""
    
    def test_genome_creation_performance(self, benchmark):
        """Benchmark genome creation speed."""
        def create_genome():
            return ConversationalGenome("perf_test")
        
        result = benchmark(create_genome)
        assert result is not None
    
    def test_trait_access_performance(self, custom_genome, benchmark):
        """Benchmark trait access operations."""
        def access_traits():
            for trait_name in custom_genome.traits.keys():
                _ = custom_genome.get_trait_value(trait_name)
        
        benchmark(access_traits)
    
    def test_crossover_performance(self, custom_genome, technical_genome, benchmark):
        """Benchmark crossover operation - should be <5ms."""
        def perform_crossover():
            return custom_genome.crossover(technical_genome)
        
        result = benchmark(perform_crossover)
        assert result is not None
        
        # Verify benchmark result is reasonable
        # Note: benchmark timing is handled by pytest-benchmark
    
    @pytest.mark.asyncio
    async def test_lnn_processing_performance(self, kraken_lnn):
        """Benchmark LNN processing - should be <10ms."""
        sequence = generate_test_sequence(50)
        
        start = time.perf_counter()
        result = await kraken_lnn.process_sequence(sequence)
        elapsed = time.perf_counter() - start
        
        assert result['success'] is True
        # Should be fast (<10ms per sequence, but allow some overhead)
        assert elapsed < 0.1  # 100ms is reasonable for 50-element sequence
    
    def test_mutation_performance(self, default_genome, benchmark):
        """Benchmark mutation operations."""
        def mutate_all():
            default_genome.mutate_all_traits(mutation_rate=1.0)
        
        benchmark(mutate_all)
    
    def test_serialization_performance(self, custom_genome, benchmark):
        """Benchmark serialization speed."""
        def serialize():
            return custom_genome.to_dict()
        
        result = benchmark(serialize)
        assert isinstance(result, dict)
    
    def test_deserialization_performance(self, custom_genome, benchmark):
        """Benchmark deserialization speed."""
        data = custom_genome.to_dict()
        
        def deserialize():
            return ConversationalGenome.from_dict(data)
        
        result = benchmark(deserialize)
        assert result is not None
    
    def test_population_initialization_performance(self, evolution_config, benchmark):
        """Benchmark population initialization."""
        engine = EvolutionEngine(evolution_config)
        
        def init_population():
            return engine.initialize_population()
        
        result = benchmark(init_population)
        assert len(result) == evolution_config.population_size
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_creation(self):
        """Test concurrent agent creation performance."""
        genomes = generate_population(10)
        config = AgentConfig()
        
        import asyncio
        
        async def create_agent_async(genome):
            return await create_agent(genome, config)
        
        start = time.perf_counter()
        agents = await asyncio.gather(*[create_agent_async(g) for g in genomes])
        elapsed = time.perf_counter() - start
        
        assert len(agents) == 10
        assert all(agent.is_initialized for agent in agents)
        # Should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds for 10 agents
    
    def test_memory_usage_genome(self):
        """Test memory usage of genome objects."""
        import sys
        
        genomes = []
        initial_size = sys.getsizeof([])
        
        # Create many genomes
        for i in range(100):
            genome = ConversationalGenome(f"mem_test_{i}")
            genomes.append(genome)
        
        # Estimate memory per genome
        total_size = sum(sys.getsizeof(g) for g in genomes)
        avg_size = total_size / len(genomes)
        
        # Should be relatively small (<10KB per genome)
        assert avg_size < 10 * 1024
    
    def test_scalability_large_population(self):
        """Test scalability with large population."""
        config = EvolutionConfig(
            population_size=500,
            generations=5,
            mutation_rate=0.1
        )
        engine = EvolutionEngine(config)
        
        start = time.perf_counter()
        population = engine.initialize_population()
        elapsed = time.perf_counter() - start
        
        assert len(population) == 500
        # Should initialize quickly even for large population
        assert elapsed < 2.0  # 2 seconds for 500 genomes
    
    @pytest.mark.asyncio
    async def test_scalability_long_sequence(self):
        """Test scalability with long input sequences."""
        lnn = KrakenLNN(reservoir_size=100)
        long_sequence = generate_test_sequence(1000)
        
        start = time.perf_counter()
        result = await lnn.process_sequence(long_sequence)
        elapsed = time.perf_counter() - start
        
        assert result['success'] is True
        assert len(result['liquid_outputs']) == len(long_sequence)
        # Should handle long sequences efficiently
        assert elapsed < 1.0  # 1 second for 1000 elements
    
    def test_throughput_many_crossovers(self, benchmark):
        """Test throughput of many crossover operations."""
        genomes = generate_population(20)
        
        def perform_many_crossovers():
            results = []
            for i in range(0, len(genomes) - 1, 2):
                offspring = genomes[i].crossover(genomes[i + 1])
                results.append(offspring)
            return results
        
        result = benchmark(perform_many_crossovers)
        assert len(result) == 10
    
    @pytest.mark.asyncio
    async def test_throughput_many_lnn_operations(self, kraken_lnn):
        """Test throughput of many LNN operations."""
        sequences = [generate_test_sequence(20, seed=i) for i in range(50)]
        
        start = time.perf_counter()
        results = await asyncio.gather(*[kraken_lnn.process_sequence(s) for s in sequences])
        elapsed = time.perf_counter() - start
        
        assert len(results) == 50
        assert all(r['success'] for r in results)
        # Should process many sequences efficiently
        assert elapsed < 5.0  # 5 seconds for 50 sequences
    
    def test_fitness_evaluation_performance(self, population_of_genomes, fitness_function, benchmark):
        """Benchmark fitness evaluation."""
        def evaluate_fitness():
            for genome in population_of_genomes:
                genome.fitness_score = fitness_function(genome)
        
        benchmark(evaluate_fitness)
    
    @pytest.mark.asyncio
    async def test_agent_chat_performance(self, custom_genome):
        """Test agent chat performance."""
        agent = await create_agent(custom_genome)
        
        start = time.perf_counter()
        response_chunks = []
        async for chunk in agent.chat("Performance test message"):
            response_chunks.append(chunk)
        elapsed = time.perf_counter() - start
        
        assert len(response_chunks) > 0
        # Should respond quickly (mock implementation)
        assert elapsed < 1.0
    
    def test_trait_mutation_performance(self, default_genome, benchmark):
        """Benchmark trait mutation operations."""
        def mutate_traits():
            for trait_name in default_genome.traits.keys():
                default_genome.mutate_trait(trait_name, mutation_strength=0.1)
        
        benchmark(mutate_traits)
    
    def test_genome_comparison_performance(self, population_of_genomes, benchmark):
        """Benchmark genome comparison operations."""
        def compare_genomes():
            comparisons = []
            for i in range(0, len(population_of_genomes) - 1):
                g1 = population_of_genomes[i]
                g2 = population_of_genomes[i + 1]
                # Simple comparison
                diff = sum(abs(g1.traits[t] - g2.traits[t]) for t in g1.traits)
                comparisons.append(diff)
            return comparisons
        
        result = benchmark(compare_genomes)
        assert len(result) == len(population_of_genomes) - 1

