"""
Pytest configuration and shared fixtures for Allele tests.

This module provides reusable fixtures and test utilities for all test modules.
"""

import pytest
import numpy as np
from typing import Dict, List, Generator, Any
from datetime import datetime, timezone

from allele import (
    ConversationalGenome,
    EvolutionEngine,
    EvolutionConfig,
    AgentConfig,
    KrakenLNN,
    LiquidDynamics,
)


@pytest.fixture
def sample_traits() -> Dict[str, float]:
    """Sample trait dictionary for testing."""
    return {
        'empathy': 0.85,
        'engagement': 0.75,
        'technical_knowledge': 0.90,
        'creativity': 0.65,
        'conciseness': 0.80,
        'context_awareness': 0.85,
        'adaptability': 0.70,
        'personability': 0.80
    }


@pytest.fixture
def default_genome() -> ConversationalGenome:
    """Create a genome with default traits."""
    return ConversationalGenome(genome_id="test_default")


@pytest.fixture
def custom_genome(sample_traits: Dict[str, float]) -> ConversationalGenome:
    """Create a genome with custom traits."""
    return ConversationalGenome(
        genome_id="test_custom",
        traits=sample_traits
    )


@pytest.fixture
def high_empathy_genome() -> ConversationalGenome:
    """Create a genome with high empathy trait."""
    return ConversationalGenome(
        genome_id="test_high_empathy",
        traits={'empathy': 0.95, 'engagement': 0.90}
    )


@pytest.fixture
def technical_genome() -> ConversationalGenome:
    """Create a genome optimized for technical knowledge."""
    return ConversationalGenome(
        genome_id="test_technical",
        traits={
            'technical_knowledge': 0.95,
            'conciseness': 0.90,
            'creativity': 0.40
        }
    )


@pytest.fixture
def population_of_genomes() -> List[ConversationalGenome]:
    """Create a diverse population of genomes for testing."""
    population = []
    np.random.seed(42)  # Reproducible randomness
    
    for i in range(20):
        traits = {}
        for trait_name in ConversationalGenome.DEFAULT_TRAITS.keys():
            traits[trait_name] = np.random.uniform(0.0, 1.0)
        
        genome = ConversationalGenome(
            genome_id=f"pop_genome_{i:03d}",
            traits=traits
        )
        population.append(genome)
    
    return population


@pytest.fixture
def evolution_config() -> EvolutionConfig:
    """Create a standard evolution configuration for testing."""
    return EvolutionConfig(
        population_size=20,
        generations=10,
        mutation_rate=0.15,
        crossover_rate=0.80,
        selection_pressure=0.20,
        elitism_enabled=True,
        tournament_size=3
    )


@pytest.fixture
def evolution_engine(evolution_config: EvolutionConfig) -> EvolutionEngine:
    """Create an evolution engine instance."""
    return EvolutionEngine(evolution_config)


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create a standard agent configuration for testing."""
    return AgentConfig(
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=2048,
        streaming=True,
        memory_enabled=True,
        evolution_enabled=True,
        kraken_enabled=True
    )


@pytest.fixture
def kraken_lnn() -> KrakenLNN:
    """Create a Kraken LNN instance for testing."""
    return KrakenLNN(
        reservoir_size=100,
        connectivity=0.1,
        memory_buffer_size=1000
    )


@pytest.fixture
def custom_liquid_dynamics() -> LiquidDynamics:
    """Create custom liquid dynamics configuration."""
    return LiquidDynamics(
        viscosity=0.15,
        temperature=1.2,
        pressure=0.9,
        flow_rate=0.6,
        turbulence=0.08
    )


@pytest.fixture
def sample_sequence() -> List[float]:
    """Generate a sample input sequence for LNN processing."""
    np.random.seed(42)
    return [np.random.uniform(0.0, 1.0) for _ in range(50)]


@pytest.fixture
def long_sequence() -> List[float]:
    """Generate a long input sequence for stress testing."""
    np.random.seed(42)
    return [np.random.uniform(0.0, 1.0) for _ in range(1000)]


@pytest.fixture
def fitness_function():
    """Create a simple fitness function for testing."""
    def fitness(genome: ConversationalGenome) -> float:
        """Simple fitness: reward balanced traits with high mean."""
        traits = genome.traits
        trait_values = list(traits.values())
        mean_value = sum(trait_values) / len(trait_values)
        variance = sum((v - mean_value) ** 2 for v in trait_values) / len(trait_values)
        
        balance_score = 1.0 - min(variance, 1.0)
        mean_score = mean_value
        
        return 0.6 * mean_score + 0.4 * balance_score
    
    return fitness


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    yield
    np.random.seed(None)


@pytest.fixture
def performance_timer():
    """Fixture for performance timing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def __enter__(self):
            self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.end_time = time.perf_counter()
        
        @property
        def elapsed(self) -> float:
            if self.start_time is None or self.end_time is None:
                return 0.0
            return self.end_time - self.start_time
    
    return Timer


@pytest.fixture
def memory_monitor():
    """Fixture for memory monitoring."""
    try:
        from memory_profiler import memory_usage
        
        class MemoryMonitor:
            def __init__(self):
                self.baseline = None
                self.peak = None
            
            def start(self):
                self.baseline = memory_usage()[0]
            
            def stop(self):
                current = memory_usage()[0]
                self.peak = current - self.baseline if self.baseline else current
            
            @property
            def delta(self) -> float:
                return self.peak if self.peak else 0.0
        
        return MemoryMonitor()
    except ImportError:
        # Fallback if memory-profiler not available
        class DummyMonitor:
            def start(self):
                pass
            
            def stop(self):
                pass
            
            @property
            def delta(self) -> float:
                return 0.0
        
        return DummyMonitor()

