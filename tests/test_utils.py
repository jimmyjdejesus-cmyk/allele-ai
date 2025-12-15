"""
Testing utilities and helper functions for Allele tests.

This module provides utility functions for test data generation,
comparison, and analysis.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from allele import ConversationalGenome, TraitDict


def generate_random_genome(
    genome_id: str,
    seed: Optional[int] = None
) -> ConversationalGenome:
    """Generate a genome with random traits.
    
    Args:
        genome_id: Unique identifier for the genome
        seed: Optional random seed for reproducibility
    
    Returns:
        ConversationalGenome with random trait values
    """
    if seed is not None:
        np.random.seed(seed)

    traits = {}
    for trait_name in ConversationalGenome.DEFAULT_TRAITS.keys():
        traits[trait_name] = np.random.uniform(0.0, 1.0)

    return ConversationalGenome(genome_id=genome_id, traits=traits)


def generate_population(
    size: int,
    base_traits: Optional[TraitDict] = None,
    seed: Optional[int] = None
) -> List[ConversationalGenome]:
    """Generate a population of genomes.
    
    Args:
        size: Number of genomes to generate
        base_traits: Optional base traits to vary from
        seed: Optional random seed for reproducibility
    
    Returns:
        List of ConversationalGenome instances
    """
    if seed is not None:
        np.random.seed(seed)

    if base_traits is None:
        base_traits = ConversationalGenome.DEFAULT_TRAITS.copy()

    population = []
    for i in range(size):
        traits = {}
        for trait_name, base_value in base_traits.items():
            variation = np.random.uniform(-0.2, 0.2)
            traits[trait_name] = np.clip(base_value + variation, 0.0, 1.0)

        genome = ConversationalGenome(
            genome_id=f"genome_{i:04d}",
            traits=traits
        )
        population.append(genome)

    return population


def compare_genomes(
    genome1: ConversationalGenome,
    genome2: ConversationalGenome
) -> Dict[str, Any]:
    """Compare two genomes and return differences.
    
    Args:
        genome1: First genome to compare
        genome2: Second genome to compare
    
    Returns:
        Dictionary with comparison metrics
    """
    trait_diffs = {}
    for trait_name in ConversationalGenome.DEFAULT_TRAITS.keys():
        val1 = genome1.get_trait_value(trait_name)
        val2 = genome2.get_trait_value(trait_name)
        trait_diffs[trait_name] = abs(val1 - val2)

    mean_diff = np.mean(list(trait_diffs.values()))
    max_diff = max(trait_diffs.values())

    return {
        'trait_differences': trait_diffs,
        'mean_difference': mean_diff,
        'max_difference': max_diff,
        'identical': all(diff < 1e-10 for diff in trait_diffs.values())
    }


def calculate_population_diversity(
    population: List[ConversationalGenome]
) -> float:
    """Calculate genetic diversity in a population.
    
    Args:
        population: List of genomes
    
    Returns:
        Diversity score (0.0-1.0)
    """
    if len(population) < 2:
        return 0.0

    trait_values = {}
    for trait_name in ConversationalGenome.DEFAULT_TRAITS.keys():
        trait_values[trait_name] = [
            g.get_trait_value(trait_name) for g in population
        ]

    diversities = []
    for trait, values in trait_values.items():
        std_dev = np.std(values)
        diversities.append(std_dev)

    return float(np.mean(diversities)) if diversities else 0.0


def calculate_population_statistics(
    population: List[ConversationalGenome]
) -> Dict[str, Any]:
    """Calculate statistics for a population.
    
    Args:
        population: List of genomes
    
    Returns:
        Dictionary with population statistics
    """
    if not population:
        return {
            'size': 0,
            'mean_fitness': 0.0,
            'std_fitness': 0.0,
            'best_fitness': 0.0,
            'worst_fitness': 0.0,
            'diversity': 0.0
        }

    fitness_scores = [g.fitness_score for g in population]

    return {
        'size': len(population),
        'mean_fitness': float(np.mean(fitness_scores)),
        'std_fitness': float(np.std(fitness_scores)),
        'best_fitness': float(max(fitness_scores)),
        'worst_fitness': float(min(fitness_scores)),
        'diversity': calculate_population_diversity(population)
    }


def generate_fitness_function(
    weights: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None
):
    """Generate a fitness function with custom weights.

    Args:
        weights: Optional dictionary mapping trait names to weights
        seed: Optional random seed for deterministic results

    Returns:
        Fitness function that takes a genome and returns a score
    """
    if weights is None:
        weights = dict.fromkeys(ConversationalGenome.DEFAULT_TRAITS.keys(), 1.0)

    def fitness(genome: ConversationalGenome) -> float:
        """Calculate fitness based on weighted traits."""
        score = 0.0
        total_weight = 0.0

        for trait_name, weight in weights.items():
            trait_value = genome.get_trait_value(trait_name)
            score += trait_value * weight
            total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0

    return fitness


def assert_genome_valid(genome: ConversationalGenome) -> None:
    """Assert that a genome is valid.
    
    Args:
        genome: Genome to validate
    
    Raises:
        AssertionError: If genome is invalid
    """
    assert genome.genome_id is not None
    assert len(genome.genome_id) > 0

    assert len(genome.traits) == 8
    assert len(genome.genes) == 8

    for trait_name, value in genome.traits.items():
        assert trait_name in ConversationalGenome.DEFAULT_TRAITS
        assert 0.0 <= value <= 1.0

    assert genome.generation >= 0
    assert 0.0 <= genome.fitness_score <= 1.0


def generate_test_sequence(
    length: int,
    pattern: str = "random",
    seed: Optional[int] = None
) -> List[float]:
    """Generate test sequences for LNN processing.
    
    Args:
        length: Length of the sequence
        pattern: Pattern type ("random", "sine", "step", "noise")
        seed: Optional random seed
    
    Returns:
        List of float values
    """
    if seed is not None:
        np.random.seed(seed)

    if pattern == "random":
        return [np.random.uniform(0.0, 1.0) for _ in range(length)]
    elif pattern == "sine":
        return [0.5 + 0.5 * np.sin(2 * np.pi * i / length) for i in range(length)]
    elif pattern == "step":
        return [0.0 if i < length // 2 else 1.0 for i in range(length)]
    elif pattern == "noise":
        return [np.random.normal(0.5, 0.2) for _ in range(length)]
    else:
        return [np.random.uniform(0.0, 1.0) for _ in range(length)]


def measure_execution_time(func, *args, **kwargs) -> Tuple[Any, float]:
    """Measure execution time of a function.
    
    Args:
        func: Function to measure
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function
    
    Returns:
        Tuple of (result, execution_time_seconds)
    """
    import time

    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()

    return result, end - start


async def measure_async_execution_time(func, *args, **kwargs) -> Tuple[Any, float]:
    """Measure execution time of an async function.
    
    Args:
        func: Async function to measure
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function
    
    Returns:
        Tuple of (result, execution_time_seconds)
    """
    import time

    start = time.perf_counter()
    result = await func(*args, **kwargs)
    end = time.perf_counter()

    return result, end - start


def create_serialization_test_data() -> Dict[str, Any]:
    """Create test data for serialization tests.
    
    Returns:
        Dictionary with test genome data
    """
    return {
        "genome_id": "test_serialize_001",
        "traits": {
            'empathy': 0.85,
            'engagement': 0.75,
            'technical_knowledge': 0.90,
            'creativity': 0.65,
            'conciseness': 0.80,
            'context_awareness': 0.85,
            'adaptability': 0.70,
            'personability': 0.80
        },
        "fitness_score": 0.82,
        "fitness_history": [0.75, 0.78, 0.80, 0.82],
        "generation": 5,
        "metadata": {
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
            "last_mutation": datetime.now(timezone.utc).isoformat(),
            "parent_ids": ["parent_001", "parent_002"],
            "lineage": ["ancestor_001", "ancestor_002"],
            "tags": ["test", "serialization"]
        }
    }
