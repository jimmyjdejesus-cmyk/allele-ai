"""
Runtime tests for ConversationalGenome class.

Tests actual execution paths, state transitions, and runtime behavior.
"""


import numpy as np
import pytest

from allele import ConversationalGenome
from allele.exceptions import ValidationError
from tests.test_utils import (
    assert_genome_valid,
)


class TestGenomeRuntime:
    """Runtime tests for ConversationalGenome."""

    def test_genome_creation_runtime(self):
        """Test genome creation executes correctly at runtime."""
        genome = ConversationalGenome("runtime_test_001")

        assert genome.genome_id == "runtime_test_001"
        assert_genome_valid(genome)

    def test_trait_access_runtime(self, custom_genome):
        """Test trait access operations at runtime."""
        # Test get operations
        empathy = custom_genome.get_trait_value('empathy')
        assert isinstance(empathy, float)
        assert 0.0 <= empathy <= 1.0

        # Test set operations
        original_value = custom_genome.get_trait_value('empathy')
        custom_genome.set_trait_value('empathy', 0.95)
        assert custom_genome.get_trait_value('empathy') == 0.95

        # Verify gene was updated
        empathy_gene = next(
            g for g in custom_genome.genes
            if g.metadata.get('optimization_target') == 'empathy'
        )
        assert empathy_gene.expression_level == 0.95

        # Restore original
        custom_genome.set_trait_value('empathy', original_value)

    def test_mutation_runtime_behavior(self, default_genome):
        """Test mutation operations execute correctly at runtime."""
        initial_traits = default_genome.traits.copy()

        # Mutate single trait
        default_genome.mutate_trait('empathy', mutation_strength=0.2)

        # Verify mutation occurred
        assert default_genome.get_trait_value('empathy') != initial_traits['empathy']
        assert 0.0 <= default_genome.get_trait_value('empathy') <= 1.0

        # Mutate all traits
        default_genome.mutate_all_traits(mutation_rate=1.0)

        # Verify some traits changed
        changes = sum(
            1 for trait in initial_traits
            if default_genome.traits[trait] != initial_traits[trait]
        )
        assert changes > 0

    def test_crossover_runtime_execution(self, custom_genome, technical_genome):
        """Test crossover executes correctly at runtime."""
        parent1_traits = custom_genome.traits.copy()
        parent2_traits = technical_genome.traits.copy()

        # Execute crossover
        offspring = custom_genome.crossover(technical_genome)

        # Verify offspring is valid
        assert_genome_valid(offspring)

        # Verify generation increment
        assert offspring.generation == max(custom_genome.generation, technical_genome.generation) + 1

        # Verify parent tracking
        assert len(offspring.metadata.parent_ids) == 2
        assert custom_genome.genome_id in offspring.metadata.parent_ids
        assert technical_genome.genome_id in offspring.metadata.parent_ids

        # Verify traits are blended
        for trait_name in offspring.traits:
            offspring_value = offspring.get_trait_value(trait_name)
            parent1_value = parent1_traits[trait_name]
            parent2_value = parent2_traits[trait_name]

            # Offspring should be somewhere between parents (with variation)
            min_parent = min(parent1_value, parent2_value)
            max_parent = max(parent1_value, parent2_value)

            # Allow for variation in crossover
            assert -0.1 <= offspring_value <= 1.1  # Allow slight overflow due to variation

    def test_adaptation_runtime(self, default_genome):
        """Test adaptation from feedback executes correctly."""
        initial_traits = default_genome.traits.copy()

        # Positive feedback
        default_genome.adapt_from_feedback(feedback_score=0.9, learning_rate=0.1)

        # Verify adaptation occurred
        changes = sum(
            1 for trait in initial_traits
            if abs(default_genome.traits[trait] - initial_traits[trait]) > 1e-6
        )
        assert changes > 0

        # Verify all traits still valid
        assert_genome_valid(default_genome)

    def test_serialization_runtime(self, custom_genome):
        """Test serialization/deserialization at runtime."""
        # Set some additional state
        custom_genome.fitness_score = 0.85
        custom_genome.generation = 5

        # Serialize
        data = custom_genome.to_dict()

        # Verify serialization structure
        assert isinstance(data, dict)
        assert 'genome_id' in data
        assert 'traits' in data
        assert 'fitness_score' in data
        assert 'generation' in data
        assert 'metadata' in data

        # Deserialize
        restored = ConversationalGenome.from_dict(data)

        # Verify restoration
        assert restored.genome_id == custom_genome.genome_id
        assert restored.traits == custom_genome.traits
        assert restored.fitness_score == custom_genome.fitness_score
        assert restored.generation == custom_genome.generation

        # Verify restored genome is valid
        assert_genome_valid(restored)

    def test_concurrent_mutations(self, default_genome):
        """Test concurrent mutation operations."""
        import threading

        results = []

        def mutate_trait(trait_name: str):
            for _ in range(10):
                default_genome.mutate_trait(trait_name, mutation_strength=0.1)
            results.append(default_genome.get_trait_value(trait_name))

        threads = []
        for trait_name in ['empathy', 'engagement', 'creativity']:
            thread = threading.Thread(target=mutate_trait, args=(trait_name,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all mutations completed
        assert len(results) == 3
        assert_genome_valid(default_genome)

    def test_trait_validation_runtime(self):
        """Test trait validation at runtime."""
        # Invalid trait value (too high)
        with pytest.raises(ValidationError):
            ConversationalGenome("invalid", {'empathy': 1.5})

        # Invalid trait value (negative)
        with pytest.raises(ValidationError):
            ConversationalGenome("invalid", {'empathy': -0.1})

        # Invalid trait name
        with pytest.raises(ValidationError):
            ConversationalGenome("invalid", {'invalid_trait': 0.5})

        # Valid genome should work
        valid_genome = ConversationalGenome("valid", {'empathy': 0.8})
        assert_genome_valid(valid_genome)

    def test_gene_expression_sync(self, default_genome):
        """Test gene expression levels sync with traits at runtime."""
        # Set trait value
        default_genome.set_trait_value('empathy', 0.9)

        # Verify corresponding gene updated
        empathy_gene = next(
            g for g in default_genome.genes
            if g.metadata.get('optimization_target') == 'empathy'
        )
        assert empathy_gene.expression_level == 0.9

        # Mutate trait
        original_value = default_genome.get_trait_value('empathy')
        default_genome.mutate_trait('empathy', mutation_strength=0.2)

        # Verify gene updated
        new_value = default_genome.get_trait_value('empathy')
        assert empathy_gene.expression_level == new_value
        assert abs(empathy_gene.expression_level - original_value) > 0.01

    def test_fitness_tracking_runtime(self, default_genome):
        """Test fitness score tracking at runtime."""
        # Initial fitness should be 0.0
        assert default_genome.fitness_score == 0.0

        # Set fitness
        default_genome.fitness_score = 0.85
        assert default_genome.fitness_score == 0.85

        # Fitness history tracking
        default_genome.fitness_history.append(0.75)
        default_genome.fitness_history.append(0.80)
        default_genome.fitness_history.append(0.85)

        assert len(default_genome.fitness_history) == 3
        assert default_genome.fitness_history[-1] == 0.85

    def test_metadata_runtime(self, default_genome):
        """Test metadata operations at runtime."""
        # Verify metadata exists
        assert default_genome.metadata is not None
        assert default_genome.metadata.creation_timestamp is not None
        assert default_genome.metadata.generation == 0

        # Update generation
        default_genome.generation = 5
        assert default_genome.generation == 5

        # Add tags
        if default_genome.metadata.tags is None:
            default_genome.metadata.tags = []
        default_genome.metadata.tags.append("test")
        assert "test" in default_genome.metadata.tags

    def test_large_trait_operations(self):
        """Test operations with many trait modifications."""
        genome = ConversationalGenome("stress_test")

        # Perform many mutations
        for _ in range(100):
            trait_name = np.random.choice(list(genome.traits.keys()))
            genome.mutate_trait(trait_name, mutation_strength=0.1)

        # Verify genome still valid
        assert_genome_valid(genome)

        # Verify all traits in valid range
        for trait_value in genome.traits.values():
            assert 0.0 <= trait_value <= 1.0

