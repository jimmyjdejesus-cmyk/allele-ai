#!/usr/bin/env python3
"""
Unit tests for ConversationalGenome class.

Tests genome creation, mutation, crossover, and serialization.
"""

import pytest
from allele import ConversationalGenome
from allele.exceptions import ValidationError


class TestConversationalGenome:
    """Test suite for ConversationalGenome."""
    
    def test_genome_creation_default(self):
        """Test genome creation with default traits."""
        genome = ConversationalGenome("test_genome_001")
        
        assert genome.genome_id == "test_genome_001"
        assert len(genome.traits) == 8
        assert len(genome.genes) == 8
        
        # Check all default traits are 0.5
        for trait_value in genome.traits.values():
            assert trait_value == 0.5
    
    def test_genome_creation_custom_traits(self):
        """Test genome creation with custom traits."""
        custom_traits = {
            'empathy': 0.9,
            'technical_knowledge': 0.95,
            'creativity': 0.7
        }
        
        genome = ConversationalGenome("test_genome_002", custom_traits)
        
        assert genome.get_trait_value('empathy') == 0.9
        assert genome.get_trait_value('technical_knowledge') == 0.95
        assert genome.get_trait_value('creativity') == 0.7
        
        # Other traits should use defaults
        assert genome.get_trait_value('engagement') == 0.5
    
    def test_genome_trait_validation(self):
        """Test trait validation."""
        # Invalid trait value (too high)
        with pytest.raises(ValidationError):
            ConversationalGenome("invalid", {'empathy': 1.5})
        
        # Invalid trait value (negative)
        with pytest.raises(ValidationError):
            ConversationalGenome("invalid", {'empathy': -0.1})
        
        # Invalid trait name
        with pytest.raises(ValidationError):
            ConversationalGenome("invalid", {'invalid_trait': 0.5})
    
    def test_get_trait_value(self):
        """Test getting trait values."""
        genome = ConversationalGenome(
            "test",
            {'empathy': 0.8, 'creativity': 0.6}
        )
        
        assert genome.get_trait_value('empathy') == 0.8
        assert genome.get_trait_value('creativity') == 0.6
        assert genome.get_trait_value('engagement') == 0.5  # default
        assert genome.get_trait_value('unknown', default=0.3) == 0.3
    
    def test_set_trait_value(self):
        """Test setting trait values."""
        genome = ConversationalGenome("test")
        
        genome.set_trait_value('empathy', 0.9)
        assert genome.get_trait_value('empathy') == 0.9
        
        # Check corresponding gene was updated
        empathy_gene = next(
            g for g in genome.genes
            if g.metadata.get('optimization_target') == 'empathy'
        )
        assert empathy_gene.expression_level == 0.9
    
    def test_mutate_trait(self):
        """Test trait mutation."""
        genome = ConversationalGenome("test", {'empathy': 0.5})
        
        initial_empathy = genome.get_trait_value('empathy')
        
        # Mutate multiple times - at least one should change
        for _ in range(10):
            genome.mutate_trait('empathy', mutation_strength=0.2)
            new_empathy = genome.get_trait_value('empathy')
            if new_empathy != initial_empathy:
                break
        
        # Value should have changed
        assert genome.get_trait_value('empathy') != initial_empathy
        
        # Value should be in valid range
        assert 0.0 <= genome.get_trait_value('empathy') <= 1.0
    
    def test_mutate_all_traits(self):
        """Test mutating all traits."""
        genome = ConversationalGenome("test")
        
        initial_traits = genome.traits.copy()
        
        # With high mutation rate, some traits should change
        genome.mutate_all_traits(mutation_rate=1.0)
        
        # At least some traits should be different
        changes = sum(
            1 for trait in initial_traits
            if genome.traits[trait] != initial_traits[trait]
        )
        assert changes > 0
    
    def test_crossover(self):
        """Test genome crossover."""
        parent1 = ConversationalGenome(
            "parent1",
            {trait: 0.3 for trait in ConversationalGenome.DEFAULT_TRAITS}
        )
        parent2 = ConversationalGenome(
            "parent2",
            {trait: 0.7 for trait in ConversationalGenome.DEFAULT_TRAITS}
        )
        
        offspring = parent1.crossover(parent2)
        
        # Offspring should have intermediate values
        for trait in offspring.traits:
            assert 0.0 <= offspring.traits[trait] <= 1.0
            # Should be roughly between parents (accounting for variation)
            assert 0.2 <= offspring.traits[trait] <= 0.8
        
        # Offspring should have both parents listed
        assert len(offspring.metadata.parent_ids) == 2
        assert "parent1" in offspring.metadata.parent_ids
        assert "parent2" in offspring.metadata.parent_ids
        
        # Generation should increment
        assert offspring.generation == max(parent1.generation, parent2.generation) + 1
    
    def test_adapt_from_feedback(self):
        """Test genome adaptation from feedback."""
        genome = ConversationalGenome("test")
        
        initial_traits = genome.traits.copy()
        
        # Positive feedback should strengthen traits
        genome.adapt_from_feedback(feedback_score=0.9, learning_rate=0.1)
        
        # Some traits should have increased
        changes = sum(
            1 for trait in initial_traits
            if genome.traits[trait] > initial_traits[trait]
        )
        assert changes > 0
    
    def test_serialization(self):
        """Test genome serialization and deserialization."""
        original = ConversationalGenome(
            "test_serialize",
            {
                'empathy': 0.8,
                'technical_knowledge': 0.9,
                'creativity': 0.6
            }
        )
        original.fitness_score = 0.85
        original.generation = 5
        
        # Serialize
        data = original.to_dict()
        
        assert data['genome_id'] == "test_serialize"
        assert data['traits']['empathy'] == 0.8
        assert data['fitness_score'] == 0.85
        assert data['generation'] == 5
        
        # Deserialize
        restored = ConversationalGenome.from_dict(data)
        
        assert restored.genome_id == original.genome_id
        assert restored.traits == original.traits
        assert restored.fitness_score == original.fitness_score
        assert restored.generation == original.generation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

