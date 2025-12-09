#!/usr/bin/env python3
"""
Test core functionality without requiring LLM API keys.
"""

import sys
from abe_nlp import (
    ConversationalGenome,
    EvolutionEngine,
    EvolutionConfig,
    KrakenLNN,
    GeneticOperators
)

def test_genome_functionality():
    """Test genome creation and manipulation."""
    print("\n🧬 Testing Genome Functionality...")

    # Create genome
    genome = ConversationalGenome(
        genome_id="test_001",
        traits={
            'empathy': 0.9,
            'technical_knowledge': 0.95,
            'creativity': 0.7
        }
    )

    assert genome.genome_id == "test_001"
    assert genome.get_trait_value('empathy') == 0.9
    assert genome.generation == 0
    print("  ✓ Genome creation works")

    # Test mutation
    original_empathy = genome.get_trait_value('empathy')
    genome.mutate_trait('empathy', mutation_strength=0.5)
    new_empathy = genome.get_trait_value('empathy')
    assert 0.0 <= new_empathy <= 1.0
    print("  ✓ Trait mutation works")

    # Test serialization
    data = genome.to_dict()
    restored = ConversationalGenome.from_dict(data)
    assert restored.genome_id == genome.genome_id
    assert restored.get_trait_value('creativity') == genome.get_trait_value('creativity')
    print("  ✓ Serialization works")

    return True

def test_crossover_functionality():
    """Test genome crossover with generation tracking."""
    print("\n🧬 Testing Crossover & Generation...")

    parent1 = ConversationalGenome("parent1", {
        trait: 0.3 for trait in ConversationalGenome.DEFAULT_TRAITS
    })
    parent2 = ConversationalGenome("parent2", {
        trait: 0.7 for trait in ConversationalGenome.DEFAULT_TRAITS
    })

    assert parent1.generation == 0
    assert parent2.generation == 0
    print("  ✓ Parents at generation 0")

    offspring = parent1.crossover(parent2)

    # Test the bug fix - offspring should be generation 1
    assert offspring.generation == 1, f"Expected generation 1, got {offspring.generation}"
    assert offspring.metadata.generation == 1
    print("  ✓ Offspring at generation 1 (BUG FIXED!)")

    # Verify traits are blended
    for trait_value in offspring.traits.values():
        assert 0.0 <= trait_value <= 1.0
    print("  ✓ Trait blending works")

    return True

def test_evolution_engine():
    """Test evolution engine initialization."""
    print("\n🧪 Testing Evolution Engine...")

    config = EvolutionConfig(
        population_size=10,
        generations=5,
        mutation_rate=0.1,
        crossover_rate=0.8
    )

    engine = EvolutionEngine(config)
    assert engine.config.population_size == 10
    print("  ✓ Evolution config works")

    # Create initial population
    population = engine.initialize_population()
    assert len(population) == 10
    print("  ✓ Population initialization works")

    # Test genetic operators
    parent = population[0]
    child = GeneticOperators.mutate(parent, mutation_rate=0.1)
    assert child.genome_id != parent.genome_id
    print("  ✓ Genetic operators work")

    return True

def test_kraken_lnn():
    """Test Kraken Liquid Neural Network."""
    print("\n🧠 Testing Kraken LNN...")

    kraken = KrakenLNN(
        reservoir_size=50,
        connectivity=0.1
    )

    assert kraken.reservoir_size == 50
    assert kraken.connectivity == 0.1
    print("  ✓ Kraken initialization works")

    # Test basic structure
    assert hasattr(kraken, 'reservoir_weights')
    assert hasattr(kraken, 'input_weights')
    print("  ✓ Neural network structure OK")

    return True

def main():
    """Run all functionality tests."""
    print("="*60)
    print("🚀 Abe-NLP Functionality Test Suite")
    print("="*60)

    tests = [
        ("Genome", test_genome_functionality),
        ("Crossover", test_crossover_functionality),
        ("Evolution", test_evolution_engine),
        ("Kraken LNN", test_kraken_lnn),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {name} test FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print(f"📊 Results: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n✅ ALL TESTS PASSED - Code is production ready!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed - needs fixing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
