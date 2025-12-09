#!/usr/bin/env python3
"""
Evolution example for Abe-NLP SDK.

This example demonstrates:
- Initializing a population of genomes
- Running evolutionary optimization
- Analyzing evolution results
"""

import asyncio
import random
from allele import (
    ConversationalGenome,
    EvolutionEngine,
    EvolutionConfig,
    GeneticOperators
)


def simple_fitness_function(genome: ConversationalGenome) -> float:
    """Simple fitness function for demonstration.
    
    In production, this would evaluate actual conversation quality.
    For this example, we reward balanced traits.
    """
    traits = genome.traits
    
    # Reward balance across all traits
    trait_values = list(traits.values())
    mean_value = sum(trait_values) / len(trait_values)
    variance = sum((v - mean_value) ** 2 for v in trait_values) / len(trait_values)
    
    # Fitness is high for balanced genomes with high mean
    balance_score = 1.0 - min(variance, 1.0)
    mean_score = mean_value
    
    fitness = 0.6 * mean_score + 0.4 * balance_score
    
    return fitness


async def main():
    """Run evolution example."""
    print("🧬 Abe-NLP Evolution Example\n")
    print("="*60)
    
    # Step 1: Configure evolution
    print("\n⚙️  Step 1: Configuring evolution parameters...")
    config = EvolutionConfig(
        population_size=20,
        generations=10,
        mutation_rate=0.15,
        crossover_rate=0.80,
        selection_pressure=0.20,
        elitism_enabled=True,
        tournament_size=3
    )
    
    print(f"   ✓ Population size: {config.population_size}")
    print(f"   ✓ Generations: {config.generations}")
    print(f"   ✓ Mutation rate: {config.mutation_rate}")
    print(f"   ✓ Crossover rate: {config.crossover_rate}")
    
    # Step 2: Initialize evolution engine
    print("\n🔬 Step 2: Initializing evolution engine...")
    engine = EvolutionEngine(config)
    
    # Step 3: Create initial population
    print("\n🌱 Step 3: Creating initial population...")
    population = engine.initialize_population()
    
    print(f"   ✓ Created {len(population)} diverse genomes")
    
    # Show initial population stats
    print("\n   Initial Population Stats:")
    for i, genome in enumerate(population[:3]):  # Show first 3
        print(f"     Genome {i+1}: {genome.genome_id}")
        print(f"       Traits: {genome.traits}")
    
    # Step 4: Run evolution
    print("\n⚡ Step 4: Running evolution...")
    print("\n   Generation | Best Fitness | Avg Fitness | Diversity")
    print("   " + "-"*56)
    
    # Create a callback to track progress
    generation_count = 0
    def fitness_with_tracking(genome):
        nonlocal generation_count
        fitness = simple_fitness_function(genome)
        return fitness
    
    # Run evolution
    best_genome = await engine.evolve(
        population,
        fitness_with_tracking,
        generations=config.generations
    )
    
    # Print evolution history
    for record in engine.evolution_history:
        print(f"   {record['generation']:>10} | "
              f"{record['best_fitness']:>12.4f} | "
              f"{record['avg_fitness']:>11.4f} | "
              f"{record['diversity']:>9.4f}")
    
    # Step 5: Analyze results
    print("\n" + "="*60)
    print("\n🏆 Step 5: Evolution Results\n")
    
    print(f"Best Genome ID: {best_genome.genome_id}")
    print(f"Fitness Score: {best_genome.fitness_score:.4f}")
    print(f"Generation: {best_genome.generation}")
    
    print("\nOptimized Traits:")
    for trait, value in best_genome.traits.items():
        bar_length = int(value * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {trait:20s}: {bar} {value:.2f}")
    
    # Step 6: Demonstrate genetic operators
    print("\n" + "="*60)
    print("\n🧪 Step 6: Genetic Operators Demo\n")
    
    # Tournament selection
    print("  Tournament Selection:")
    selected = GeneticOperators.tournament_selection(population, tournament_size=3)
    print(f"    Selected genome: {selected.genome_id}")
    print(f"    Fitness: {selected.fitness_score:.4f}")
    
    # Crossover
    print("\n  Crossover:")
    parent1 = population[0]
    parent2 = population[1]
    offspring = GeneticOperators.crossover(parent1, parent2)
    print(f"    Parent 1: {parent1.genome_id}")
    print(f"    Parent 2: {parent2.genome_id}")
    print(f"    Offspring: {offspring.genome_id}")
    
    # Mutation
    print("\n  Mutation:")
    original_traits = offspring.traits.copy()
    GeneticOperators.mutate(offspring, mutation_rate=0.5)
    print("    Trait changes:")
    for trait in original_traits:
        old_val = original_traits[trait]
        new_val = offspring.traits[trait]
        change = new_val - old_val
        if abs(change) > 0.01:
            print(f"      {trait}: {old_val:.2f} → {new_val:.2f} ({change:+.2f})")
    
    print("\n" + "="*60)
    print("\n✅ Evolution example completed!")
    print("\nKey Takeaways:")
    print("  - Evolution improves genome fitness over generations")
    print("  - Genetic operators (selection, crossover, mutation) drive evolution")
    print("  - Population diversity is maintained for exploration")
    print("  - Elitism preserves best solutions")


if __name__ == "__main__":
    asyncio.run(main())

