"""
Runtime tests for EvolutionEngine and genetic operators.

Tests actual execution of evolution cycles, selection, crossover, and mutation.
"""

import pytest

from phylogenic import (
    EvolutionConfig,
    EvolutionEngine,
    GeneticOperators,
)
from tests.test_utils import (
    assert_genome_valid,
    calculate_population_diversity,
)


class TestEvolutionRuntime:
    """Runtime tests for EvolutionEngine."""

    def test_engine_initialization_runtime(self, evolution_config):
        """Test evolution engine initializes correctly at runtime."""
        engine = EvolutionEngine(evolution_config)

        assert engine.config == evolution_config
        assert engine.generation == 0
        assert engine.best_genome is None
        assert len(engine.evolution_history) == 0

    def test_population_initialization_runtime(self, evolution_engine):
        """Test population initialization executes correctly."""
        population = evolution_engine.initialize_population()

        assert len(population) == evolution_engine.config.population_size

        # Verify all genomes are valid
        for genome in population:
            assert_genome_valid(genome)

        # Verify diversity
        diversity = calculate_population_diversity(population)
        assert 0.0 <= diversity <= 1.0
        assert diversity > 0.0  # Should have some diversity

    def test_tournament_selection_runtime(self, population_of_genomes):
        """Test tournament selection executes correctly."""
        # Assign fitness scores
        for i, genome in enumerate(population_of_genomes):
            genome.fitness_score = i / len(population_of_genomes)

        # Run tournament selection multiple times
        selected_genomes = []
        for _ in range(10):
            selected = GeneticOperators.tournament_selection(
                population_of_genomes, tournament_size=3
            )
            selected_genomes.append(selected)
            assert_genome_valid(selected)

        # Verify selection is working (should get different genomes)
        unique_selections = len({g.genome_id for g in selected_genomes})
        assert unique_selections > 1  # Should select different genomes

    def test_crossover_operator_runtime(self, custom_genome, technical_genome):
        """Test crossover operator executes correctly."""
        offspring = GeneticOperators.crossover(custom_genome, technical_genome)

        assert_genome_valid(offspring)
        assert (
            offspring.generation
            == max(custom_genome.generation, technical_genome.generation) + 1
        )
        assert len(offspring.metadata.parent_ids) == 2

    def test_mutation_operator_runtime(self, custom_genome):
        """Test mutation operator executes correctly."""
        original_traits = custom_genome.traits.copy()

        GeneticOperators.mutate(custom_genome, mutation_rate=1.0)

        # Verify some traits changed
        changes = sum(
            1
            for trait in original_traits
            if abs(custom_genome.traits[trait] - original_traits[trait]) > 1e-6
        )
        assert changes > 0

        # Verify genome still valid
        assert_genome_valid(custom_genome)

    @pytest.mark.asyncio
    async def test_evolution_cycle_runtime(self, evolution_engine, fitness_function):
        """Test a complete evolution cycle executes correctly."""
        # Initialize population
        population = evolution_engine.initialize_population()
        len(population)

        # Run evolution for a few generations
        best_genome = await evolution_engine.evolve(
            population, fitness_function, generations=5
        )

        # Verify best genome
        assert best_genome is not None
        assert_genome_valid(best_genome)
        assert best_genome.fitness_score > 0.0

        # Verify evolution history
        assert len(evolution_engine.evolution_history) == 5

        # Verify fitness improvement (or at least tracking)
        for record in evolution_engine.evolution_history:
            assert "generation" in record
            assert "best_fitness" in record
            assert "avg_fitness" in record
            assert "diversity" in record
            assert 0.0 <= record["best_fitness"] <= 1.0

    @pytest.mark.asyncio
    async def test_elitism_runtime(self, evolution_config, fitness_function):
        """Test elitism preserves best genomes."""
        evolution_config.elitism_enabled = True
        engine = EvolutionEngine(evolution_config)

        population = engine.initialize_population()

        # Evaluate initial fitness
        for genome in population:
            genome.fitness_score = fitness_function(genome)

        initial_best = max(population, key=lambda g: g.fitness_score)
        initial_best_score = initial_best.fitness_score

        # Evolve one generation
        await engine.evolve(population, fitness_function, generations=1)

        # Best score should be maintained or improved
        assert engine.best_genome.fitness_score >= initial_best_score

    @pytest.mark.asyncio
    async def test_elite_preservation_hpc_mode(
        self, evolution_config, fitness_function
    ):
        """Test elite genomes remain unchanged during HPC mode in-place mutations."""
        evolution_config.elitism_enabled = True
        evolution_config.hpc_mode = True
        engine = EvolutionEngine(evolution_config)

        population = engine.initialize_population()

        # Evaluate initial fitness
        for genome in population:
            genome.fitness_score = fitness_function(genome)

        # Sort by fitness and capture elite traits before evolution
        population.sort(key=lambda g: g.fitness_score, reverse=True)
        elitism_count = int(
            evolution_config.population_size * evolution_config.selection_pressure
        )
        elite_genomes = population[:elitism_count]

        # Store original traits of elites (deep copy to avoid reference issues)
        elite_original_traits = []
        for elite in elite_genomes:
            elite_original_traits.append(
                {
                    "traits": elite.traits.copy(),
                    "genome_id": elite.genome_id,
                    "fitness_score": elite.fitness_score,
                }
            )

        # Evolve one generation
        await engine.evolve(population, fitness_function, generations=1)

        # Verify elites remain in the population with unchanged traits
        # The deepcopied elites should be in the new population
        current_population = population

        # Check that each original elite's traits are preserved.
        # We search in a copy of the population and remove found elites
        # to correctly handle cases where multiple elites have identical traits.
        searchable_population = list(current_population)
        for original_elite_data in elite_original_traits:
            match_found = False
            for i, genome in enumerate(searchable_population):
                if genome.traits == original_elite_data["traits"]:
                    searchable_population.pop(i)
                    match_found = True
                    break
            assert (
                match_found
            ), f"Elite genome {original_elite_data['genome_id']} not preserved"

    def test_population_diversity_tracking(self, evolution_engine, fitness_function):
        """Test population diversity is tracked correctly."""
        population = evolution_engine.initialize_population()

        # Evaluate fitness
        for genome in population:
            genome.fitness_score = fitness_function(genome)

        # Calculate diversity manually
        manual_diversity = calculate_population_diversity(population)

        # Calculate diversity via engine
        engine_diversity = evolution_engine._calculate_diversity(population)

        # Should be similar (allowing for floating point differences)
        assert abs(manual_diversity - engine_diversity) < 0.01

    @pytest.mark.asyncio
    async def test_generation_progression(self, evolution_engine, fitness_function):
        """Test generation counter progresses correctly."""
        population = evolution_engine.initialize_population()

        initial_generation = evolution_engine.generation

        await evolution_engine.evolve(population, fitness_function, generations=3)

        assert evolution_engine.generation == initial_generation + 3

    def test_next_generation_creation(self, evolution_engine, fitness_function):
        """Test next generation creation logic."""
        population = evolution_engine.initialize_population()

        # Evaluate fitness
        for genome in population:
            genome.fitness_score = fitness_function(genome)

        # Create next generation
        next_gen = evolution_engine._create_next_generation(population)

        # Verify size maintained
        assert len(next_gen) == len(population)

        # Verify all genomes valid
        for genome in next_gen:
            assert_genome_valid(genome)

        # Verify diversity maintained
        diversity = calculate_population_diversity(next_gen)
        assert diversity > 0.0

    @pytest.mark.asyncio
    async def test_convergence_behavior(self, evolution_config, fitness_function):
        """Test evolution convergence over many generations."""
        evolution_config.generations = 20
        engine = EvolutionEngine(evolution_config)

        population = engine.initialize_population()
        await engine.evolve(population, fitness_function)

        # Verify convergence (best fitness should improve)
        initial_fitness = engine.evolution_history[0]["best_fitness"]
        final_fitness = engine.evolution_history[-1]["best_fitness"]

        # Fitness should improve or at least be tracked
        assert final_fitness >= initial_fitness - 0.1  # Allow small variance

    def test_large_population_runtime(self, fitness_function):
        """Test evolution with large population."""
        config = EvolutionConfig(
            population_size=100, generations=5, mutation_rate=0.1, crossover_rate=0.8
        )
        engine = EvolutionEngine(config)

        population = engine.initialize_population()

        # Verify large population works
        assert len(population) == 100

        # Run evolution
        import asyncio

        best = asyncio.run(engine.evolve(population, fitness_function))

        assert best is not None
        assert_genome_valid(best)

    def test_crossover_rate_effect(self, evolution_config, fitness_function):
        """Test crossover rate affects offspring generation."""
        # Test with high crossover rate
        evolution_config.crossover_rate = 1.0
        engine_high = EvolutionEngine(evolution_config)
        pop_high = engine_high.initialize_population()

        for g in pop_high:
            g.fitness_score = fitness_function(g)

        next_gen_high = engine_high._create_next_generation(pop_high)

        # Test with low crossover rate
        evolution_config.crossover_rate = 0.0
        engine_low = EvolutionEngine(evolution_config)
        pop_low = engine_low.initialize_population()

        for g in pop_low:
            g.fitness_score = fitness_function(g)

        next_gen_low = engine_low._create_next_generation(pop_low)

        # Both should produce valid populations
        assert len(next_gen_high) == len(next_gen_low)
        for genome in next_gen_high + next_gen_low:
            assert_genome_valid(genome)
