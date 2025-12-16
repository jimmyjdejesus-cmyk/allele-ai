"""Deterministic tests for evolution mutation and elitism with seeded RNG."""


import copy
import numpy as np
import pytest

from allele.evolution import EvolutionConfig, EvolutionEngine, GeneticOperators
from allele.genome import ConversationalGenome
from tests.test_utils import generate_fitness_function, generate_population


class TestEvolutionMutationAndElitism:
    """Deterministic tests for mutation and elitism using seeded random number generators."""

    @pytest.fixture
    def deterministic_evolution_config(self):
        """Create evolution config with deterministic settings."""
        return EvolutionConfig(
            population_size=20,
            generations=3,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elitism_enabled=True,
            selection_pressure=0.2,
            hpc_mode=False
        )

    @pytest.fixture
    def deterministic_population(self):
        """Create deterministic population with seeded RNG."""
        np.random.seed(123)  # Seed for deterministic population
        return generate_population(20, seed=456)

    @pytest.fixture
    def deterministic_fitness_function(self):
        """Create deterministic fitness function."""
        return generate_fitness_function(seed=789)

    def test_mutation_deterministic_behavior_same_seed(self):
        """Test that mutation produces identical results with same seed."""
        # Create two genomes with identical initial state
        np.random.seed(100)
        genome1 = ConversationalGenome(
            genome_id="test_genome_1",
            traits={'empathy': 0.5, 'technical_knowledge': 0.6, 'creativity': 0.7}
        )

        np.random.seed(100)
        genome2 = ConversationalGenome(
            genome_id="test_genome_2",
            traits={'empathy': 0.5, 'technical_knowledge': 0.6, 'creativity': 0.7}
        )

        original_traits1 = genome1.traits.copy()
        genome2.traits.copy()

        # Apply mutation with same seed
        GeneticOperators.mutate(genome1, mutation_rate=0.5, seed=200)
        GeneticOperators.mutate(genome2, mutation_rate=0.5, seed=200)

        # Results should be identical
        np.testing.assert_array_equal(
            genome1.traits,
            genome2.traits
        )

        # Verify some change occurred
        changes = sum(
            1 for trait in original_traits1
            if abs(genome1.traits[trait] - original_traits1[trait]) > 1e-6
        )
        assert changes > 0

    def test_mutation_deterministic_behavior_different_seed(self):
        """Test that mutation produces different results with different seeds."""
        np.random.seed(100)
        genome1 = ConversationalGenome(
            genome_id="test_genome_1",
            traits={'empathy': 0.5, 'technical_knowledge': 0.6, 'creativity': 0.7}
        )

        np.random.seed(101)
        genome2 = ConversationalGenome(
            genome_id="test_genome_2",
            traits={'empathy': 0.5, 'technical_knowledge': 0.6, 'creativity': 0.7}
        )

        # Apply mutation with different seeds
        GeneticOperators.mutate(genome1, mutation_rate=0.5, seed=200)
        GeneticOperators.mutate(genome2, mutation_rate=0.5, seed=201)

        # Results should be different
        assert not np.array_equal(genome1.traits, genome2.traits)

    def test_crossover_deterministic_behavior(self):
        """Test that crossover produces deterministic offspring."""
        np.random.seed(150)
        parent1 = ConversationalGenome(
            genome_id="parent1",
            traits={'empathy': 0.8, 'technical_knowledge': 0.9, 'creativity': 0.3}
        )

        np.random.seed(151)
        parent2 = ConversationalGenome(
            genome_id="parent2",
            traits={'empathy': 0.2, 'technical_knowledge': 0.1, 'creativity': 0.9}
        )

        # Perform crossover twice with same seed
        np.random.seed(300)
        offspring1 = GeneticOperators.crossover(parent1, parent2, seed=400)

        np.random.seed(300)
        offspring2 = GeneticOperators.crossover(parent1, parent2, seed=400)

        # Offspring should be identical
        np.testing.assert_array_equal(
            offspring1.traits,
            offspring2.traits
        )

        # Verify offspring combines traits from both parents
        assert offspring1.genome_id != parent1.genome_id
        assert offspring1.genome_id != parent2.genome_id
        assert len(offspring1.metadata.parent_ids) == 2

    @pytest.mark.asyncio
    async def test_elitism_preservation_deterministic(self, deterministic_evolution_config, deterministic_population, deterministic_fitness_function):
        """Test that elitism preserves elite genomes deterministically."""
        config = deterministic_evolution_config
        config.elitism_enabled = True
        config.selection_pressure = 0.3  # 30% elite
        config.hpc_mode = False  # Standard mode for clear testing

        engine = EvolutionEngine(config)

        # Set specific fitness scores deterministically
        np.random.seed(500)
        for _i, genome in enumerate(deterministic_population):
            genome.fitness_score = deterministic_fitness_function(genome)

        # Sort population and identify elites
        sorted_pop = sorted(deterministic_population, key=lambda g: g.fitness_score, reverse=True)
        elitism_count = int(config.population_size * config.selection_pressure)
        elite_genomes = sorted_pop[:elitism_count]

        # Store elite genome IDs for verification
        elite_ids = [g.genome_id for g in elite_genomes]
        elite_trait_dict = {g.genome_id: g.traits.copy() for g in elite_genomes}

        # Run one generation of evolution
        await engine.evolve(deterministic_population, deterministic_fitness_function, generations=1)

        # Verify elites are preserved
        preserved_elites = 0
        for elite_id in elite_ids:
            elite_trait_dict[elite_id] = elite_trait_dict[elite_id]  # Refresh reference

            # Find this elite in the new population
            found = False
            for genome in deterministic_population:
                if genome.genome_id == elite_id:
                    # Verify traits are unchanged
                    np.testing.assert_array_equal(
                        genome.traits,
                        elite_trait_dict[elite_id]
                    )
                    found = True
                    preserved_elites += 1
                    break

            assert found, f"Elite genome {elite_id} not found in population"

        # Should preserve at least some elites
        assert preserved_elites >= 1

    def test_elitism_preservation_hpc_mode(self, deterministic_evolution_config, deterministic_population, deterministic_fitness_function):
        """Test that elite genomes remain unchanged during HPC mode in-place mutations."""
        config = deterministic_evolution_config
        config.elitism_enabled = True
        config.selection_pressure = 0.2  # 20% elite
        config.hpc_mode = True  # HPC mode for testing

        engine = EvolutionEngine(config)

        # Set specific fitness scores
        np.random.seed(600)
        for _i, genome in enumerate(deterministic_population):
            genome.fitness_score = deterministic_fitness_function(genome)

        # Sort and identify elites
        sorted_pop = sorted(deterministic_population, key=lambda g: g.fitness_score, reverse=True)
        elitism_count = int(config.population_size * config.selection_pressure)
        elite_genomes = sorted_pop[:elitism_count]

        # Store original elite traits
        elite_original_data = []
        for elite in elite_genomes:
            elite_original_data.append({
                'genome_id': elite.genome_id,
                'traits': elite.traits.copy(),
                'fitness_score': elite.fitness_score
            })

        # Trigger evolution (which may mutate in-place)
        import asyncio
        asyncio.run(engine.evolve(deterministic_population, deterministic_fitness_function, generations=1))

        # Verify each elite genome's traits remain unchanged
        for elite_data in elite_original_data:
            # Find the genome by ID in the population
            found_genome = None
            for genome in deterministic_population:
                if genome.genome_id == elite_data['genome_id']:
                    found_genome = genome
                    break

            assert found_genome is not None, f"Elite {elite_data['genome_id']} not found"

            # Verify traits are unchanged
            np.testing.assert_array_equal(
                found_genome.traits,
                elite_data['traits'],
                err_msg=f"Elite {elite_data['genome_id']} traits were modified during evolution"
            )

    def test_mutation_observable_on_non_elites(self, deterministic_evolution_config, deterministic_population, deterministic_fitness_function):
        """Test that mutation is observable on non-elite genomes."""
        config = deterministic_evolution_config
        config.elitism_enabled = True
        config.selection_pressure = 0.2  # Top 20% are elite

        engine = EvolutionEngine(config)

        # Set fitness scores
        np.random.seed(700)
        for _i, genome in enumerate(deterministic_population):
            genome.fitness_score = deterministic_fitness_function(genome)

        # Sort and identify elites vs non-elites
        sorted_pop = sorted(deterministic_population, key=lambda g: g.fitness_score, reverse=True)
        elitism_count = int(config.population_size * config.selection_pressure)
        set(sorted_pop[:elitism_count])
        non_elite_genomes = sorted_pop[elitism_count:]

        # Store original traits for non-elites
        non_elite_original_traits = {
            genome.genome_id: genome.traits.copy()
            for genome in non_elite_genomes
        }

        # Run evolution
        import asyncio
        asyncio.run(engine.evolve(deterministic_population, deterministic_fitness_function, generations=1))

        # Verify mutations occurred in non-elites
        mutations_observed = 0
        for genome in deterministic_population:
            if genome.genome_id in non_elite_original_traits:
                original_traits = non_elite_original_traits[genome.genome_id]
                if not np.array_equal(genome.traits, original_traits):
                    mutations_observed += 1

        # Should observe mutations in some non-elites (at least 40%)
        mutation_rate = mutations_observed / len(non_elite_genomes)
        assert mutation_rate >= 0.4, f"Mutation rate too low: {mutation_rate}"

    def test_tournament_selection_deterministic(self, deterministic_population):
        """Test that tournament selection produces deterministic results."""
        # Set deterministic fitness scores
        np.random.seed(800)
        for i, genome in enumerate(deterministic_population):
            genome.fitness_score = i / len(deterministic_population)

        # Run tournament selection multiple times with same seed
        selections1 = []
        for _ in range(10):
            selected = GeneticOperators.tournament_selection(
                deterministic_population,
                tournament_size=3,
                seed=900
            )
            selections1.append(selected.genome_id)

        # Reset and run again
        np.random.seed(800)  # Reset to same state
        selections2 = []
        for _ in range(10):
            selected = GeneticOperators.tournament_selection(
                deterministic_population,
                tournament_size=3,
                seed=900
            )
            selections2.append(selected.genome_id)

        # Selections should be identical
        assert selections1 == selections2

    def test_evolution_deterministic_progress(self, deterministic_evolution_config, deterministic_population, deterministic_fitness_function):
        """Test that evolution progress is deterministic with same seeds."""
        config = deterministic_evolution_config

        # Run evolution twice with same seeds
        engine1 = EvolutionEngine(config, seed=999)
        engine2 = EvolutionEngine(config, seed=999)

        import asyncio

        # Run first evolution
        np.random.seed(1000)
        pop1 = [copy.deepcopy(g) for g in deterministic_population]
        result1 = asyncio.run(engine1.evolve(pop1, deterministic_fitness_function))
        
        # Reset and run second evolution
        np.random.seed(1000)
        pop2 = [copy.deepcopy(g) for g in deterministic_population]
        result2 = asyncio.run(engine2.evolve(pop2, deterministic_fitness_function))        # Results should be deterministic
        assert result1.genome_id == result2.genome_id
        assert abs(result1.fitness_score - result2.fitness_score) < 1e-6

        # Evolution history should match
        assert len(engine1.evolution_history) == len(engine2.evolution_history)
        for record1, record2 in zip(engine1.evolution_history, engine2.evolution_history):
            assert abs(record1['best_fitness'] - record2['best_fitness']) < 1e-6

    def test_mutation_bounds_enforcement(self):
        """Test that mutation respects trait bounds."""
        np.random.seed(1100)
        genome = ConversationalGenome(
            genome_id="bounds_test",
            traits={'empathy': 0.1, 'technical_knowledge': 0.9, 'creativity': 0.5}
        )

        # Apply high mutation rate
        GeneticOperators.mutate(genome, mutation_rate=1.0, seed=1200)

        # All traits should remain within bounds [0.0, 1.0]
        for trait_name, trait_value in genome.traits.items():
            assert 0.0 <= trait_value <= 1.0, f"Trait {trait_name} out of bounds: {trait_value}"

    @pytest.mark.asyncio
    async def test_population_replacement_deterministic(self, deterministic_evolution_config, deterministic_fitness_function):
        """Test that population replacement is deterministic."""
        config = deterministic_evolution_config
        config.population_size = 15  # Smaller for testing

        engine = EvolutionEngine(config, seed=888)

        # Generate fresh population for run 1
        pop1 = generate_population(config.population_size, seed=456)

        # Set deterministic fitness
        np.random.seed(1300)
        for genome in pop1:
            genome.fitness_score = deterministic_fitness_function(genome)

        # Run evolution
        np.random.seed(1400)
        await engine.evolve(pop1, deterministic_fitness_function, generations=1)

        # Store result population
        result_population = [copy.deepcopy(g) for g in pop1]

        # Reset and run again
        engine = EvolutionEngine(config, seed=888)
        deterministic_population2 = generate_population(config.population_size, seed=456)
        np.random.seed(1300)
        for genome in deterministic_population2:
            genome.fitness_score = deterministic_fitness_function(genome)

        np.random.seed(1400)
        await engine.evolve(deterministic_population2, deterministic_fitness_function, generations=1)

        # Populations should be similar (allowing for floating point differences)
        assert len(result_population) == len(deterministic_population2)

        # Compare fitness scores
        fitness1 = sorted([g.fitness_score for g in result_population])
        fitness2 = sorted([g.fitness_score for g in deterministic_population2])

        for f1, f2 in zip(fitness1, fitness2):
            assert abs(f1 - f2) < 1e-6

    def test_evolution_with_varying_mutation_rates(self):
        """Test that different mutation rates produce deterministically different results."""
        base_genome = ConversationalGenome(
            genome_id="base",
            traits={'empathy': 0.5, 'technical_knowledge': 0.5, 'creativity': 0.5}
        )

        # Test low mutation rate
        np.random.seed(1500)
        low_mut_genome = ConversationalGenome(
            genome_id="low_mut",
            traits={'empathy': 0.5, 'technical_knowledge': 0.5, 'creativity': 0.5}
        )
        GeneticOperators.mutate(low_mut_genome, mutation_rate=0.01, seed=1600)

        # Test high mutation rate
        np.random.seed(1500)
        high_mut_genome = ConversationalGenome(
            genome_id="high_mut",
            traits={'empathy': 0.5, 'technical_knowledge': 0.5, 'creativity': 0.5}
        )
        GeneticOperators.mutate(high_mut_genome, mutation_rate=0.5, seed=1600)

        # Low mutation should result in fewer/smaller changes
        changes_low = sum(
            1 for trait in base_genome.traits
            if abs(low_mut_genome.traits[trait] - base_genome.traits[trait]) > 1e-6
        )

        changes_high = sum(
            1 for trait in base_genome.traits
            if abs(high_mut_genome.traits[trait] - base_genome.traits[trait]) > 1e-6
        )

        assert changes_high >= changes_low, "Higher mutation rate should produce more changes"
