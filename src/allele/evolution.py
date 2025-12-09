"""
Evolution engine for Allele genomes.

This module implements genetic algorithms for evolving conversational genomes
with support for selection, crossover, mutation, and fitness evaluation.

Author: Bravetto AI Systems
Version: 1.0.0
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np

from .genome import ConversationalGenome
from .types import FitnessMetrics
from .exceptions import EvolutionError


@dataclass
class EvolutionConfig:
    """Configuration for evolution engine.

    Attributes:
        population_size: Number of genomes in population
        generations: Maximum number of generations to evolve
        mutation_rate: Probability of gene mutation (0.0-1.0)
        crossover_rate: Probability of parent crossover (0.0-1.0)
        selection_pressure: Fraction of best individuals to keep (0.0-1.0)
        elitism_enabled: Whether to preserve best genomes
        tournament_size: Size of tournament for selection
    """
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_pressure: float = 0.2
    elitism_enabled: bool = True
    tournament_size: int = 3


class GeneticOperators:
    """Genetic operators for evolution."""

    @staticmethod
    def tournament_selection(
        population: List[ConversationalGenome],
        tournament_size: int = 3
    ) -> ConversationalGenome:
        """Tournament selection for parent selection.

        Args:
            population: Population of genomes
            tournament_size: Number of genomes in tournament

        Returns:
            Selected genome
        """
        tournament = np.random.choice(population, tournament_size, replace=False)
        return max(tournament, key=lambda g: g.fitness_score)

    @staticmethod
    def crossover(
        parent1: ConversationalGenome,
        parent2: ConversationalGenome
    ) -> ConversationalGenome:
        """Perform crossover between two genomes.

        Args:
            parent1: First parent genome
            parent2: Second parent genome

        Returns:
            Offspring genome
        """
        return parent1.crossover(parent2)

    @staticmethod
    def mutate(
        genome: ConversationalGenome,
        mutation_rate: float = 0.1
    ) -> None:
        """Apply mutation to genome.

        Args:
            genome: Genome to mutate
            mutation_rate: Probability of mutation
        """
        genome.mutate_all_traits(mutation_rate)


class EvolutionEngine:
    """Evolution engine for conversational genomes.

    Example:
        >>> config = EvolutionConfig(population_size=50, generations=20)
        >>> engine = EvolutionEngine(config)
        >>> population = engine.initialize_population()
        >>> best = await engine.evolve(population, fitness_fn)
    """

    def __init__(self, config: EvolutionConfig):
        """Initialize evolution engine.

        Args:
            config: Evolution configuration
        """
        self.config = config
        self.generation = 0
        self.best_genome: Optional[ConversationalGenome] = None
        self.evolution_history: List[Dict[str, Any]] = []

    def initialize_population(
        self,
        base_traits: Optional[Dict[str, float]] = None
    ) -> List[ConversationalGenome]:
        """Initialize population with diverse genomes.

        Args:
            base_traits: Optional base traits to start from

        Returns:
            List of initialized genomes
        """
        population = []

        # Default base traits
        if base_traits is None:
            base_traits = ConversationalGenome.DEFAULT_TRAITS.copy()

        for i in range(self.config.population_size):
            # Add genetic variation
            individual_traits = {}
            for trait, base_value in base_traits.items():
                # Add random variation (±20%)
                variation = np.random.uniform(-0.2, 0.2)
                individual_traits[trait] = np.clip(base_value + variation, 0.0, 1.0)

            genome = ConversationalGenome(
                genome_id=f"genome_{i:04d}",
                traits=individual_traits
            )
            population.append(genome)

        return population

    async def evolve(
        self,
        population: List[ConversationalGenome],
        fitness_function: Callable[[ConversationalGenome], float],
        generations: Optional[int] = None
    ) -> ConversationalGenome:
        """Evolve population for specified generations.

        Args:
            population: Initial population
            fitness_function: Function to evaluate genome fitness
            generations: Number of generations (uses config if None)

        Returns:
            Best genome after evolution
        """
        if generations is None:
            generations = self.config.generations

        for gen in range(generations):
            # Evaluate fitness
            for genome in population:
                genome.fitness_score = fitness_function(genome)

            # Update best genome
            current_best = max(population, key=lambda g: g.fitness_score)
            if self.best_genome is None or current_best.fitness_score > self.best_genome.fitness_score:
                self.best_genome = current_best

            # Record history
            self.evolution_history.append({
                "generation": gen,
                "best_fitness": self.best_genome.fitness_score,
                "avg_fitness": np.mean([g.fitness_score for g in population]),
                "diversity": self._calculate_diversity(population)
            })

            # Create next generation
            population = self._create_next_generation(population)

            self.generation += 1

        return self.best_genome

    def _create_next_generation(
        self,
        population: List[ConversationalGenome]
    ) -> List[ConversationalGenome]:
        """Create next generation through selection, crossover, and mutation.

        Args:
            population: Current population

        Returns:
            Next generation population
        """
        # Sort by fitness
        population.sort(key=lambda g: g.fitness_score, reverse=True)

        # Elitism - keep top individuals
        next_generation = []
        if self.config.elitism_enabled:
            elitism_count = int(
                self.config.population_size * self.config.selection_pressure
            )
            next_generation = population[:elitism_count]

        # Create offspring
        while len(next_generation) < self.config.population_size:
            # Select parents
            parent1 = GeneticOperators.tournament_selection(
                population, self.config.tournament_size
            )
            parent2 = GeneticOperators.tournament_selection(
                population, self.config.tournament_size
            )

            # Crossover
            if np.random.random() < self.config.crossover_rate:
                offspring = GeneticOperators.crossover(parent1, parent2)
            else:
                offspring = parent1  # Clone if no crossover

            # Mutation
            GeneticOperators.mutate(offspring, self.config.mutation_rate)

            next_generation.append(offspring)

        return next_generation[:self.config.population_size]

    def _calculate_diversity(
        self,
        population: List[ConversationalGenome]
    ) -> float:
        """Calculate genetic diversity in population.

        Args:
            population: Population to analyze

        Returns:
            Diversity score (0.0-1.0)
        """
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

