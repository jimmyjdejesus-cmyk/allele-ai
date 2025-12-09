import pytest
import numpy as np

from allele.evolution import EvolutionConfig, EvolutionEngine
from allele.genome import ConversationalGenome


def _make_population(size=4):
    pop = []
    base_traits = {t: 0.5 for t in ConversationalGenome.DEFAULT_TRAITS.keys()}
    for i in range(size):
        pop.append(ConversationalGenome(genome_id=f"g_{i}", traits=base_traits.copy()))
    return pop


@pytest.mark.asyncio
async def test_evolution_inplace_default():
    config = EvolutionConfig(population_size=4, generations=1, mutation_rate=1.0, crossover_rate=0.0, immutable_evolution=False, hpc_mode=True)
    engine = EvolutionEngine(config)
    population = _make_population(4)
    original_ids = [id(g) for g in population]
    original_traits = [g.traits.copy() for g in population]
    original_objects = population.copy()

    best = await engine.evolve(population, lambda g: 1.0)

    # At least one original object reference should be retained in in-place mode
    new_ids = [id(g) for g in population]
    assert any(o == n for o in original_ids for n in new_ids)

    # Expect traits to have mutated in-place for at least one preserved object
    assert any(population[i].traits != original_traits[i] for i in range(len(population)))


@pytest.mark.asyncio
async def test_evolution_immutable_mode():
    config = EvolutionConfig(population_size=4, generations=1, mutation_rate=1.0, crossover_rate=0.0, immutable_evolution=True, hpc_mode=False)
    engine = EvolutionEngine(config)
    population = _make_population(4)
    original_ids = [id(g) for g in population]
    original_traits = [g.traits.copy() for g in population]
    original_objects = population.copy()

    best = await engine.evolve(population, lambda g: 1.0)

    new_ids = [id(g) for g in population]
    # No original objects should be in the new population (immutable mode creates clones)
    assert not any(o == n for o in original_ids for n in new_ids)

    # Original objects should remain unchanged
    # Verify original objects kept by caller remained unchanged
    assert all(original_objects[i].traits == original_traits[i] for i in range(len(original_objects)))
