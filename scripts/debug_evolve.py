import asyncio
import sys

sys.path.insert(0, ".")
from phylogenic.evolution import EvolutionConfig, EvolutionEngine
from tests.test_utils import generate_fitness_function


async def main():
    trials = 200
    success = 0
    for _i in range(trials):
        cfg = EvolutionConfig(population_size=20, generations=3, mutation_rate=0.1)
        engine = EvolutionEngine(cfg)
        population = engine.initialize_population()

        initial_traits = {g.genome_id: g.traits.copy() for g in population[:5]}

        await engine.evolve(population, generate_fitness_function(), generations=3)

        mutated = False
        for g in population[:5]:
            if g.genome_id in initial_traits:
                orig = initial_traits[g.genome_id]
                for t in g.traits:
                    if abs(g.traits[t] - orig[t]) > 1e-6:
                        mutated = True
                        break
                if mutated:
                    break

        if mutated:
            success += 1

    print(f"Out of {trials} trials, initial sample mutated in {success} runs")


if __name__ == "__main__":
    asyncio.run(main())
