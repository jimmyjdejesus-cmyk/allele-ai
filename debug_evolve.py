from allele import EvolutionConfig, EvolutionEngine
from tests.test_utils import generate_fitness_function

config = EvolutionConfig(population_size=20, generations=5, mutation_rate=0.1, crossover_rate=0.8)
engine = EvolutionEngine(config)
pop = engine.initialize_population()
initial = {g.genome_id: g.traits.copy() for g in pop[:5]}
print('initial_ids', list(initial.keys()))
import asyncio

best = asyncio.get_event_loop().run_until_complete(engine.evolve(pop, generate_fitness_function(), generations=3))
print('\nFinal first 5:')
for g in pop[:5]:
    same = g.genome_id in initial
    changed = False
    if same:
        orig = initial[g.genome_id]
        changed = any(abs(g.traits[t]-orig[t])>1e-6 for t in g.traits)
    print(g.genome_id, 'in_initial=', same, 'changed=', changed)
print('\nAny mutated among first5 present? ', any((g.genome_id in initial and any(abs(g.traits[t]-initial[g.genome_id][t])>1e-6 for t in g.traits)) for g in pop[:5]))
