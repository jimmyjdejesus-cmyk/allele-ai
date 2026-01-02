# Evolution Guide

The Phylogenic Evolution Engine implements genetic algorithms for breeding and evolving conversational genomes with expressed traits like empathy, creativity, technical knowledge, and engagement. This guide covers the core concepts, usage patterns, and configuration options.

## Core Concepts

### Genetic Encoding

Phylogenic genes encode personality as 8 quantized traits (0.0 to 1.0):

- **empathy**: Emotional understanding and compassion
- **technical_knowledge**: Technical depth and accuracy
- **creativity**: Innovative problem-solving
- **conciseness**: Balance of thoroughness vs brevity
- **context_awareness**: Conversation history retention
- **engagement**: Conversational energy and enthusiasm
- **adaptability**: Style flexibility
- **personability**: Friendliness and human-like interaction

### Evolutionary Process

The engine uses standard genetic operators:

1. **Selection**: Tournament selection picks fittest parents
2. **Crossover**: Combines genetic material from two parents
3. **Mutation**: Random trait variations for exploration
4. **Elitism**: Preserves best individuals across generations

## Basic Usage

```python
from phylogenic import EvolutionEngine, EvolutionConfig, ConversationalGenome

# Configure evolution parameters
config = EvolutionConfig(
    population_size=50,      # Number of genomes
    generations=20,          # Evolution iterations
    mutation_rate=0.1,       # Gene mutation probability
    crossover_rate=0.8,      # Breeding probability
    elitism_enabled=True     # Preserve best genomes
)

# Initialize evolution engine
engine = EvolutionEngine(config)

# Create initial random population
population = engine.initialize_population()

# Define fitness function (real-world performance metric)
def conversation_fitness(genome):
    """
    Evaluate genome by testing conversation quality.
    In practice, this would measure user satisfaction,
    response coherence, task completion, etc.
    """
    # Mock fitness based on some optimal trait balance
    traits = genome.traits
    score = (
        traits['empathy'] * 0.2 +
        traits['technical_knowledge'] * 0.25 +
        traits['engagement'] * 0.15 +
        traits['conciseness'] * 0.2 +
        traits['context_awareness'] * 0.2
    )
    return score

# Evolve toward optimal fitness
best_genome = await engine.evolve(population, conversation_fitness)

print(f"Best fitness: {best_genome.fitness_score}")
print(f"Winning traits: {best_genome.traits}")
```

## Advanced Configuration

### HPC Mode

For high-performance computing environments, enable HPC optimizations:

```python
config = EvolutionConfig(
    hpc_mode=True,           # Enable performance optimizations
    immutable_evolution=False # In-place mutations (default)
)

# Population mutations happen in-place for speed
# Useful when memory footprint is critical
```

### Immutable Evolution

For reproducible and thread-safe evolution:

```python
config = EvolutionConfig(
    immutable_evolution=True  # Create new genome objects each generation
)

# Each generation creates fresh copies
# Prevents side effects but uses more memory
```

### Custom Selection Strategies

Override default tournament selection:

```python
class CustomSelectors(GeneticOperators):
    @staticmethod
    def tournament_selection(population, tournament_size=5):
        # Custom selection logic
        return max(population, key=lambda g: g.fitness_score * custom_weight)

# Use custom selector in evolution
config = EvolutionConfig(...)
engine = EvolutionEngine(config)
# Custom selector will be used automatically
```

## Evolution Monitoring

Track evolution progress and convergence:

```python
# Access evolution history
for gen_stats in engine.evolution_history:
    print(f"Generation {gen_stats['generation']}: "
          f"best={gen_stats['best_fitness']:.3f}, "
          f"avg={gen_stats['avg_fitness']:.3f}, "
          f"diversity={gen_stats['diversity']:.3f}")
```

## Fitness Function Design

Effective fitness functions are crucial for successful evolution:

### Conversation Quality Metrics

```python
def evaluate_conversation_quality(genome):
    """
    Test genome in actual conversation scenarios
    """
    # Simulate conversations with different personas
    scenarios = [
        ("technical_question", "How does recursion work?"),
        ("emotional_support", "I'm feeling overwhelmed"),
        ("creative_task", "Design a new app idea")
    ]

    total_score = 0
    for context, message in scenarios:
        response = await generate_response(genome, message, context)

        # Evaluate response quality
        coherence = measure_coherence(response)
        relevance = measure_relevance(response, context)
        trait_alignment = measure_trait_expression(genome, response)

        score = (coherence + relevance + trait_alignment) / 3
        total_score += score

    return total_score / len(scenarios)
```

### Multi-Objective Fitness

Balance competing traits:

```python
def balanced_fitness(genome):
    """
    Optimize for multiple objectives simultaneously
    """
    traits = genome.traits

    # Define optimal ranges for customer support agent
    objectives = {
        'empathy': (0.8, 1.0),          # High empathy crucial
        'technical_knowledge': (0.6, 0.8), # Moderate technical depth
        'conciseness': (0.7, 0.9),        # Clear responses
        'engagement': (0.7, 0.9),         # Friendly interaction
    }

    score = 0
    for trait, (min_val, max_val) in objectives.items():
        val = traits[trait]
        if min_val <= val <= max_val:
            score += 1.0
        else:
            # Penalty for deviation
            distance = min(abs(val - min_val), abs(val - max_val))
            score += max(0, 1.0 - distance)

    return score / len(objectives)
```

## Population Management

### Initialization Strategies

```python
# Random initialization (default)
population = engine.initialize_population()

# Custom base traits
population = engine.initialize_population(
    base_traits={
        'empathy': 0.8,
        'technical_knowledge': 0.6,
        'creativity': 0.4,
        # Other defaults will be used
    }
)

# Preserve specific genomes
population.append(my_custom_genome)
```

### Diversity Maintenance

Monitor and maintain genetic diversity:

```python
def population_diversity(population):
    """Calculate trait variance across population"""
    trait_values = {}
    for trait in ConversationalGenome.DEFAULT_TRAITS:
        values = [g.traits[trait] for g in population]
        trait_values[trait] = np.var(values)

    return np.mean(list(trait_values.values()))

# Check diversity before evolution
if population_diversity(population) < 0.01:
    print("Warning: Low genetic diversity detected")
```

## Performance Considerations

### Scaling Population Size

- **Small populations (<50)**: Fast but may converge prematurely
- **Medium populations (50-200)**: Good balance of speed and diversity
- **Large populations (200+)**: Better exploration but slower evolution

### Memory Usage

```python
# Estimate memory for population
genome_size_kb = 2  # Approximate per genome
population_memory_mb = (config.population_size * genome_size_kb) / 1024

print(f"Population memory: {population_memory_mb:.1f} MB")
```

### Parallel Evaluation

For large populations, parallelize fitness evaluation:

```python
import asyncio

async def evaluate_population_parallel(population, fitness_fn):
    """Evaluate all genomes concurrently"""
    tasks = [asyncio.create_task(evaluate_genome(g, fitness_fn))
             for g in population]

    results = await asyncio.gather(*tasks)
    return results

# Use in evolution loop
async def custom_evolution(engine, population, fitness_fn):
    for generation in range(engine.config.generations):
        # Parallel fitness evaluation
        await evaluate_population_parallel(population, fitness_fn)

        # Rest of evolution logic...
```

## Troubleshooting

### Common Issues

**No Fitness Improvement**
- Increase mutation rate
- Check fitness function validity
- Ensure population diversity

**Evolution Too Slow**
- Reduce population size
- Enable HPC mode
- Lower generation count

**Premature Convergence**
- Increase mutation rate
- Add more genetic diversity
- Use larger population

### Debugging Evolution

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor trait distributions
def log_trait_stats(population, generation):
    stats = {}
    for trait in ConversationalGenome.DEFAULT_TRAITS:
        values = [g.traits[trait] for g in population]
        stats[trait] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    print(f"Generation {generation} trait stats:")
    for trait, data in stats.items():
        print(f"  {trait}: {data['mean']:.3f} ± {data['std']:.3f}")
```

## Use Cases

### Customer Support Agent Breeding

```python
# Breed agents that handle customer queries effectively
config = EvolutionConfig(
    population_size=100,
    generations=30,
    mutation_rate=0.15,
    elitism_enabled=True
)

def support_fitness(genome):
    return evaluate_support_conversations(genome)

# Result: Optimized mix of empathy + technical knowledge
```

### Creative Content Generation

```python
# Evolve creative writing assistants
config = EvolutionConfig(
    population_size=50,
    generations=25,
    mutation_rate=0.2  # Higher creativity exploration
)

def creative_fitness(genome):
    return evaluate_creative_output(genome)
```

### Adaptive Learning Systems

```python
# Continuously evolve agents based on user feedback
def online_fitness(genome):
    # Real-time user ratings and engagement metrics
    return genome.user_satisfaction_score

# Continuous evolution
while True:
    feedback = collect_user_feedback()
    update_genome_fitness(population, feedback)

    best = await engine.evolve(population, online_fitness, generations=1)
```

The evolution engine provides automatic optimization of conversational traits, removing manual prompt tuning and enabling dynamic adaptation to new requirements.
