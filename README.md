# ALLELE
## Phylogenic AI Agents

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-allele-blue)](https://pypi.org/project/allele/)

**Beyond Prompt Engineering. Evolve Genetically Optimized Personalities with Liquid Memory.**

---

## Don't Write Prompts. Breed Agents.

Traditional Agents are brittle. They hallucinate, drift, and forget.

**Allele changes the substrate.**

We replaced static prompts with **8-Trait Genetic Code** and **Liquid Neural Networks (LNNs)**.

---

## The Problem

**Prompting is guessing.** You change one word, the whole personality breaks.

- ❌ Brittle system prompts
- ❌ No memory coherence
- ❌ Manual trial-and-error optimization
- ❌ Agents that drift over time

## The Solution

**Allele treats Agent personalities like DNA, not text.**

Instead of writing prompts, you define a **Genome** with 8 evolved traits:

```python
from allele import ConversationalGenome, create_agent, AgentConfig

# Define personality as genetic code
genome = ConversationalGenome(
    genome_id="support_agent_v1",
    traits={
        'empathy': 0.95,              # High emotional intelligence
        'technical_knowledge': 0.70,  # Moderate technical depth
        'creativity': 0.30,           # Focused responses
        'conciseness': 0.85,          # Brief and clear
        'context_awareness': 0.90,    # Strong memory
        'engagement': 0.85,           # Warm personality
        'adaptability': 0.75,         # Flexible style
        'personability': 0.90         # Friendly demeanor
    }
)

# Create agent from genome
config = AgentConfig(model_name="gpt-4", kraken_enabled=True)
agent = await create_agent(genome, config)

# Chat with genetically-defined personality
async for response in agent.chat("I need help"):
    print(response)
```

---

## Core Innovation

### 🧬 Genetic Personality Encoding

8 quantified personality traits (0.0 to 1.0) define each agent:

- **Empathy** - Emotional understanding
- **Technical Knowledge** - Technical depth
- **Creativity** - Problem-solving novelty
- **Conciseness** - Brevity vs detail
- **Context Awareness** - Memory retention
- **Engagement** - Conversational energy
- **Adaptability** - Style flexibility
- **Personability** - Friendliness

### 🧪 Evolutionary Optimization

```python
# Don't manually tune. Evolve.
engine = EvolutionEngine(config)
population = engine.initialize_population(size=50)

best = await engine.evolve(population, fitness_fn)
# 20 generations → optimized personality
```

### 🧠 Kraken Liquid Neural Networks

Temporal memory via Liquid Neural Networks (not static vectors):

```python
kraken = KrakenLNN(reservoir_size=100)
context = await kraken.process_sequence(conversation)
# <10ms latency, adaptive dynamics
```

---

## Installation

```bash
pip install allele

# With LLM providers
pip install allele[openai]    # OpenAI
pip install allele[anthropic] # Anthropic Claude
pip install allele[ollama]    # Ollama (local)
pip install allele[all]       # All providers
```

---

## Why Allele?

| Feature | Traditional | Allele |
|---------|------------|--------|
| **Personality** | Prompt strings | Genetic code |
| **Optimization** | Manual tweaking | Auto-evolution |
| **Memory** | Vector stores | Liquid neural nets |
| **Reproducibility** | Copy-paste prompts | Version genomes |
| **Explainability** | Black box | Trait values |

---

## Benchmarks

- **Crossover**: <5ms (breeding is cheap)
- **LNN Processing**: <10ms (temporal coherence)
- **Memory**: ~2KB per genome
- **Code Quality**: 8.83/10, 100% tests passing

---

## Use Cases

- 🏥 Healthcare: High empathy + medical knowledge
- 💼 Sales: High engagement + persuasion
- 👨‍💻 Dev Tools: High technical + conciseness
- 🎓 Education: High adaptability + patience
- 🔒 Security: High precision + context awareness

---

## Documentation

- [API Reference](docs/api.md)
- [Evolution Guide](docs/evolution.md)
- [Kraken LNN](docs/kraken_lnn.md)
- [Examples](examples/)

---

## Testing

```bash
pytest                                    # Run all tests
pytest --cov=allele --cov-report=html    # With coverage
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Links

- **GitHub**: [github.com/bravetto/allele](https://github.com/bravetto/allele)
- **PyPI**: [pypi.org/project/allele](https://pypi.org/project/allele/)
- **Issues**: [github.com/bravetto/allele/issues](https://github.com/bravetto/allele/issues)

---

**Made with genetic algorithms and liquid neural networks**

**Don't write prompts. Breed agents.** 🧬
