# Abe-NLP: Genome-Based Conversational AI SDK

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Production-ready SDK for creating genome-based conversational AI agents with evolved traits and Kraken Liquid Neural Networks.**

## 🌟 Features

- **🧬 Genome-Based Personalities**: Define AI agents using 8 evolved conversational traits
- **🧠 Kraken LNN**: Advanced Liquid Neural Network for temporal sequence processing
- **⚡ Evolutionary Optimization**: Genetic algorithms for automatic trait improvement
- **🔌 LLM Agnostic**: Works with OpenAI, Anthropic, Ollama, or any LLM provider
- **📊 Rich Type System**: Comprehensive type hints for excellent IDE support
- **🚀 Production Ready**: Battle-tested, fully documented, with comprehensive tests

## 📦 Installation

### Basic Installation

```bash
pip install abe-nlp
```

### With LLM Provider Support

```bash
# For OpenAI
pip install abe-nlp[openai]

# For Anthropic
pip install abe-nlp[anthropic]

# For Ollama
pip install abe-nlp[ollama]

# Install all providers
pip install abe-nlp[all]
```

### Development Installation

```bash
git clone https://github.com/bravetto/Abe-NLP.git
cd Abe-NLP
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Basic Usage

```python
from abe_nlp import ConversationalGenome, create_agent, AgentConfig

# Create genome with desired personality traits
genome = ConversationalGenome(
    genome_id="my_agent",
    traits={
        'empathy': 0.9,              # High emotional understanding
        'technical_knowledge': 0.95, # Expert technical depth
        'creativity': 0.7,           # Moderately creative
        'conciseness': 0.8,          # Concise responses
        'context_awareness': 0.9,    # Strong memory
        'engagement': 0.8,           # Engaging personality
        'adaptability': 0.75,        # Learns well
        'personability': 0.85        # Friendly and approachable
    }
)

# Create agent
config = AgentConfig(model_name="gpt-4", temperature=0.7)
agent = await create_agent(genome, config)

# Start conversation
async for response in agent.chat("Explain quantum computing"):
    print(response, end='')
```

### Evolution Example

```python
from abe_nlp import EvolutionEngine, EvolutionConfig

# Configure evolution
config = EvolutionConfig(
    population_size=50,
    generations=20,
    mutation_rate=0.1,
    crossover_rate=0.8
)

# Initialize engine
engine = EvolutionEngine(config)

# Create initial population
population = engine.initialize_population()

# Define fitness function
def fitness_fn(genome):
    # Evaluate genome performance
    score = evaluate_conversations(genome)
    return score

# Evolve population
best_genome = await engine.evolve(population, fitness_fn)

print(f"Best genome traits: {best_genome.traits}")
print(f"Fitness score: {best_genome.fitness_score}")
```

### Kraken LNN Processing

```python
from abe_nlp import KrakenLNN

# Initialize Kraken Liquid Neural Network
kraken = KrakenLNN(
    reservoir_size=100,
    connectivity=0.1
)

# Process temporal sequence
sequence = [0.5, 0.3, 0.8, 0.6, 0.2]
result = await kraken.process_sequence(sequence)

print(f"Liquid outputs: {result['liquid_outputs']}")
print(f"Processing time: {result['processing_time']}s")
```

## 🧬 The 8 Conversational Traits

Each genome is defined by 8 core traits (values from 0.0 to 1.0):

| Trait | Description | Example Use Case |
|-------|-------------|------------------|
| **Empathy** | Emotional understanding and response | Customer support, counseling |
| **Engagement** | Conversational enthusiasm | Education, entertainment |
| **Technical Knowledge** | Technical depth and accuracy | Developer tools, research |
| **Creativity** | Original problem-solving | Content creation, brainstorming |
| **Conciseness** | Brevity while staying informative | Quick answers, summaries |
| **Context Awareness** | Memory and conversation history | Long conversations, personalization |
| **Adaptability** | Learning and style adaptation | Multi-domain assistants |
| **Personability** | Friendliness and approachability | Social interaction, onboarding |

## 📚 Core Concepts

### Genome

The fundamental building block of Abe-NLP. A genome defines an agent's personality through trait expression levels.

```python
from abe_nlp import ConversationalGenome

# Create default genome
genome = ConversationalGenome("agent_001")

# Get trait value
empathy_level = genome.get_trait_value('empathy')

# Mutate trait
genome.mutate_trait('creativity', mutation_strength=0.1)

# Breed genomes
offspring = genome1.crossover(genome2)
```

### Evolution

Automatically improve genomes through genetic algorithms:

```python
from abe_nlp import EvolutionEngine, GeneticOperators

# Tournament selection
parent = GeneticOperators.tournament_selection(population)

# Crossover
child = GeneticOperators.crossover(parent1, parent2)

# Mutation
GeneticOperators.mutate(child, mutation_rate=0.1)
```

### Kraken LNN

Advanced neural processing for temporal coherence:

```python
from abe_nlp import KrakenLNN, LiquidDynamics

# Custom dynamics
dynamics = LiquidDynamics(
    viscosity=0.15,
    temperature=1.2,
    turbulence=0.08
)

kraken = KrakenLNN(dynamics=dynamics)
```

## 🎯 Use Cases

### Customer Support Bot

```python
# High empathy, context awareness, conciseness
support_genome = ConversationalGenome(
    "support_bot",
    traits={
        'empathy': 0.95,
        'context_awareness': 0.9,
        'conciseness': 0.85,
        'personability': 0.9
    }
)
```

### Technical Assistant

```python
# High technical knowledge, accuracy, lower creativity
tech_genome = ConversationalGenome(
    "tech_assistant",
    traits={
        'technical_knowledge': 0.95,
        'conciseness': 0.8,
        'creativity': 0.4,
        'context_awareness': 0.85
    }
)
```

### Creative Writing Partner

```python
# High creativity, engagement, moderate technical
creative_genome = ConversationalGenome(
    "creative_writer",
    traits={
        'creativity': 0.95,
        'engagement': 0.9,
        'technical_knowledge': 0.5,
        'empathy': 0.8
    }
)
```

## 📖 Documentation

- [API Reference](docs/api.md)
- [Evolution Guide](docs/evolution.md)
- [LNN Deep Dive](docs/kraken_lnn.md)
- [Examples](examples/)
- [Contributing](CONTRIBUTING.md)

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=abe_nlp --cov-report=html

# Run specific test
pytest tests/test_genome.py -v
```

## 🛠️ Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black src/ tests/

# Lint
pylint src/

# Type check
mypy src/
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌐 Links

- **GitHub**: [https://github.com/bravetto/Abe-NLP](https://github.com/bravetto/Abe-NLP)
- **Documentation**: [https://abe-nlp.readthedocs.io](https://abe-nlp.readthedocs.io)
- **Issues**: [https://github.com/bravetto/Abe-NLP/issues](https://github.com/bravetto/Abe-NLP/issues)
- **PyPI**: [https://pypi.org/project/abe-nlp/](https://pypi.org/project/abe-nlp/)

## ✨ Authors

**Bravetto AI Systems**
- GitHub: [@bravetto](https://github.com/bravetto)
- Email: contact@bravetto.ai

## 🙏 Acknowledgments

- Based on cutting-edge research in neuromorphic AI
- Inspired by biological neural systems and evolutionary algorithms
- Built with modern Python best practices

---

**Made with ❤️ by Bravetto AI Systems**

