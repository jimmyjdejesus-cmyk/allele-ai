# ALLELE
## Phylogenic AI Agents

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-AGPL%20v3-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-allele-blue)](https://pypi.org/project/allele/)
[![Pages Deploy](https://github.com/jimmyjdejesus-cmyk/allele/actions/workflows/pages.yml/badge.svg?branch=main)](https://github.com/jimmyjdejesus-cmyk/allele/actions/workflows/pages.yml)
[![Documentation Status](https://readthedocs.org/projects/allele/badge/?version=latest)](https://allele.readthedocs.io/en/latest/?badge=latest)
[![LLM Ready](https://img.shields.io/badge/LLM-Ready-green.svg)](docs/LLM_INTEGRATION.md)
[![Real API Tested](https://img.shields.io/badge/Real%20API-Tested-brightgreen.svg)](docs/REAL_INTEGRATION_TESTING.md)

**[Website](https://jimmyjdejesus-cmyk.github.io/allele-ai/) | [Documentation](https://jimmyjdejesus-cmyk.github.io/allele-ai/) | [GitHub](https://github.com/jimmyjdejesus-cmyk/allele)**

**Beyond Prompt Engineering. Evolve Genetically Optimized Personalities with Liquid Memory & Real LLM Conversations.**

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

## 🚀 **Production-Ready Integration: Real AI, Real Results**

### **Real LLM Conversational AI**
Allele connects to **actual AI services** - not mocks or simulations:

```python
# Real Ollama Cloud integration
config = AgentConfig(llm_provider="ollama", model_name="llama2")
agent = NLPAgent(genome, config)
await agent.initialize()  # Makes real HTTPS call to ollama.com

async for chunk in agent.chat("What's the weather like?"):
    print(chunk, end='')  # Real AI-generated response
```

### **Multi-Provider Support**
- **OpenAI GPT** (ChatGPT models via API)
- **Ollama Local** (Run models on your hardware)
- **Ollama Cloud** (Cloud-hosted models with authentication)
- **Anthropic Claude** (Coming soon)

### **Behavioral AI: LLMs That *Change* Behavior**
** Allele doesn't just call LLMs - it *transforms* them:**

```python
# Traditional prompting: Static text
"You are helpful and creative..."

# Allele: Dynamic personality injection
# Every LLM call gets genome-based system prompts that modify behavior
system_prompt = agent._create_system_prompt()
# OUTPUT: "You are an AI with emotional understanding (0.9/1.0),
#         technical expertise (0.8/1.0), creativity (0.7/1.0)..."
```

### **Verified Production Testing**
```bash
# Real API calls, no mocks
pytest tests/test_llm_integration.py::test_ollama_cloud_real_chat -xvs
# ✅ Validates actual personality changes via LLM behavior
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

### Quick Configuration

Allele uses **centralized configuration** for easy customization:

```python
from allele import settings, AgentConfig

# Use defaults
config = AgentConfig.from_settings()

# Or override via environment variables
# AGENT__MODEL_NAME=gpt-4-turbo
# AGENT__TEMPERATURE=0.9
```

See [Configuration Guide](docs/configuration.md) for details.

### External dependency: OpenSpec 🔧

This project used to include `OpenSpec` as a git submodule, but it is now maintained and distributed separately as an npm package. Please install it globally on your machine (it is not tracked in this repository):

```bash
# with npm
npm install -g openspec

# with pnpm
pnpm add -g openspec
```

OpenSpec requires Node.js >= 20.19.0.

**HPC mode:** Allele defaults to an in-place mutation strategy for the evolution engine to favor speed and low memory usage. If you need immutable behavior for reproducibility, set `EVOLUTION__IMMUTABLE_EVOLUTION=true` or use `EvolutionConfig(immutable_evolution=True)`.

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
- **Code Quality**: 8.83/10, 95.5% tests passing (42/44 tests passing)
- **Kraken Determinism**: 100% determinism test suite success (12/12 tests)
- **Biological Realism**: Enhanced liquid neural network accuracy preserved

---

## Use Cases

- 🏥 Healthcare: High empathy + medical knowledge
- 💼 Sales: High engagement + persuasion
- 👨‍💻 Dev Tools: High technical + conciseness
- 🎓 Education: High adaptability + patience
- 🔒 Security: High precision + context awareness

---

## Documentation

- [Configuration Guide](docs/configuration.md) - **Start here for setup**
- [Real Integration Testing](docs/REAL_INTEGRATION_TESTING.md) - **Production validation with real AI**
- [LLM Integration Guide](docs/LLM_INTEGRATION.md) - **Multi-provider AI connectivity**
- [API Reference](docs/api.md)
- [Evolution Guide](docs/evolution.md)
- [Kraken LNN](docs/kraken_lnn.md)
- [Testing Guide](docs/TESTING.md)
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

GNU AGPL v3 - see [LICENSE](LICENSE)

**Commercial Use Note:** This project uses the AGPL v3 license with a commercial exception available. See [COMMERCIAL_LICENSE.txt](COMMERCIAL_LICENSE.txt) for details on commercial licensing.

For academic/research use, the AGPL v3 terms are automatically satisfied without requiring a commercial license.

---

## Links

- **Homepage**: [allele.ai](https://allele-ai.github.io/allele/)
- **Documentation**: [docs.allele.ai](https://allele.readthedocs.io/)
- **GitHub**: [github.com/allele-ai/allele](https://github.com/allele-ai/allele)
- **PyPI**: [pypi.org/project/allele](https://pypi.org/project/allele/)
- **Issues**: [github.com/allele-ai/allele/issues](https://github.com/allele-ai/allele/issues)

---

**Made with genetic algorithms and liquid neural networks**

**Don't write prompts. Breed agents.** 🧬
