# Phylogenic Documentation

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-AGPL%20v3-blue.svg)](https://github.com/phylogenic-ai/phylogenic/blob/main/LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-phylogenic-blue)](https://pypi.org/project/phylogenic/)
[![Documentation Status](https://readthedocs.org/projects/phylogenic/badge/?version=latest)](https://phylogenic.readthedocs.io/en/latest/?badge=latest)

**Beyond Prompt Engineering. Evolve Genetically Optimized Personalities with Liquid Memory & Real LLM Conversations.**

---

## Don't Write Prompts. Breed Agents.

Traditional Agents are brittle. They hallucinate, drift, and forget.

**Phylogenic changes the substrate.**

We replaced static prompts with **8-Trait Genetic Code** and **Liquid Neural Networks (LNNs)**.

---

## The Problem

**Prompting is guessing.** You change one word, the whole personality breaks.

- ❌ Brittle system prompts
- ❌ No memory coherence
- ❌ Manual trial-and-error optimization
- ❌ Agents that drift over time

## The Solution

**Phylogenic treats Agent personalities like DNA, not text.**

Instead of writing prompts, you define a **Genome** with 8 evolved traits:

```python
from phylogenic import ConversationalGenome, create_agent, AgentConfig

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
Phylogenic connects to **actual AI services** - not mocks or simulations:

```python
# Multi-provider support
config = AgentConfig(llm_provider="ollama", model_name="llama2")
agent = NLPAgent(genome, config)
await agent.initialize()  # Makes real HTTPS calls

async for chunk in agent.chat("What's the weather like?"):
    print(chunk, end='')  # Real AI-generated response
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
# Automated personality optimization
```

### 🧠 Kraken Liquid Neural Networks

Temporal memory via Liquid Neural Networks:

```python
kraken = KrakenLNN(reservoir_size=100)
context = await kraken.process_sequence(conversation)
# <10ms latency, adaptive dynamics
```

---

## Installation

```bash
pip install phylogenic

# With LLM providers
pip install phylogenic[openai]    # OpenAI GPT
pip install phylogenic[anthropic] # Anthropic Claude
pip install phylogenic[ollama]    # Ollama (local/cloud)
```

---

## Quick Links

- [**Evolution Guide**](evolution.md) - Learn genetic algorithms for personality optimization
- [**API Reference**](api.md) - Complete REST API specifications
- [**Kraken LNN**](kraken_lnn.md) - Liquid neural network documentation
- [**Configuration**](configuration.md) - Setup and configuration options
- [**LLM Integration**](LLM_INTEGRATION.md) - Multi-provider AI connectivity
- [**Testing**](TESTING.md) - Testing frameworks and benchmarks
- [**Whitepaper**](whitepaper/phylogenic_whitepaper.md) - Research and technical details

---

**Made with genetic algorithms and liquid neural networks**

**Don't write prompts. Breed agents.** 🧬

---

## License

GNU AGPL v3 - see [LICENSE](../LICENSE)

**Commercial Use Note:** This project uses the AGPL v3 license with a commercial exception available. See [COMMERCIAL_LICENSE.txt](../COMMERCIAL_LICENSE.txt) for details on commercial licensing.
