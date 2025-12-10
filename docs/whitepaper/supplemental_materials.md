# Supplemental Materials: Allele Genome-Based Conversational AI

## Code and Data Availability

All source code, experimental data, and analysis scripts are available at:
**GitHub Repository**: https://github.com/jimmyjdejesus-cmyk/allele
**DOI (forthcoming)**: Zenodo archive pending publication

## Computational Requirements

### Hardware Specifications
During experimental evaluation, all benchmarks were conducted on:
- **CPU**: Intel Core i7-9750H @ 2.60GHz
- **RAM**: 16GB DDR4-2666
- **GPU**: NVIDIA GeForce GTX 1650 (4GB VRAM)
- **Storage**: 512GB NVMe SSD

### Software Environment
- **Operating System**: Windows 11 (x64), Ubuntu 22.04 LTS
- **Python Version**: Python 3.14.0 (CPython)
- **Key Dependencies**:
  - numpy==2.3.3
  - pydantic==2.12.1
  - openai==1.109.1
  - httpx==0.28.1

## Detailed Statistical Analysis

### Raw Performance Metrics Dataset

Complete benchmarking data available in `docs/whitepaper/appendix_a_experimental_data.md`:

```
Benchmark Results Archive
├── performance_metrics.csv     (raw timing data)
├── evolutionary_runs.csv       (generation-by-generation fitness)
├── trait_stability.csv        (stability analysis across 100 generations)
├── comparative_baseline.csv   (prompt engineering vs genome comparisons)
└── statistical_tests.R        (R script for all statistical analyses)
```

### Statistical Methodology Details

All hypothesis tests were performed using Python scipy.stats with:
- **Alpha level**: 0.05 (95% confidence)
- **Sampling method**: Bootstrapped confidence intervals (1000 iterations)
- **Effect size calculations**: Cohen's d for paired differences
- **Random seed**: seed=42 for reproducibility

## Trait Implementation Details

### Genome Serialization Format

```
{
  "genome_id": "string",
  "traits": {
    "empathy": 0.8,
    "engagement": 0.7,
    "technical_knowledge": 0.6,
    "creativity": 0.9,
    "conciseness": 0.5,
    "context_awareness": 0.8,
    "adaptability": 0.7,
    "personability": 0.8
  },
  "fitness_score": 0.95,
  "metadata": {...},
  "generation": 42
}
```

### Evolutionary Parameters Used

- **Population Size**: 50 (optimized for 8-trait genomes)
- **Tournament Size**: 3
- **Crossover Rate**: 80%
- **Mutation Rate**: 10% per trait
- **Elitism Factor**: 20% retained unchanged
- **Generation Limit**: 100 (with early convergence detection)

## LLM Integration Architecture

### Multi-Provider Support Matrix

| Provider | Local Setup | Cloud Setup | API Tokens | Streaming | Cost Estimation |
|----------|-------------|--------------|------------|-----------|-----------------|
| OpenAI   | ❌          | ✅           | Required   | ✅        | ✅              |
| Ollama   | ✅          | ✅           | Optional   | ✅        | ❌              |
| Anthropic| ❌          | ✅           | Required   | ✅        | ✅              |

### Rate Limiting Implementation

- **OpenAI**: Token-based rate limiting (10k tokens/minute)
- **Ollama**: Request-based rate limiting (60 requests/minute)
- **Exponential Backoff**: Multiplier=2, Max Delay=30s, Jitter=±10%

## Benchmarking Protocols

### Performance Testing Procedure

1. **Warm-up Phase**: 10 iterations discarded
2. **Measurement Phase**: 1000 iterations sampled
3. **Stability Check**: Coefficient of variation < 5%
4. **Outlier Removal**: ±3σ statistical filtering

### Comparative Analysis Methodology

1. **Prompt Engineering Baseline**:
   - Manual crafting by experienced prompt engineers
   - 5 different prompts tested per personality
   - Human evaluation on trait consistency

2. **Consistency Metric Calculation**:
   - Intra-personality correlation: Pearson r > 0.85
   - Inter-evaluation agreement: Cohen's κ > 0.75
   - Longitudinal drift: Slope < 0.01 per generation

## Reproduction Instructions

### Environment Setup
```bash
# Clone repository
git clone https://github.com/jimmyjdejesus-cmyk/allele.git
cd allele

# Install dependencies
pip install -e .[dev]

# Download required models (for Ollama testing)
ollama pull llama2:7b
```

### Running Benchmarks
```bash
# Full test suite
pytest tests/ --benchmark-only

# Statistical analysis
jupyter notebook notebooks/statistical_analysis.ipynb

# Performance profiling
python -m cProfile src/allele/evolution.py
```

### Validation Checks
- **Genome Serialization**: Round-trip consistency 100%
- **Evolution Determinism**: Fixed seed produces identical results
- **API Key Independence**: Core functionality works without API keys

All experimental results can be reproduced using the provided seeds and identical hardware environment. Performance variations <10% are expected on different systems.
