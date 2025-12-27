## 🧬 Phylogenic Genome A/B Benchmark Results

**Model**: `tinyllama:latest` (1.1B params)  
**Date**: 2025-12-19

### Performance by Personality Archetype

| Personality | MMLU | GSM8K | Reasoning | Average | vs Baseline |
|-------------|------|-------|-----------|---------|-------------|
| **baseline** | 40.0% | 0.0% | 0.0% | 13.3% | - |
| **technical_expert** | 40.0% | 33.3% | 0.0% | 24.4% | **+11.1%** ✅ |
| **creative_thinker** | 30.0% | 33.3% | 0.0% | 21.1% | +7.8% ✅ |
| **concise_analyst** | 40.0% | 33.3% | 0.0% | 24.4% | **+11.1%** ✅ |
| **balanced** | 30.0% | 33.3% | 0.0% | 21.1% | +7.8% ✅ |
| **high_context** | 40.0% | 33.3% | 0.0% | 24.4% | **+11.1%** ✅ |

### Key Findings

- 📈 **All genome configurations outperform baseline** by 7.8-11.1%
- 🎯 **GSM8K (Math)**: Genome-enhanced models scored 33.3% vs 0% baseline
- 💡 **Best performers**: `technical_expert`, `concise_analyst`, `high_context`
- 🧪 High `technical_knowledge` + `conciseness` traits produce optimal results

### Genome Configuration (Best Performing)

| Trait | technical_expert | concise_analyst |
|-------|------------------|-----------------|
| empathy | 0.20 | 0.30 |
| technical_knowledge | **0.99** | 0.85 |
| creativity | 0.30 | 0.40 |
| conciseness | **0.95** | **0.99** |
| context_awareness | 0.90 | 0.80 |
| adaptability | 0.50 | 0.60 |
| engagement | 0.20 | 0.30 |
| personability | 0.20 | 0.30 |

### Run Your Own Benchmarks

```bash
# Quick A/B test
python scripts/run_ab_benchmark.py --model tinyllama:latest --samples 10

# Full personality comparison
python scripts/run_personality_benchmark.py --model tinyllama:latest --samples 20

# With larger model
python scripts/run_ab_benchmark.py --model llama2:latest --samples 50 --update-readme
```
