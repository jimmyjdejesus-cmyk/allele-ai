# Multi-Model Personality Matrix Evaluation Guide

This guide explains how to use the matrix evaluation system to test multiple small language models across different personality configurations and benchmarks.

## Overview

The matrix evaluation system tests all combinations of:
- **Models**: Auto-detected Ollama models (0.5B-3B parameter range)
- **Personalities**: Baseline, 5 base personalities, COT prompts
- **Benchmarks**: MMLU, HellaSwag, GSM8K, ARC-Easy, TruthfulQA

This provides comprehensive data on how different models respond to personality configurations and prompt strategies.

## Quick Start

### 1. Run Matrix Evaluation

```bash
# Quick test with limited samples
python scripts/run_matrix_evaluation.py --mode quick --limit 10

# Standard evaluation
python scripts/run_matrix_evaluation.py --mode standard --limit 50

# With specific models
python scripts/run_matrix_evaluation.py --models qwen2.5:0.5b gemma3:1b --limit 20
```

### 2. Analyze Results

```bash
python scripts/analyze_matrix_results.py \
    --input benchmark_results/matrix_evaluation/results.json \
    --output analysis.md
```

### 3. Update Documentation

```bash
# Update whitepaper
python scripts/update_whitepaper_benchmarks.py \
    --input benchmark_results/matrix_evaluation/results.json \
    --analyzer-output analysis.md

# Update README
python scripts/update_readme_matrix.py \
    --input benchmark_results/matrix_evaluation/results.json \
    --analyzer-output analysis.md
```

## Workflow

### Step 1: Model Discovery

The system automatically detects available Ollama models in the 0.5B-3B range:

```bash
# Manual model detection
python scripts/detect_model_sizes.py --min-size 0.5 --max-size 3.0
```

You can also manually specify models:
```bash
python scripts/run_matrix_evaluation.py \
    --models qwen2.5:0.5b gemma3:1b llama3.2:1b \
    --mode standard
```

### Step 2: Matrix Generation

The system generates all combinations:
- Models × Personalities × Benchmarks

Example: 2 models × 7 personalities × 5 benchmarks = 70 configurations

### Step 3: Execution

Configurations are executed in parallel batches (default: 2 concurrent):

```bash
# Custom concurrency
python scripts/run_matrix_evaluation.py \
    --mode standard \
    --concurrency 4 \
    --limit 50
```

**Checkpointing**: Progress is saved after each batch. Resume interrupted runs:
```bash
python scripts/run_matrix_evaluation.py --resume
```

### Step 4: Analysis

Analyze results to generate statistics and rankings:

```bash
python scripts/analyze_matrix_results.py \
    --input benchmark_results/matrix_evaluation/results.json \
    --output analysis.md \
    --top-n 10
```

The analyzer generates:
- Summary statistics
- Top performing configurations
- Comparison tables (Model × Personality × Benchmark)
- Statistics by model

### Step 5: Documentation Update

Automatically update whitepaper and README:

```bash
# Update both
python scripts/update_whitepaper_benchmarks.py \
    --analyzer-output analysis.md

python scripts/update_readme_matrix.py \
    --analyzer-output analysis.md
```

## Configuration Options

### Matrix Evaluation

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Benchmark mode (quick/standard/comprehensive) | standard |
| `--models` | Specific models to test | auto-detect |
| `--personalities` | Specific personalities to test | all |
| `--benchmarks` | Specific benchmarks to test | standard set |
| `--limit` | Limit samples per benchmark | None |
| `--min-size` | Minimum model size (billions) | 0.5 |
| `--max-size` | Maximum model size (billions) | 3.0 |
| `--concurrency` | Parallel executions | 2 |
| `--resume` | Resume from checkpoint | False |
| `--output-dir` | Output directory | benchmark_results/matrix_evaluation |

### Results Analysis

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Results JSON file | Required |
| `--output` | Output Markdown file | stdout |
| `--top-n` | Number of top configs to show | 10 |

## Personality Configurations

The system tests the following personality configurations:

1. **baseline**: No genome, raw model performance
2. **technical_expert**: High technical knowledge, concise
3. **creative_thinker**: High creativity, engaging
4. **concise_analyst**: High conciseness, technical
5. **balanced**: Moderate trait values
6. **high_context**: High context awareness
7. **cot**: Chain of Thought prompting (no genome traits)

## Benchmarks

### Standard Set (default)
- **MMLU**: Massive Multitask Language Understanding (knowledge)
- **HellaSwag**: Commonsense reasoning
- **GSM8K**: Grade school math word problems
- **ARC-Easy**: AI2 Reasoning Challenge (easy)
- **TruthfulQA**: Truthfulness evaluation

### Quick Mode
- MMLU, HellaSwag, GSM8K (limited samples)

### Comprehensive Mode
- All standard + ARC-Challenge, Winogrande, PIQA, BoolQ, SiQA

## Results Interpretation

### Score Metrics

- **MMLU**: Knowledge breadth (0-100%). >40% is good for 1B models
- **HellaSwag**: Commonsense (0-100%). >60% is decent for small models
- **GSM8K**: Math reasoning (0-100%). Often low (<10%) for 1B models
- **ARC-Easy**: Science reasoning (0-100%)
- **TruthfulQA**: Truthfulness (0-100%). Higher is better

### Comparison Metrics

- **Average Score**: Mean across all benchmarks
- **vs Baseline**: Improvement over baseline model
- **Std Dev**: Consistency across benchmarks

### Best Configuration Selection

Configurations are ranked by:
1. **Average Score**: Overall performance across benchmarks
2. **Improvement vs Baseline**: Relative improvement

## Example Output

### Analysis Report Structure

```
# Multi-Model Personality Matrix Analysis

## Summary Statistics
- Total configurations evaluated: 70
- Models tested: 2
- Benchmarks: mmlu, hellaswag, gsm8k, arc_easy, truthfulqa_mc2

## Top Performing Configurations

| Rank | Model | Personality | Avg Score | vs Baseline |
|------|-------|------------|-----------|-------------|
| 1 | qwen2.5:0.5b | cot | 48.5 | +3.2 |
| 2 | qwen2.5:0.5b | technical_expert | 47.8 | +2.5 |
...

## Matrix Evaluation Results

### Performance by Model × Personality × Benchmark
[Full comparison table]
```

## Smoke Testing

Quick validation with minimal configuration:

```bash
python scripts/smoke_test_matrix.py
```

Smoke test configuration:
- 1 model (auto-detected)
- Baseline + 1 personality
- 1 benchmark (MMLU)
- Limit: 10 samples
- Target: <5 minutes

## Troubleshooting

### "No models detected"
- Ensure Ollama is running: `ollama serve`
- Pull models: `ollama pull qwen2.5:0.5b`
- Manually specify: `--models qwen2.5:0.5b`

### "lm_eval module not found"
- Install: `pip install "lm-eval[all]>=0.4.0"`

### "Checkpoint not found"
- Check `benchmark_results/matrix_evaluation/checkpoint.json`
- Remove `--resume` flag to start fresh

### "Results file not found"
- Verify evaluation completed successfully
- Check `benchmark_results/matrix_evaluation/results.json`

## Performance Considerations

### Execution Time

- **Quick mode** (1 model, 7 personalities, 3 benchmarks, limit 10): ~10-15 minutes
- **Standard mode** (2 models, 7 personalities, 5 benchmarks, limit 50): ~2-4 hours
- **Comprehensive mode**: ~8-12 hours

### Resource Usage

- **GPU Memory**: Varies by model size (0.5B-3B range)
- **CPU**: Parallel execution uses multiple cores
- **Disk**: Results JSON ~1-5MB per configuration

### Optimization Tips

1. Use `--limit` for quick testing
2. Adjust `--concurrency` based on GPU memory
3. Use `--resume` for long-running evaluations
4. Start with `--mode quick` to validate setup

## Integration with Existing Systems

The matrix evaluation system integrates with:

- **lm-eval**: Uses existing `BenchmarkRunner` infrastructure
- **Personality System**: Uses `PERSONALITY_ARCHETYPES` from `run_personality_benchmark.py`
- **Model Discovery**: Uses `detect_model_sizes.py` from Phase 1
- **Documentation**: Updates whitepaper and README automatically

## Key Findings from Full Matrix Evaluation

### COT Prompting: Major Performance Uplift

Chain of Thought prompting demonstrated significant improvements, especially on smaller models:

- **llama3.2:1b (1B)**: +9.7% to +16.6% improvement (average +13%)
  - Standalone COT: +13.8% vs baseline
  - Best: balanced+cot (+16.6%)
- **gemma2:2b (2B)**: +3.4% to +5.8% improvement (average +4.5%)
  - Standalone COT: +4.2% vs baseline
- **qwen2.5:0.5b (0.5B)**: Mixed results (-10.6% to +1.8%)
  - Technical expert with COT: -10.6% (negative impact)
  - Suggests COT may be counterproductive on very small models

### Personality Traits: Minimal to Negative Impact

Across all models, personality traits alone provided **no measurable benefit** on objective reasoning tasks:

- **gemma2:2b**: All personalities = 0.95 (identical to baseline, +0.00)
- **llama3.2:1b**: Most personalities = 0.80 (identical to baseline, +0.00)
  - creative_thinker and technical_expert: -0.01 (slight negative)
- **qwen2.5:0.5b**: Mixed results (-0.03 to +0.01)
  - Only technical_expert showed minimal positive (+0.01)

### Model Size Affects COT Effectiveness

- **1B models**: Strongest COT benefit (+16.6% max)
- **2B models**: Moderate COT benefit (+5.8% max)
- **0.5B models**: Negative or neutral COT impact

This suggests diminishing returns as model size increases, and potential harm on very small models.

### Recommendations

**For Objective Reasoning Tasks**:
- ✅ Use COT prompting, especially on 1B-2B models
- ❌ Skip personality traits (no benefit, may slightly harm)
- ⚠️ Test COT carefully on 0.5B models (can reduce performance)

**For Conversational Tasks**:
- Personality traits may be more effective (requires separate evaluation)
- COT may still help, but benefit may differ

## Next Steps

After running matrix evaluation:

1. Review analysis report for insights
2. Identify best model-personality combinations
3. Compare COT vs personality vs baseline
4. Update documentation with results
5. Use findings to optimize agent configurations

## Related Documentation

- [LM-Eval Guide](LM_EVAL_GUIDE.md) - Standard benchmarking
- [Personality Benchmarking](README.md#personality-ab-benchmark-results) - Single-model personality testing
- [Whitepaper](whitepaper/phylogenic_whitepaper.md#425-multi-model-personality-evaluation) - Academic results

