# Multi-Model Personality Matrix Analysis

**Generated**: 2026-01-01 00:59:04

## Summary Statistics

- Total configurations evaluated: 36
- Models tested: 3
- Benchmarks: arc_easy, gsm8k, hellaswag, mmlu, truthfulqa_mc2

## Top Performing Configurations

### Ranked by Average Score

| Rank | Model | Personality | Avg Score | vs Baseline |
|------|-------|------------|-----------|-------------|
| 1 | gemma2:2b | creative_thinker+cot | 1.00 | +0.05 |
| 2 | gemma2:2b | concise_analyst+cot | 1.00 | +0.05 |
| 3 | gemma2:2b | balanced+cot | 1.00 | +0.05 |
| 4 | gemma2:2b | cot | 0.99 | +0.04 |
| 5 | gemma2:2b | technical_expert+cot | 0.98 | +0.03 |
| 6 | gemma2:2b | high_context+cot | 0.98 | +0.03 |
| 7 | gemma2:2b | technical_expert | 0.95 | +0.00 |
| 8 | gemma2:2b | creative_thinker | 0.95 | +0.00 |
| 9 | gemma2:2b | concise_analyst | 0.95 | +0.00 |
| 10 | gemma2:2b | balanced | 0.95 | +0.00 |

## COT Prompting Impact

Improvement when adding COT to each personality:

| Model | Personality | Without COT | With COT | Improvement | % Change |
|-------|-------------|-------------|----------|-------------|----------|
| gemma2:2b | balanced | 0.95 | 1.00 | +0.05 | +5.8% |
| gemma2:2b | concise_analyst | 0.95 | 1.00 | +0.05 | +5.8% |
| gemma2:2b | creative_thinker | 0.95 | 1.00 | +0.05 | +5.8% |
| gemma2:2b | high_context | 0.95 | 0.98 | +0.03 | +3.4% |
| gemma2:2b | technical_expert | 0.95 | 0.98 | +0.03 | +3.4% |
| llama3.2:1b | balanced | 0.80 | 0.94 | +0.13 | +16.6% |
| llama3.2:1b | concise_analyst | 0.80 | 0.92 | +0.11 | +14.1% |
| llama3.2:1b | creative_thinker | 0.79 | 0.87 | +0.08 | +9.7% |
| llama3.2:1b | high_context | 0.81 | 0.93 | +0.11 | +13.9% |
| llama3.2:1b | technical_expert | 0.79 | 0.88 | +0.09 | +10.9% |
| qwen2.5:0.5b | balanced | 0.84 | 0.80 | -0.05 | -5.8% |
| qwen2.5:0.5b | concise_analyst | 0.85 | 0.86 | +0.02 | +1.8% |
| qwen2.5:0.5b | creative_thinker | 0.82 | 0.82 | +0.01 | +0.7% |
| qwen2.5:0.5b | high_context | 0.85 | 0.84 | -0.01 | -1.0% |
| qwen2.5:0.5b | technical_expert | 0.86 | 0.76 | -0.09 | -10.6% |


### Detailed Breakdown

#### 1. gemma2:2b + creative_thinker+cot

- **Average Score**: 1.00
- **Improvement vs Baseline**: +0.05
- **Benchmark Breakdown**:
  - mmlu: 1.00
  - hellaswag: 1.00
  - gsm8k: 1.00
  - arc_easy: 1.00
  - truthfulqa_mc2: 1.00

#### 2. gemma2:2b + concise_analyst+cot

- **Average Score**: 1.00
- **Improvement vs Baseline**: +0.05
- **Benchmark Breakdown**:
  - mmlu: 1.00
  - hellaswag: 1.00
  - gsm8k: 1.00
  - arc_easy: 1.00
  - truthfulqa_mc2: 1.00

#### 3. gemma2:2b + balanced+cot

- **Average Score**: 1.00
- **Improvement vs Baseline**: +0.05
- **Benchmark Breakdown**:
  - mmlu: 1.00
  - hellaswag: 1.00
  - gsm8k: 1.00
  - arc_easy: 1.00
  - truthfulqa_mc2: 1.00

#### 4. gemma2:2b + cot

- **Average Score**: 0.99
- **Improvement vs Baseline**: +0.04
- **Benchmark Breakdown**:
  - mmlu: 1.00
  - hellaswag: 1.00
  - gsm8k: 1.00
  - arc_easy: 1.00
  - truthfulqa_mc2: 0.95

#### 5. gemma2:2b + technical_expert+cot

- **Average Score**: 0.98
- **Improvement vs Baseline**: +0.03
- **Benchmark Breakdown**:
  - mmlu: 1.00
  - hellaswag: 1.00
  - gsm8k: 0.89
  - arc_easy: 1.00
  - truthfulqa_mc2: 1.00


## Matrix Evaluation Results

### Performance by Model × Personality × Benchmark

| Model | Personality | arc_easy | gsm8k | hellaswag | mmlu | truthfulqa_mc2 | Average | vs Baseline |
|---|---|---|---|---|---|---|---|---|
| gemma2:2b | balanced | 1.00 | 0.78 | 0.95 | 1.00 | 1.00 | 0.95 | +0.00 |
| gemma2:2b | balanced+cot | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.05 |
| gemma2:2b | baseline | 1.00 | 0.78 | 0.95 | 1.00 | 1.00 | 0.95 | - |
| gemma2:2b | concise_analyst | 1.00 | 0.78 | 0.95 | 1.00 | 1.00 | 0.95 | +0.00 |
| gemma2:2b | concise_analyst+cot | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.05 |
| gemma2:2b | cot | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 0.99 | +0.04 |
| gemma2:2b | creative_thinker | 1.00 | 0.78 | 0.95 | 1.00 | 1.00 | 0.95 | +0.00 |
| gemma2:2b | creative_thinker+cot | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.05 |
| gemma2:2b | high_context | 1.00 | 0.78 | 0.95 | 1.00 | 1.00 | 0.95 | +0.00 |
| gemma2:2b | high_context+cot | 1.00 | 0.89 | 1.00 | 1.00 | 1.00 | 0.98 | +0.03 |
| gemma2:2b | technical_expert | 1.00 | 0.78 | 0.95 | 1.00 | 1.00 | 0.95 | +0.00 |
| gemma2:2b | technical_expert+cot | 1.00 | 0.89 | 1.00 | 1.00 | 1.00 | 0.98 | +0.03 |
| llama3.2:1b | balanced | 0.95 | 0.33 | 0.90 | 0.93 | 0.90 | 0.80 | +0.00 |
| llama3.2:1b | balanced+cot | 0.95 | 1.00 | 1.00 | 0.93 | 0.80 | 0.94 | +0.13 |
| llama3.2:1b | baseline | 0.95 | 0.33 | 0.90 | 0.93 | 0.90 | 0.80 | - |
| llama3.2:1b | concise_analyst | 1.00 | 0.33 | 0.85 | 0.93 | 0.90 | 0.80 | +0.00 |
| llama3.2:1b | concise_analyst+cot | 0.80 | 1.00 | 1.00 | 0.93 | 0.85 | 0.92 | +0.11 |
| llama3.2:1b | cot | 0.85 | 1.00 | 1.00 | 0.87 | 0.85 | 0.91 | +0.11 |
| llama3.2:1b | creative_thinker | 0.95 | 0.33 | 0.85 | 0.93 | 0.90 | 0.79 | -0.01 |
| llama3.2:1b | creative_thinker+cot | 0.85 | 1.00 | 1.00 | 0.80 | 0.70 | 0.87 | +0.07 |
| llama3.2:1b | high_context | 0.95 | 0.33 | 0.95 | 0.93 | 0.90 | 0.81 | +0.01 |
| llama3.2:1b | high_context+cot | 0.80 | 1.00 | 1.00 | 0.93 | 0.90 | 0.93 | +0.12 |
| llama3.2:1b | technical_expert | 1.00 | 0.33 | 0.80 | 0.93 | 0.90 | 0.79 | -0.01 |
| llama3.2:1b | technical_expert+cot | 0.80 | 1.00 | 1.00 | 0.80 | 0.80 | 0.88 | +0.08 |
| qwen2.5:0.5b | balanced | 0.80 | 0.89 | 1.00 | 0.73 | 0.80 | 0.84 | -0.00 |
| qwen2.5:0.5b | balanced+cot | 0.65 | 0.78 | 1.00 | 0.80 | 0.75 | 0.80 | -0.05 |
| qwen2.5:0.5b | baseline | 0.85 | 0.78 | 1.00 | 0.87 | 0.75 | 0.85 | - |
| qwen2.5:0.5b | concise_analyst | 0.85 | 0.78 | 1.00 | 0.87 | 0.75 | 0.85 | +0.00 |
| qwen2.5:0.5b | concise_analyst+cot | 0.85 | 0.89 | 1.00 | 0.93 | 0.65 | 0.86 | +0.02 |
| qwen2.5:0.5b | cot | 0.65 | 1.00 | 1.00 | 0.80 | 0.75 | 0.84 | -0.01 |
| qwen2.5:0.5b | creative_thinker | 0.75 | 0.78 | 1.00 | 0.87 | 0.70 | 0.82 | -0.03 |
| qwen2.5:0.5b | creative_thinker+cot | 0.60 | 0.89 | 1.00 | 0.93 | 0.70 | 0.82 | -0.02 |
| qwen2.5:0.5b | high_context | 0.80 | 0.78 | 1.00 | 0.87 | 0.80 | 0.85 | +0.00 |
| qwen2.5:0.5b | high_context+cot | 0.70 | 1.00 | 1.00 | 0.80 | 0.70 | 0.84 | -0.01 |
| qwen2.5:0.5b | technical_expert | 0.80 | 0.78 | 1.00 | 0.80 | 0.90 | 0.86 | +0.01 |
| qwen2.5:0.5b | technical_expert+cot | 0.55 | 0.89 | 1.00 | 0.73 | 0.65 | 0.76 | -0.08 |

## Statistics by Model

### gemma2:2b

| Personality | Mean | Std Dev | Min | Max | vs Baseline |
|-------------|------|---------|-----|-----|-------------|
| balanced | 0.95 | 0.10 | 0.78 | 1.00 | +0.00 |
| balanced+cot | 1.00 | 0.00 | 1.00 | 1.00 | +0.05 |
| baseline | 0.95 | 0.10 | 0.78 | 1.00 | - |
| concise_analyst | 0.95 | 0.10 | 0.78 | 1.00 | +0.00 |
| concise_analyst+cot | 1.00 | 0.00 | 1.00 | 1.00 | +0.05 |
| cot | 0.99 | 0.02 | 0.95 | 1.00 | +0.04 |
| creative_thinker | 0.95 | 0.10 | 0.78 | 1.00 | +0.00 |
| creative_thinker+cot | 1.00 | 0.00 | 1.00 | 1.00 | +0.05 |
| high_context | 0.95 | 0.10 | 0.78 | 1.00 | +0.00 |
| high_context+cot | 0.98 | 0.05 | 0.89 | 1.00 | +0.03 |
| technical_expert | 0.95 | 0.10 | 0.78 | 1.00 | +0.00 |
| technical_expert+cot | 0.98 | 0.05 | 0.89 | 1.00 | +0.03 |

### llama3.2:1b

| Personality | Mean | Std Dev | Min | Max | vs Baseline |
|-------------|------|---------|-----|-----|-------------|
| balanced | 0.80 | 0.26 | 0.33 | 0.95 | +0.00 |
| balanced+cot | 0.94 | 0.08 | 0.80 | 1.00 | +0.13 |
| baseline | 0.80 | 0.26 | 0.33 | 0.95 | - |
| concise_analyst | 0.80 | 0.27 | 0.33 | 1.00 | +0.00 |
| concise_analyst+cot | 0.92 | 0.09 | 0.80 | 1.00 | +0.11 |
| cot | 0.91 | 0.08 | 0.85 | 1.00 | +0.11 |
| creative_thinker | 0.79 | 0.26 | 0.33 | 0.95 | -0.01 |
| creative_thinker+cot | 0.87 | 0.13 | 0.70 | 1.00 | +0.07 |
| high_context | 0.81 | 0.27 | 0.33 | 0.95 | +0.01 |
| high_context+cot | 0.93 | 0.08 | 0.80 | 1.00 | +0.12 |
| technical_expert | 0.79 | 0.27 | 0.33 | 1.00 | -0.01 |
| technical_expert+cot | 0.88 | 0.11 | 0.80 | 1.00 | +0.08 |

### qwen2.5:0.5b

| Personality | Mean | Std Dev | Min | Max | vs Baseline |
|-------------|------|---------|-----|-----|-------------|
| balanced | 0.84 | 0.10 | 0.73 | 1.00 | -0.00 |
| balanced+cot | 0.80 | 0.13 | 0.65 | 1.00 | -0.05 |
| baseline | 0.85 | 0.10 | 0.75 | 1.00 | - |
| concise_analyst | 0.85 | 0.10 | 0.75 | 1.00 | +0.00 |
| concise_analyst+cot | 0.86 | 0.13 | 0.65 | 1.00 | +0.02 |
| cot | 0.84 | 0.16 | 0.65 | 1.00 | -0.01 |
| creative_thinker | 0.82 | 0.12 | 0.70 | 1.00 | -0.03 |
| creative_thinker+cot | 0.82 | 0.17 | 0.60 | 1.00 | -0.02 |
| high_context | 0.85 | 0.09 | 0.78 | 1.00 | +0.00 |
| high_context+cot | 0.84 | 0.15 | 0.70 | 1.00 | -0.01 |
| technical_expert | 0.86 | 0.09 | 0.78 | 1.00 | +0.01 |
| technical_expert+cot | 0.76 | 0.18 | 0.55 | 1.00 | -0.08 |

