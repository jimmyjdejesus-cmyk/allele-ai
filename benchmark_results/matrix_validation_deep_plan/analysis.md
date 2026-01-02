# Multi-Model Personality Matrix Analysis

**Generated**: 2025-12-31 18:15:01

## Summary Statistics

- Total configurations evaluated: 12
- Models tested: 1
- Benchmarks: gsm8k, mmlu

## Top Performing Configurations

### Ranked by Average Score

| Rank | Model | Personality | Avg Score | vs Baseline |
|------|-------|------------|-----------|-------------|
| 1 | llama3.2:1b | technical_expert+cot | 0.95 | +0.33 |
| 2 | llama3.2:1b | concise_analyst+cot | 0.95 | +0.33 |
| 3 | llama3.2:1b | balanced+cot | 0.95 | +0.33 |
| 4 | llama3.2:1b | high_context+cot | 0.95 | +0.33 |
| 5 | llama3.2:1b | cot | 0.95 | +0.33 |
| 6 | llama3.2:1b | creative_thinker+cot | 0.90 | +0.28 |
| 7 | llama3.2:1b | technical_expert | 0.62 | +0.00 |
| 8 | llama3.2:1b | creative_thinker | 0.62 | +0.00 |
| 9 | llama3.2:1b | concise_analyst | 0.62 | +0.00 |
| 10 | llama3.2:1b | balanced | 0.62 | +0.00 |

## COT Prompting Impact

Improvement when adding COT to each personality:

| Model | Personality | Without COT | With COT | Improvement | % Change |
|-------|-------------|-------------|----------|-------------|----------|
| llama3.2:1b | balanced | 0.62 | 0.95 | +0.33 | +54.1% |
| llama3.2:1b | concise_analyst | 0.62 | 0.95 | +0.33 | +54.1% |
| llama3.2:1b | creative_thinker | 0.62 | 0.90 | +0.28 | +45.9% |
| llama3.2:1b | high_context | 0.62 | 0.95 | +0.33 | +54.1% |
| llama3.2:1b | technical_expert | 0.62 | 0.95 | +0.33 | +54.1% |


### Detailed Breakdown

#### 1. llama3.2:1b + technical_expert+cot

- **Average Score**: 0.95
- **Improvement vs Baseline**: +0.33
- **Benchmark Breakdown**:
  - mmlu: 0.90
  - gsm8k: 1.00

#### 2. llama3.2:1b + concise_analyst+cot

- **Average Score**: 0.95
- **Improvement vs Baseline**: +0.33
- **Benchmark Breakdown**:
  - mmlu: 0.90
  - gsm8k: 1.00

#### 3. llama3.2:1b + balanced+cot

- **Average Score**: 0.95
- **Improvement vs Baseline**: +0.33
- **Benchmark Breakdown**:
  - mmlu: 0.90
  - gsm8k: 1.00

#### 4. llama3.2:1b + high_context+cot

- **Average Score**: 0.95
- **Improvement vs Baseline**: +0.33
- **Benchmark Breakdown**:
  - mmlu: 0.90
  - gsm8k: 1.00

#### 5. llama3.2:1b + cot

- **Average Score**: 0.95
- **Improvement vs Baseline**: +0.33
- **Benchmark Breakdown**:
  - mmlu: 0.90
  - gsm8k: 1.00


## Matrix Evaluation Results

### Performance by Model × Personality × Benchmark

| Model | Personality | gsm8k | mmlu | Average | vs Baseline |
|---|---|---|---|---|---|
| llama3.2:1b | balanced | 0.33 | 0.90 | 0.62 | +0.00 |
| llama3.2:1b | balanced+cot | 1.00 | 0.90 | 0.95 | +0.33 |
| llama3.2:1b | baseline | 0.33 | 0.90 | 0.62 | - |
| llama3.2:1b | concise_analyst | 0.33 | 0.90 | 0.62 | +0.00 |
| llama3.2:1b | concise_analyst+cot | 1.00 | 0.90 | 0.95 | +0.33 |
| llama3.2:1b | cot | 1.00 | 0.90 | 0.95 | +0.33 |
| llama3.2:1b | creative_thinker | 0.33 | 0.90 | 0.62 | +0.00 |
| llama3.2:1b | creative_thinker+cot | 1.00 | 0.80 | 0.90 | +0.28 |
| llama3.2:1b | high_context | 0.33 | 0.90 | 0.62 | +0.00 |
| llama3.2:1b | high_context+cot | 1.00 | 0.90 | 0.95 | +0.33 |
| llama3.2:1b | technical_expert | 0.33 | 0.90 | 0.62 | +0.00 |
| llama3.2:1b | technical_expert+cot | 1.00 | 0.90 | 0.95 | +0.33 |

## Statistics by Model

### llama3.2:1b

| Personality | Mean | Std Dev | Min | Max | vs Baseline |
|-------------|------|---------|-----|-----|-------------|
| balanced | 0.62 | 0.40 | 0.33 | 0.90 | +0.00 |
| balanced+cot | 0.95 | 0.07 | 0.90 | 1.00 | +0.33 |
| baseline | 0.62 | 0.40 | 0.33 | 0.90 | - |
| concise_analyst | 0.62 | 0.40 | 0.33 | 0.90 | +0.00 |
| concise_analyst+cot | 0.95 | 0.07 | 0.90 | 1.00 | +0.33 |
| cot | 0.95 | 0.07 | 0.90 | 1.00 | +0.33 |
| creative_thinker | 0.62 | 0.40 | 0.33 | 0.90 | +0.00 |
| creative_thinker+cot | 0.90 | 0.14 | 0.80 | 1.00 | +0.28 |
| high_context | 0.62 | 0.40 | 0.33 | 0.90 | +0.00 |
| high_context+cot | 0.95 | 0.07 | 0.90 | 1.00 | +0.33 |
| technical_expert | 0.62 | 0.40 | 0.33 | 0.90 | +0.00 |
| technical_expert+cot | 0.95 | 0.07 | 0.90 | 1.00 | +0.33 |

