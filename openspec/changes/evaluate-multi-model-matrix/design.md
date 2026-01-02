# Design: Multi-Model Personality Matrix Evaluation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              Matrix Evaluation Orchestrator                  │
│         (scripts/run_matrix_evaluation.py)                   │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Model      │  │ Personality  │  │  Benchmark   │      │
│  │  Discovery   │  │  Generator   │  │  Executor    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                 │                │
│         └─────────────────┴─────────────────┘              │
│                           │                                  │
│                  ┌────────▼─────────┐                       │
│                  │  Configuration   │                       │
│                  │     Matrix       │                       │
│                  └────────┬─────────┘                       │
│                           │                                  │
│                  ┌────────▼─────────┐                       │
│                  │  Parallel        │                       │
│                  │  Execution       │                       │
│                  │  Engine          │                       │
│                  └────────┬─────────┘                       │
│                           │                                  │
│                  ┌────────▼─────────┐                       │
│                  │  Results          │                       │
│                  │  Aggregator       │                       │
│                  └────────┬─────────┘                       │
│                           │                                  │
│         ┌─────────────────┴─────────────────┐              │
│         │                                     │              │
│  ┌──────▼──────┐                    ┌────────▼────────┐     │
│  │ Whitepaper  │                    │  README         │     │
│  │  Updater    │                    │  Updater        │     │
│  └─────────────┘                    └─────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Matrix Configuration Generation

**Decision**: Generate all combinations upfront, then execute in parallel batches.

**Rationale**: 
- Allows progress tracking and checkpointing
- Enables dependency management (e.g., baseline must run before personality comparisons)
- Simplifies error handling and retry logic

**Implementation**:
```python
MatrixConfig = {
    models: List[str],  # Auto-detected
    personalities: List[str],  # baseline + 5 archetypes + cot
    benchmarks: List[str]  # 5 standard benchmarks
}
# Total combinations: |models| × |personalities| × |benchmarks|
```

### 2. COT Prompt Integration

**Decision**: Add COT as a special "personality" mode that wraps prompts with "Let's think step by step:"

**Rationale**:
- Treats COT as a prompt strategy, not a personality trait
- Allows direct comparison: baseline vs personality vs COT
- Minimal changes to existing infrastructure

**Implementation**:
- Extend `build_system_prompt()` in `src/benchmark/utils.py` with `build_cot_prompt()`
- Add `cot` mode to `GenomeModel` that applies COT wrapper without genome traits

### 3. Model Discovery

**Decision**: Query Ollama API, parse model names for parameter counts, filter to 0.5b-3b range.

**Rationale**:
- Automated discovery reduces manual configuration
- Parameter range filtering ensures focus on small models suitable for local GPU
- Fallback to manual list if detection fails

**Implementation**:
- Use `ollama list` command or API endpoint
- Parse model names for patterns like "1b", "0.5b", "2b", "3b"
- Handle edge cases (e.g., "gemma3:1b" → 1B parameters)

### 4. Results Storage

**Decision**: JSON format for structured data, Markdown for human-readable tables.

**Rationale**:
- JSON enables programmatic analysis and aggregation
- Markdown provides immediate visibility in documentation
- Both formats support version control and diff tracking

**Structure**:
```json
{
  "metadata": {
    "timestamp": "2025-12-22T...",
    "models": [...],
    "personalities": [...],
    "benchmarks": [...]
  },
  "results": {
    "model_name": {
      "personality_name": {
        "benchmark_name": {
          "score": 0.45,
          "raw_score": 45,
          "total": 100,
          "execution_time": 120.5
        }
      }
    }
  },
  "statistics": {
    "model_name": {
      "personality_name": {
        "mean_score": 0.48,
        "std_dev": 0.03,
        "vs_baseline": 0.05
      }
    }
  }
}
```

### 5. Documentation Auto-Update

**Decision**: Marker-based updates with validation.

**Rationale**:
- Preserves manual content outside markers
- Enables idempotent updates (safe to re-run)
- Clear boundaries for automated vs manual content

**Markers**:
- Whitepaper: `<!-- MATRIX_RESULTS_START -->` / `<!-- MATRIX_RESULTS_END -->`
- README: `<!-- MATRIX_EVALUATION_START -->` / `<!-- MATRIX_EVALUATION_END -->`

## Trade-offs

### Parallel Execution vs Sequential

**Chosen**: Parallel with configurable concurrency limit.

**Trade-off**: 
- ✅ Faster execution
- ❌ Higher resource usage (GPU memory, CPU)
- **Mitigation**: Default to 2 concurrent evaluations, allow override

### Full Matrix vs Sampling

**Chosen**: Full matrix with optional `--limit` for quick testing.

**Trade-off**:
- ✅ Complete data for analysis
- ❌ Longer execution time
- **Mitigation**: Checkpoint/resume, progress reporting, `--limit` flag

### Auto-Detection vs Manual List

**Chosen**: Auto-detection with manual override.

**Trade-off**:
- ✅ Convenience and automation
- ❌ Potential false positives/negatives in model size detection
- **Mitigation**: Clear logging, manual override flag, validation warnings

## Integration Points

### Extends Existing Systems

1. **Personality Benchmark System** (`run_personality_benchmark.py`)
   - Adds COT mode to `GenomeModel`
   - Reuses personality archetypes

2. **LM-Eval Integration** (`run_lm_eval_mass.py`)
   - Reuses benchmark execution infrastructure
   - Extends with personality/COT prompt injection

3. **Benchmark Utils** (`src/benchmark/utils.py`)
   - Adds `build_cot_prompt()` function
   - Extends `build_system_prompt()` if needed

### New Components

1. **Matrix Runner** (`scripts/run_matrix_evaluation.py`)
   - Orchestrates full evaluation matrix
   - Manages parallel execution
   - Handles checkpointing/resume

2. **Model Discovery** (`scripts/detect_model_sizes.py`)
   - Queries Ollama for available models
   - Filters by parameter range
   - Returns validated model list

3. **Results Analyzer** (`scripts/analyze_matrix_results.py`)
   - Aggregates results across dimensions
   - Calculates statistics
   - Generates comparison tables

4. **Documentation Updaters**
   - `scripts/update_whitepaper_benchmarks.py`
   - `scripts/update_readme_matrix.py`

## Performance Considerations

- **Execution Time**: Full matrix (e.g., 5 models × 7 personalities × 5 benchmarks = 175 combinations) may take 4-8 hours
- **Memory**: Parallel execution limited by GPU memory (RX6800)
- **Storage**: Results JSON files ~1-5MB per run
- **Optimization**: Checkpointing, resume capability, progress reporting

## Error Handling

- **Model unavailable**: Skip with warning, continue with other models
- **Benchmark failure**: Log error, mark as failed, continue
- **Ollama connection loss**: Retry with exponential backoff, fail after 3 attempts
- **Documentation update failure**: Log error, preserve existing content, continue

## Testing Strategy

1. **Unit Tests**: Model discovery, COT prompt building, results aggregation
2. **Integration Tests**: End-to-end with 1-2 models, 2-3 personalities, 1 benchmark
3. **Smoke Tests**: Quick validation with `--limit 10` before full run
4. **Validation**: Whitepaper/README syntax validation after updates

