# Matrix Evaluation System

## ADDED Requirements

### REQ-MATRIX-001: Matrix Configuration Generation
The system MUST generate a configuration matrix that includes all combinations of:
- Models: Auto-detected Ollama models in 0.5b-3b parameter range
- Personalities: Baseline, 5 base personality archetypes (technical_expert, creative_thinker, concise_analyst, balanced, high_context), and COT prompt mode
- Benchmarks: MMLU, HellaSwag, GSM8K, ARC-Easy, TruthfulQA

#### Scenario: Generate full matrix
Given available models ["qwen2.5:0.5b", "gemma3:1b", "llama3.2:1b"]
And personalities ["baseline", "technical_expert", "creative_thinker", "concise_analyst", "balanced", "high_context", "cot"]
And benchmarks ["mmlu", "hellaswag", "gsm8k", "arc_easy", "truthfulqa_mc2"]
When the matrix is generated
Then it produces 3 × 7 × 5 = 105 unique combinations
And each combination includes model name, personality configuration, and benchmark task

#### Scenario: Handle empty model list
Given no models detected in 0.5b-3b range
When the matrix is generated
Then it returns an empty configuration
And logs a warning message
And exits gracefully

### REQ-MATRIX-002: Parallel Execution Engine
The system MUST execute benchmark combinations in parallel with configurable concurrency.

#### Scenario: Execute with default concurrency
Given a matrix with 105 combinations
And default concurrency limit of 2
When execution starts
Then it runs 2 combinations concurrently
And maintains 2 active executions until completion
And tracks progress for all combinations

#### Scenario: Execute with custom concurrency
Given a matrix with 105 combinations
And concurrency limit set to 4
When execution starts
Then it runs 4 combinations concurrently
And maintains 4 active executions until completion

#### Scenario: Handle execution failures
Given a combination that fails during execution
When the failure occurs
Then it logs the error
And marks the combination as failed
And continues with remaining combinations
And includes failure information in results

### REQ-MATRIX-003: Checkpointing and Resume
The system MUST support checkpointing progress and resuming from checkpoints.

#### Scenario: Save checkpoint after completion
Given an active matrix evaluation
When a combination completes successfully
Then it saves a checkpoint file with:
- Completed combinations
- Remaining combinations
- Current results
- Timestamp

#### Scenario: Resume from checkpoint
Given a checkpoint file from a previous run
When execution is started with --resume flag
Then it loads the checkpoint
And skips already-completed combinations
And continues with remaining combinations
And merges new results with existing results

### REQ-MATRIX-004: Results Storage
The system MUST store evaluation results in structured JSON format.

#### Scenario: Save results after completion
Given a completed matrix evaluation
When results are saved
Then it creates a JSON file with:
- Metadata (timestamp, models, personalities, benchmarks)
- Results (model × personality × benchmark scores)
- Statistics (mean, std dev, vs baseline)
- Execution times

#### Scenario: Results file structure
Given evaluation results
When saved to JSON
Then the structure matches:
```json
{
  "metadata": {...},
  "results": {
    "model_name": {
      "personality_name": {
        "benchmark_name": {
          "score": float,
          "raw_score": int,
          "total": int,
          "execution_time": float
        }
      }
    }
  }
}
```

### REQ-MATRIX-005: Progress Reporting
The system MUST provide progress reporting during execution.

#### Scenario: Display progress during execution
Given an active matrix evaluation
When combinations are executing
Then it displays:
- Current combination being executed
- Completed count / total count
- Estimated time remaining
- Success/failure counts

#### Scenario: Final summary report
Given a completed matrix evaluation
When execution finishes
Then it displays:
- Total combinations executed
- Success count
- Failure count
- Total execution time
- Results file location

## MODIFIED Requirements

### REQ-MATRIX-006: Integration with LM-Eval
The system MUST integrate with existing lm-eval infrastructure while supporting personality and COT prompt injection.

#### Scenario: Execute benchmark with personality
Given a model "qwen2.5:0.5b"
And personality "technical_expert"
And benchmark "mmlu"
When the combination is executed
Then it calls lm-eval with:
- Model configuration
- System prompt built from personality traits
- Benchmark task
And captures the results

#### Scenario: Execute benchmark with COT
Given a model "qwen2.5:0.5b"
And personality "cot"
And benchmark "gsm8k"
When the combination is executed
Then it calls lm-eval with:
- Model configuration
- COT prompt wrapper
- Benchmark task
And captures the results

