# Results Aggregation and Analysis

## ADDED Requirements

### REQ-AGGREGATE-001: Results Parsing
The system MUST parse JSON results from matrix evaluation and extract scores for analysis.

#### Scenario: Parse matrix results JSON
Given a results JSON file from matrix evaluation
When parse_results() is called
Then it loads the JSON structure
And extracts scores for each model × personality × benchmark combination
And validates the structure is correct

#### Scenario: Handle malformed results
Given a malformed JSON file
When parse_results() is called
Then it logs an error
And returns None or empty structure
And does not raise an exception

### REQ-AGGREGATE-002: Statistical Analysis
The system MUST calculate statistical metrics for each model and personality combination.

#### Scenario: Calculate mean score per personality
Given results for model "qwen2.5:0.5b"
And personality "technical_expert"
And scores across 5 benchmarks: [0.45, 0.52, 0.48, 0.50, 0.47]
When statistics are calculated
Then mean = 0.484
And std dev is calculated
And min/max are identified

#### Scenario: Calculate improvement vs baseline
Given baseline average score: 0.40
And personality "technical_expert" average score: 0.48
When improvement is calculated
Then delta = +0.08 (20% improvement)
And percentage improvement = 20%

### REQ-AGGREGATE-003: Comparison Table Generation
The system MUST generate comparison tables showing Model × Personality × Benchmark performance.

#### Scenario: Generate full matrix table
Given results for 3 models, 7 personalities, 5 benchmarks
When comparison table is generated
Then it creates a Markdown table with:
- Rows: Model × Personality combinations
- Columns: Benchmark scores
- Additional columns: Average, vs Baseline
And table is properly formatted

#### Scenario: Generate summary statistics table
Given aggregated results
When summary table is generated
Then it shows:
- Best performing model per benchmark
- Best performing personality per benchmark
- Overall best combinations
And highlights top performers

### REQ-AGGREGATE-004: Best Configuration Identification
The system MUST identify the best performing configurations across different metrics.

#### Scenario: Find best overall configuration
Given all model × personality combinations
When best configuration is identified
Then it ranks by average score across all benchmarks
And returns top 3 configurations
And includes improvement metrics

#### Scenario: Find best per benchmark
Given all results
When best per benchmark is identified
Then it finds:
- Best model for MMLU
- Best personality for GSM8K
- Best combination for each benchmark
And includes scores and improvements

### REQ-AGGREGATE-005: Export Analysis Results
The system MUST export analysis results to Markdown format for documentation.

#### Scenario: Export to Markdown
Given aggregated results and statistics
When export_to_markdown() is called
Then it generates:
- Summary statistics section
- Full matrix table
- Best performers section
- Improvement analysis
And saves to Markdown file

#### Scenario: Markdown formatting validation
Given exported Markdown file
When validated
Then it has correct table syntax
And proper heading hierarchy
And valid Markdown structure

## MODIFIED Requirements

### REQ-AGGREGATE-006: Extend Existing Analysis Tools
The existing analysis tools (e.g., analyze_lm_eval_results.py) MUST be extended or new tools MUST be created to handle matrix results.

#### Scenario: Reuse existing analysis patterns
Given existing analyze_lm_eval_results.py patterns
When creating matrix analyzer
Then it follows similar structure
And reuses common utilities
And maintains consistency with existing tools

