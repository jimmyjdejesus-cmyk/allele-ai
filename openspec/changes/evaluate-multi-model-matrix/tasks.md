# Tasks: Evaluate Multi-Model Personality Matrix

## Task List

### Phase 1: Foundation (COT Support & Model Discovery)

- [x] 1. **Add COT prompt utility function**
   - File: `src/benchmark/utils.py`
   - Add `build_cot_prompt(base_prompt: str) -> str` function
   - Returns prompt wrapped with "Let's think step by step:"
   - Unit test: Verify COT wrapper is correctly applied
   - Validation: Test with sample prompts

- [x] 2. **Extend GenomeModel class with COT mode**
   - File: `scripts/run_personality_benchmark.py`
   - Add `cot_mode: bool` parameter to `GenomeModel.__init__()`
   - Modify `_build_system_prompt()` to handle COT mode
   - When `cot_mode=True`, apply COT wrapper without genome traits
   - Unit test: Verify COT mode generates correct prompts
   - Integration test: Run single benchmark with COT mode

- [x] 3. **Create model size detection utility**
   - File: `scripts/detect_model_sizes.py`
   - Implement `detect_ollama_models() -> List[str]` function
   - Query Ollama API or `ollama list` command
   - Parse model names for parameter counts (0.5b, 1b, 2b, 3b)
   - Filter to 0.5b-3b range
   - Handle edge cases (model name variations, missing metadata)
   - Return validated model list
   - Unit test: Test parsing logic with various model name formats
   - Integration test: Verify detection works with actual Ollama installation

### Phase 2: Matrix Evaluation Runner

- [x] 4. **Create matrix evaluation script structure**
   - File: `scripts/run_matrix_evaluation.py`
   - Implement `MatrixEvaluator` class
   - Add CLI argument parsing (--mode, --limit, --models, --resume)
   - Initialize output directory structure
   - Validation: Script runs without errors, help text displays

- [x] 5. **Implement configuration matrix generation**
   - File: `scripts/run_matrix_evaluation.py`
   - Implement `generate_matrix_config()` method
   - Combine models × personalities × benchmarks
   - Handle dependencies (baseline should run first for comparisons)
   - Generate execution plan with ordering
   - Unit test: Verify matrix generation produces correct combinations
   - Validation: Print matrix size and combinations for verification

- [x] 6. **Implement parallel benchmark execution**
   - File: `scripts/run_matrix_evaluation.py`
   - Integrate with existing `lm-eval` infrastructure
   - Support personality injection via system prompts
   - Support COT prompt wrapping
   - Implement parallel execution with configurable concurrency (default: 2)
   - Add progress tracking and logging
   - Handle errors gracefully (skip failed combinations, continue)
   - Integration test: Run with 1 model, 2 personalities, 1 benchmark

- [x] 7. **Add checkpointing and resume functionality**
   - File: `scripts/run_matrix_evaluation.py`
   - Save progress after each completed combination
   - Implement `--resume` flag to load checkpoint
   - Skip already-completed combinations
   - Validation: Start run, interrupt, resume, verify completion

### Phase 3: Results Analysis & Aggregation

- [x] 8. **Create results analyzer**
   - File: `scripts/analyze_matrix_results.py`
   - Implement `MatrixResultsAnalyzer` class
   - Parse JSON results from matrix evaluation
   - Calculate statistics: mean, std dev, min, max per model/personality
   - Calculate improvements vs baseline
   - Generate comparison tables (Model × Personality × Benchmark)
   - Export to Markdown format
   - Unit test: Test aggregation logic with sample data
   - Validation: Verify tables are correctly formatted

- [x] 9. **Identify best performing configurations**
   - File: `scripts/analyze_matrix_results.py`
   - Implement `find_best_configurations()` method
   - Rank by average score across benchmarks
   - Rank by improvement vs baseline
   - Generate summary report with top performers
   - Validation: Verify rankings match expected results

### Phase 4: Documentation Auto-Update

- [x] 10. **Create whitepaper updater script**
    - File: `scripts/update_whitepaper_benchmarks.py`
    - Parse matrix results JSON
    - Generate Section 4.2.5 content with tables
    - Insert/update content between `<!-- MATRIX_RESULTS_START -->` and `<!-- MATRIX_RESULTS_END -->` markers
    - Validate Markdown syntax
    - Backup original file before update
    - Unit test: Test marker insertion logic
    - Integration test: Update whitepaper with sample results, verify formatting

- [x] 11. **Add matrix results section to whitepaper**
    - File: `docs/whitepaper/phylogenic_whitepaper.md`
    - Add `<!-- MATRIX_RESULTS_START -->` / `<!-- MATRIX_RESULTS_END -->` markers in Section 4.2
    - Add placeholder Section 4.2.5 "Multi-Model Personality Evaluation"
    - Validation: Markers are correctly placed, section structure is valid

- [x] 12. **Create README matrix updater script**
    - File: `scripts/update_readme_matrix.py`
    - Parse matrix results JSON
    - Generate comprehensive matrix table
    - Insert/update content between `<!-- MATRIX_EVALUATION_START -->` and `<!-- MATRIX_EVALUATION_END -->` markers
    - Include summary statistics and best performers
    - Validate Markdown syntax
    - Unit test: Test table generation and marker insertion
    - Integration test: Update README with sample results

- [x] 13. **Add matrix evaluation section to README**
    - File: `README.md`
    - Add `<!-- MATRIX_EVALUATION_START -->` / `<!-- MATRIX_EVALUATION_END -->` markers
    - Add placeholder section for matrix results
    - Validation: Markers are correctly placed

### Phase 5: Integration & Testing

- [x] 14. **Create end-to-end integration test**
    - File: `tests/integration/test_matrix_evaluation.py`
    - Test full workflow: model discovery → matrix generation → execution → analysis → documentation update
    - Use 1-2 small models, 2-3 personalities, 1 benchmark with `--limit 5`
    - Verify results JSON is created correctly
    - Verify whitepaper and README are updated
    - Validation: All steps complete successfully

- [x] 15. **Add smoke test script**
    - File: `scripts/smoke_test_matrix.py` or extend existing
    - Quick validation with minimal configuration
    - 1 model, baseline + 1 personality, 1 benchmark, limit 10
    - Verify no errors, results are generated
    - Validation: Smoke test completes in <5 minutes

- [x] 16. **Update documentation with usage examples**
    - File: `docs/LM_EVAL_GUIDE.md` or new `docs/MATRIX_EVALUATION.md`
    - Document matrix evaluation workflow
    - Provide usage examples
    - Explain results interpretation
    - Validation: Documentation is clear and accurate

### Phase 6: Validation & Polish

- [x] 17. **Add error handling and logging improvements**
    - Review all scripts for comprehensive error handling
    - Add structured logging with progress indicators
    - Add validation for model availability before execution
    - Validation: Test error scenarios (model unavailable, Ollama down, etc.)

- [x] 18. **Performance optimization**
    - Profile matrix evaluation execution
    - Optimize parallel execution concurrency
    - Add caching for model discovery results
    - Validation: Execution time is acceptable

- [x] 19. **Final validation and testing**
    - Run full matrix evaluation with 2-3 models (if time permits)
    - Verify all results are correct
    - Verify documentation updates are accurate
    - Verify no regressions in existing functionality
    - Validation: All tests pass, documentation is updated correctly

## Dependencies

- Tasks 1-2: Can be done in parallel
- Task 3: Independent, can be done in parallel with 1-2
- Tasks 4-7: Sequential (each builds on previous)
- Task 8-9: Depends on Task 7 (needs results)
- Tasks 10-11: Can be done in parallel, depend on Task 8
- Tasks 12-13: Can be done in parallel, depend on Task 8
- Task 14: Depends on all previous tasks
- Tasks 15-19: Can be done in parallel after Task 14

## Acceptance Criteria

- ✅ Matrix evaluation runs successfully for all combinations
- ✅ COT prompts are correctly applied
- ✅ Model discovery filters to 0.5b-3b range
- ✅ Results are aggregated with statistics
- ✅ Whitepaper Section 4.2.5 is updated automatically
- ✅ README matrix table is updated automatically
- ✅ All unit and integration tests pass
- ✅ Documentation is complete and accurate

