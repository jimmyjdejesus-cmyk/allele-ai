# Proposal: Evaluate Multi-Model Personality Matrix

## Change ID
`evaluate-multi-model-matrix`

## Summary

Create a comprehensive matrix evaluation system that tests multiple small language models (0.5b-3b parameters) across different personality configurations (baseline, base personalities, Chain of Thought prompts) and standardized LLM benchmarks, with automated documentation updates.

## Motivation

Currently, the benchmarking system evaluates single models with personality variations, but lacks:
1. **Cross-model comparison**: No systematic way to compare how different models respond to the same personality configurations
2. **COT prompt evaluation**: Missing Chain of Thought prompt support for reasoning benchmarks
3. **Automated discovery**: Manual model selection instead of automatic detection of available models in target parameter range
4. **Documentation automation**: Manual updates to whitepaper and README after benchmark runs

This change enables comprehensive evaluation of how personality configurations and prompt strategies affect different model architectures, providing data-driven insights for model selection and personality optimization.

## Scope

### In Scope
- Matrix evaluation runner that tests all combinations of:
  - Models: Auto-detected Ollama models in 0.5b-3b parameter range
  - Personalities: Baseline, 5 base personalities, COT prompts
  - Benchmarks: MMLU, HellaSwag, GSM8K, ARC-Easy, TruthfulQA
- COT prompt integration for reasoning tasks
- Model size detection and filtering
- Results aggregation and statistical analysis
- Automated whitepaper and README updates with results tables

### Out of Scope
- Model fine-tuning or training
- New benchmark implementations (uses existing lm-eval harness)
- Real-time evaluation dashboard (results stored in JSON/Markdown)
- Model performance optimization (evaluation only)

## Dependencies

- Existing `lm-eval` infrastructure (`run_lm_eval_mass.py`)
- Existing personality system (`run_personality_benchmark.py`)
- Ollama API for model discovery
- Whitepaper and README structure for auto-updates

## Success Criteria

1. ✅ Matrix evaluation runs successfully for all model × personality × benchmark combinations
2. ✅ COT prompts are correctly applied to reasoning benchmarks
3. ✅ Model discovery automatically filters to 0.5b-3b parameter range
4. ✅ Results are aggregated with statistical analysis (mean, std dev, improvements)
5. ✅ Whitepaper Section 4.2.5 is automatically updated with matrix results
6. ✅ README matrix table is automatically updated with latest results
7. ✅ All tests pass and integration is validated

## Risks & Mitigations

- **Risk**: Large evaluation matrix may take significant time to complete
  - **Mitigation**: Support `--limit` flag for quick testing, checkpoint/resume functionality
- **Risk**: Model size detection may be inaccurate for some model names
  - **Mitigation**: Fallback to manual model list, clear error messages
- **Risk**: Whitepaper/README updates may break formatting
  - **Mitigation**: Use marker-based updates, validate markdown syntax

## Related Work

- Extends `scripts/run_personality_benchmark.py` with COT support
- Leverages `scripts/run_lm_eval_mass.py` for benchmark execution
- Uses `src/benchmark/utils.py` for answer checking and prompt building

