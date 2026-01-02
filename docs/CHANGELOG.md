# Changelog

All notable changes to this repository will be documented in this file.

## [1.0.2] - 2026-01-01
### Added
- Comprehensive matrix evaluation results (180 configurations)
- Perfect 1.00 benchmark scores achieved for gemma2:2b with COT configurations
- Analysis documentation in `benchmark_results/matrix_full_expanded/analysis.md`

### Changed
- Updated README.md with latest January 2026 benchmark results
- Updated whitepaper to reflect January 2026 version

### Performance
- gemma2:2b + creative_thinker+cot: 1.00 average (perfect score)
- gemma2:2b + concise_analyst+cot: 1.00 average (perfect score)
- gemma2:2b + balanced+cot: 1.00 average (perfect score)
- llama3.2:1b shows +9.7% to +16.6% improvement with COT
- qwen2.5:0.5b shows mixed results with COT (-10.6% to +1.8%)

## [Unreleased] - 2025-12-27
### Added
- `.github/ISSUE_TEMPLATES/agent_security_review.md` - new security review issue template
- `docs/SECURITY_REVIEW.md` - guidance on requesting and running security reviews
- `docs/CI.md` - documentation for CI behavior, troubleshooting, and local checks
- `AGENTS/example_agent_spec.md` - sample filled agent spec to illustrate expectations

### Changed
- `AGENTS_GUIDELINES.md` - added reference to the agent security review flow
- `docs/agents.md` - expanded developer-facing agent lifecycle and quickstart

### Notes
- CI hardening and lint fixes were applied across the repo (pipefail, ruff fixes, heavy-deps job, workflow-lint). See PRs #14, #15.
