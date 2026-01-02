# Repository Organization

## Directory Structure

### Root Directory
**Active Documentation**:
- `README.md` - Main project documentation
- `MATRIX_EVALUATION_FINDINGS.md` - Key findings from matrix evaluation
- `AGENTS.md` - Agent guidelines overview
- `AGENTS_GUIDELINES.md` - Detailed agent guidelines
- `CONTRIBUTING.md` - Contribution guidelines
- `CODE_OF_CONDUCT.md` - Code of conduct
- `LICENSE` - AGPL v3 license

**Configuration**:
- `pyproject.toml` - Project configuration
- `ruff.toml` - Linting configuration
- `mkdocs.yml` - Documentation configuration
- `.gitignore` - Git ignore patterns

### Source Code
- `src/phylogenic/` - Main package code
- `src/benchmark/` - Benchmark utilities
- `scripts/` - Utility scripts
- `tests/` - Test suite
- `benchmarks/` - Benchmark implementations
- `examples/` - Usage examples

### Documentation
- `docs/` - Main documentation
  - `whitepaper/` - Academic whitepaper
  - `api/` - API documentation
  - `archive/` - Historical documents (archived)
- `AGENTS/` - Agent specifications and templates

### Benchmark Results
- `benchmark_results/` - Benchmark evaluation results
  - `matrix_full_expanded/` - Latest full matrix evaluation (180 configs)
  - `matrix_validation_deep_plan/` - Validation run results
  - `matrix_evaluation/` - Active evaluation directory
  - `lm_eval/` - LM-eval harness results
  - `archive/` - Historical results (archived)

### OpenSpec
- `openspec/` - OpenSpec change proposals
  - `changes/` - Individual change proposals
  - `project.md` - Project specification

### Launch Materials
- `launch_materials/` - Marketing and launch materials

## Archive Directories

### `docs/archive/`
Contains historical analysis, planning, and implementation documents:
- Implementation reports
- Analysis documents
- Planning documents
- Technical debt documentation
- Verification reports

### `benchmark_results/archive/`
Contains historical benchmark results:
- Old A/B test results
- Historical personality benchmarks
- Test/demo evaluation runs
- Planning documents

## File Naming Conventions

### Documentation
- `*.md` - Markdown documentation
- `*_FINDINGS.md` - Research findings
- `*_ANALYSIS.md` - Analysis reports
- `*_REPORT.md` - Status reports

### Benchmark Results
- `results.json` - Evaluation results
- `checkpoint.json` - Progress checkpoint
- `analysis.md` - Analysis report

### Scripts
- `run_*.py` - Execution scripts
- `analyze_*.py` - Analysis scripts
- `update_*.py` - Update scripts
- `*_test.py` - Test scripts

## Cleanup Rules

### Files to Archive
- Historical analysis documents
- Old benchmark results
- Planning documents (after completion)
- Status reports (after completion)

### Files to Remove
- Backup files (`*.backup.*`)
- Log files (`*.log`) - should be in logs/ or ignored
- Temporary files
- Duplicate documentation

### Files to Keep in Root
- Active documentation (README, LICENSE, etc.)
- Key findings documents
- Configuration files
- Project guidelines

## Maintenance

### Regular Cleanup
1. Move completed analysis documents to `docs/archive/`
2. Archive old benchmark results to `benchmark_results/archive/`
3. Remove backup files and logs
4. Update `.gitignore` as needed

### Before Committing
1. Check for backup files
2. Check for log files in root
3. Verify archive directories are properly organized
4. Ensure `.gitignore` is up to date

