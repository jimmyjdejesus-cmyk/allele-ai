# CI & Developer Checks

This document explains the repository CI, how it was hardened, and how to run important checks locally.

## What changed recently
- Matrix installs limited to `.[dev]` to keep PR jobs fast and avoid optional heavy extras.
- Added a `heavy-deps` job to perform `.[docs,all]` install in isolation.
- Removed silent `|| true` installs and replaced with best-effort stub installers that log warnings.
- Added `workflow-lint` job using `yamllint` to catch YAML/syntax issues early.
- Added `agent-safety-check` workflow that runs `scripts/agent_safety_check.py` (scans for secret-like patterns and `trust_remote_code` usage). See `.github/workflows/agent-safety-check.yml`.

## Running checks locally

### Requirements
Make sure Python 3.9+ is installed. Use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### Lint & format

Run ruff:

```bash
python3 -m ruff check src/ tests/
# auto-fix where sensible
python3 -m ruff check src/ tests/ --fix
```

Run mypy:

```bash
mypy src/
```

### Tests

```bash
pytest -q
```

### Run agent safety scan locally

```bash
python scripts/agent_safety_check.py
```

This script scans `src/` and `AGENTS/` for secret-like patterns and reports `trust_remote_code=True` occurrences.

### Agent-spec check
We added `scripts/check_agent_spec.py` and `.github/workflows/agent-spec-check.yml` to enforce presence of an agent spec when agent code changes are detected. The script supports a `--local` mode for developers to run the check before pushing (see `scripts/check_agent_spec.py --help`).

The check looks for agent-related file changes (files under `AGENTS/`, `src/**/agent*.py`, or filenames containing `agent`) and fails the PR if no matching agent spec is attached. An override is available via the PR label `allow-agent-without-spec` or the env var `AGENT_SAFETY_IGNORE_SPEC=1` (documented in `docs/SECURITY_REVIEW.md`).

## Troubleshooting
- If CI fails due to missing optional typing stubs, consult the pip logs uploaded by the workflow artifacts and reference the heavy-deps job logs.
- If `ruff` or `mypy` reports errors, run the local commands above and apply `ruff --fix` first.

## Contact
If CI behavior is confusing, open an issue and tag `@jimmyjdejesus-cmyk` and the Security team if relevant.
