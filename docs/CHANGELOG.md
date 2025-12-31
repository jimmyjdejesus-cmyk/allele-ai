# Changelog

All notable changes to this repository will be documented in this file.

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
