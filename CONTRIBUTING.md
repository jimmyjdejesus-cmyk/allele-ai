# Contributing to Phylogenic

Thank you for your interest in contributing to Phylogenic! We welcome contributions from everyone. Please read this guide to get started.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct
This project follows a [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for all contributors. Please be respectful and considerate in your interactions.

## Getting Started
1. Fork the repository on GitHub.
2. Clone your fork locally: `git clone https://github.com/your-username/phylogenic.git`
3. Create a new branch for your changes: `git checkout -b your-feature-branch`

## Development Environment
1. Install Python 3.9 or later.
2. Install dependencies: `pip install -e .[dev]`
3. Set up pre-commit hooks: `pre-commit install`

### OpenSpec dependency

Some documentation and CI jobs rely on the `openspec` CLI. It is distributed separately as an npm package. Please install it globally if you want to run docs or local tools that depend on it:

```bash
# with npm
npm install -g openspec

# with pnpm
pnpm add -g openspec
```

OpenSpec requires Node.js >= 20.19.0.

## Making Changes
- Follow the coding standards in this project.
- Use [ruff](https://github.com/astral-sh/ruff) for linting.
- Use [black](https://github.com/psf/black) for code formatting.
- Use [isort](https://pycqa.github.io/isort/) for import sorting.
- Use [mypy](https://mypy.readthedocs.io/) for type checking.
- Write clear, concise commit messages.

## Testing
- Run the test suite: `pytest`
- Ensure all tests pass before submitting.
- Add tests for new features and bug fixes.

## Submitting Changes
1. Push your changes to your fork: `git push origin your-feature-branch`
2. Create a pull request on GitHub.
3. Fill out the PR template completely.
4. Wait for review and address any feedback.

## Reporting Issues
Use the issue templates provided in the repository to report bugs, request features, or ask questions.

## Agent development
If you're proposing or changing an AI agent, follow the repository's agent lifecycle and templates documented in `AGENTS_GUIDELINES.md` and use the `.github/ISSUE_TEMPLATES/agent_proposal.md` template to begin. Agent-specific work requires a completed spec (`AGENTS/TEMPLATES/agent_spec_template.md`), safety review, and CI `agent-safety-check` sign-off before merging.
