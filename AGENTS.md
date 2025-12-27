<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# Agent Guidelines (Overview)

This repository includes guidelines and templates for proposing, developing, testing, and operating AI agents built on top of the Phylogenic framework. The full, canonical guidelines are in `AGENTS_GUIDELINES.md`.

Key resources:

- **Agent rules & lifecycle**: `AGENTS_GUIDELINES.md`
- **Agent spec template**: `AGENTS/TEMPLATES/agent_spec_template.md`
- **Issue templates**: `.github/ISSUE_TEMPLATES/agent_proposal.md`, `.github/ISSUE_TEMPLATES/agent_incident.md`
- **Agent CI workflow (skeleton)**: `.github/workflows/agent-safety-check.yml`

When proposing or modifying an agent, follow the agent lifecycle in `AGENTS_GUIDELINES.md` and use the provided templates. See CONTRIBUTING.md for how to submit PRs and required approvals.