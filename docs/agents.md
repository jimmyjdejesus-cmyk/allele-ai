# Agents (Project Guidelines)

This page summarizes the repository's agent policies, developer guidance, and operational workflows.

## Quick links
- **Agent guidelines & lifecycle**: `AGENTS_GUIDELINES.md`
- **Agent spec template**: `AGENTS/TEMPLATES/agent_spec_template.md`
- **Example filled agent spec**: `AGENTS/example_agent_spec.md`
- **Proposal & incident templates**: `.github/ISSUE_TEMPLATES/agent_proposal.md`, `.github/ISSUE_TEMPLATES/agent_incident.md`
- **Security review template**: `.github/ISSUE_TEMPLATES/agent_security_review.md` (use this to request Security/Privacy review)
- **CI & developer checks**: `docs/CI.md`
- **Security review guidance**: `docs/SECURITY_REVIEW.md`

## Agent lifecycle (practical steps)
1. **Draft Spec**: Fill `AGENTS/TEMPLATES/agent_spec_template.md` with agent name, owner, permissions, inputs/outputs, and safety mitigations.
2. **Open Proposal**: Create a proposal issue using the `Agent proposal` template and attach the filled spec.
3. **Develop in Branch**: Implement agent logic under `src/` or `AGENTS/`, add unit tests, safety tests, and documentation.
4. **Run Local Checks**: Run `ruff`, `mypy`, `pytest`, and `python scripts/agent_safety_check.py` locally.
5. **Request Security Review**: Create an `Agent security review` issue (template available) and link your PR; tag Security and Privacy owners.
6. **Address Feedback**: Incorporate reviewer comments, add missing tests, and re-run CI.
7. **Merge & Monitor**: After approvals, merge and monitor agent behavior in staging; ensure alerts and rollback are in place.

## Example: proposing a simple review agent
- Fill `AGENTS/example_agent_spec.md` as a real example (owner: `@your-handle`, model: `gpt-4-turbo`, no external writes). Attach it to your proposal.
- Add tests in `tests/` and a small benchmark invocation in `scripts/` to validate behavior.

## Best practices
- Keep scope narrow and document clearly in the spec
- Validate assumptions with unit tests and adversarial safety tests
- Limit permissions and avoid embedding secrets in code
- Use the security review for any external integrations or data handling

Use the issue templates to propose new agents or report incidents. Follow the lifecycle in `AGENTS_GUIDELINES.md` for reviews, testing, and deployment.
