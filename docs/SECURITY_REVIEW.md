# Security Review for Agents

This document explains the security review workflow for agents in this repository.

## When to request a review
- Any PR that **adds** or **changes** agent code (`AGENTS/**`, `src/**/agent*.py`, or files containing `agent` in their name).
- Any change that introduces new permissions, secrets usage, external network access, or `trust_remote_code=True`.

### trust_remote_code policy
- **Default**: `trust_remote_code` must be **disabled** by default. Any use of `trust_remote_code=True` must be explicitly opt-in, justified in the agent spec, and approved through the security review process.
- The repository includes an automated static check (`scripts/agent_safety_check.py`) which will warn about `trust_remote_code` occurrences; CI can be configured to fail on detection by setting `AGENT_SAFETY_FAIL_ON_TRC=1` in the workflow environment.

## How to request a review
1. Open your PR with your changes and attach a filled agent spec (see `AGENTS/TEMPLATES/agent_spec_template.md`).
2. Use the security issue template: **New issue → _Agent security review_** or click **Create Issue** and choose the template `Agent security review`.
3. Add the `security-review` label to the issue (the repository provides this label) and mention the PR (e.g., `Related PR: #14`).

## Checklist
Use the checklist in `.github/ISSUE_TEMPLATES/agent_security_review.md`. Key items:
- Spec attached & complete
- `trust_remote_code` reviewed & justified
- Secrets & credential usage verified
- Least-privilege enforced for APIs and write access
- Data retention & PII handling documented and approved
- Logging & monitoring defined
- Safety/adversarial tests included
- Incident response & rollback plan provided
- Dependency & license review completed
- Security & Privacy sign-off recorded

## Review timeline & expectations
- **Priority**: High for PRs that request external access/secrets or write permissions; otherwise Medium.
- **Initial review timeframe**: 3 business days for an initial assessment.
- Reviewers should include Security and Privacy representatives when PII or external data sources are involved.

## Overrides
- Overrides are allowed only in emergencies and must be recorded. Use the label `allow-agent-without-spec` or set `AGENT_SAFETY_IGNORE_SPEC=1` in the workflow, and add a comment in the security issue explaining the reason.
- Overrides require a follow-up retrospective and an owner to file a retroactive agent spec.

## Follow-up
- When the security review is complete, record the approval comment on the issue using the format:
  `Security approved by @<handle> on YYYY-MM-DD`.

## Notes
- This repository includes an automated `agent-safety-check` workflow that performs static scans. The security review is a complementary human step for risk evaluation and approvals.
