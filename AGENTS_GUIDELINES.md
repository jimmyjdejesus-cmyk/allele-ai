# AI Agent Guidelines ✅

## Purpose
Provide a single, opinionated, and actionable set of rules, workflows, and templates for proposing, building, testing, and operating AI agents in this repository.

## Scope
Applies to all agent-related code, tests, CI, documentation, and deployed agents that use this repository (including experimental and research agents).

---

## Principles 💡
- **Safety first**: Avoid actions that can cause harm, leak secrets, or violate privacy.
- **Transparency**: All agent capabilities, data sources, and limitations must be documented.
- **Least privilege**: Agents must request only the permissions they need.
- **Human oversight**: A human owner must be named and reachable for each agent.
- **Auditability**: Actions and decisions must be logged to enable incident investigation.

---

## Hard Rules (must follow) ⚠️
- Do NOT embed secrets or credentials in code or in prompts; use secure secrets management.
- Do NOT allow direct writes to production systems without an explicit human approval step or automated safeguards and tests.
- Do NOT train or expose personal data or PII unless explicitly approved and documented with data handling controls.
- All agents must have a named owner and at least one reviewer for safety/security.

---

## Lifecycle / Workflow 🔁
1. **Proposal**: Open an issue using `.github/ISSUE_TEMPLATES/agent_proposal.md` and attach a filled `AGENTS/TEMPLATES/agent_spec_template.md`.
2. **Design & Spec**: Complete the agent spec including scope, inputs/outputs, safety mitigations, and acceptance criteria.
3. **Review**: Peer review for correctness and a safety/security review (explicit sign-off from security or owner).
4. **Sandbox & Tests**: Run in an isolated environment; include unit tests, integration tests, and adversarial safety tests.
5. **CI / Compliance**: Pass `agent-safety-check` workflow and all CI jobs.
6. **Staged Deployment**: Deploy behind feature flags or canaries; monitor metrics and logs.
7. **Operate & Monitor**: Logging, telemetry and alerts configured; periodic re-review for drift and failures.
8. **Incident Response**: Use `.github/ISSUE_TEMPLATES/agent_incident.md` for incident reporting and follow the documented mitigation path.

---

## Templates & Artifacts 📁
- **Agent spec template**: `AGENTS/TEMPLATES/agent_spec_template.md` (required for proposals)
- **Issue templates**: `.github/ISSUE_TEMPLATES/agent_proposal.md`, `.github/ISSUE_TEMPLATES/agent_incident.md`
- **CI workflow**: `.github/workflows/agent-safety-check.yml` (runs static and sandbox checks)

---

## Testing & Evaluation ✅
- Tests must include:
  - Unit tests for core logic
  - Integration tests for API/IO boundaries
  - Safety/adversarial tests that probe for unwanted behaviors
  - Load/latency tests if agent is performance-sensitive
- Define evaluation metrics in the spec and include benchmark results in PR.

---

## Security & Secrets 🔐
- Use environment variables and GitHub secrets for credentials.
- Review third-party data sources and models for licensing and privacy implications.
- Limit external network access during sandbox testing by default.

---

## Monitoring & Logging 📊
- Log agent actions, decisions, and errors with enough context for debugging.
- Define SLOs and alert thresholds in the agent spec.

---

## Owners & Approvals 🧭
- Every agent must state an **owner** and **reviewers** in its spec.
- Some changes require additional approvals (security, privacy, or legal) — list these in the spec.

---

## Enforcement & Reviews
- Maintainers may block or revert agents that violate these guidelines.
- Regular audits (quarterly suggested) should be scheduled for production agents.

---

For agent changes, file a security review using `.github/ISSUE_TEMPLATES/agent_security_review.md` and request Security/Privacy signoff before merging.

## Quick PR checklist ✅
- [ ] Agent spec attached and complete
- [ ] Tests added and passing locally
- [ ] Safety and security sign-offs recorded
- [ ] CI `agent-safety-check` passed
- [ ] Monitoring & rollback plan in the spec
- [ ] Owner and reviewers listed

---

If you have questions or want help filling out the templates, open an issue or contact the repository owners.
