# Agent Specification Template 🧾

> Fill out this template when proposing a new agent. Attach it to the agent proposal issue.

## Basic Info
- **Agent name**: 
- **Owner** (name / GitHub): 
- **Purpose / Description**: Short summary of what the agent does.
- **Scope / Boundaries**: (what it may and may not do)

## Inputs & Outputs
- **Inputs**: (types, shape, sources)
- **Outputs**: (types, destinations)

## Data & Model
- **Model(s)**: (name, version, license)
- **Data sources**: (what data is used, retention policy)
- **PII / Privacy concerns**: (yes/no; mitigation)

## Security & Access
- **Permissions required**: (secrets, APIs, write access)
- **Secrets handling plan**: (how secrets will be stored and scoped)
- **Network access**: (external calls allowed?)

## Safety & Mitigations
- **Known failure modes**:
- **Adversarial scenarios to test**:
- **Mitigation controls**: (rate limits, human-in-the-loop, filters)

## Testing & Acceptance
- **Unit tests**: (what to test)
- **Integration tests**: (what to test)
- **Safety tests**: (adversarial test plan)
- **Acceptance criteria**: (concrete pass/fail)

## Deployment & Monitoring
- **Deployment targets**: (staging/production)
- **Canary / rollout plan**:
- **Monitoring metrics & alerts**:
- **Rollback plan**:

## Compliance & Approvals
- **Required approvals**: (security, privacy, legal)
- **Approvals** (list reviewers and dates):

## Additional notes
- **Related issues/PRs**:
- **Documentation to update**:

---

**Checklist (final)**
- [ ] Spec complete
- [ ] Tests added
- [ ] Safety review completed
- [ ] CI passed
- [ ] Owner and rollback plan defined
