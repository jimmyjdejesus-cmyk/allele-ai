# Example Agent Spec — Review Assistant

## Basic Info
- **Agent name**: Review Assistant
- **Owner**: Jimmy De Jesus / @jimmyjdejesus-cmyk
- **Purpose / Description**: A helper agent that reads PR diffs and suggests potential issues and test suggestions.
- **Scope / Boundaries**: Read-only access to PR diffs, no writes to repositories or external services without human approval.

## Inputs & Outputs
- **Inputs**: PR diff text, optional checklist template
- **Outputs**: Human-readable suggestions (markdown), annotated checklist

## Data & Model
- **Model(s)**: `gpt-4-turbo` (provider: OpenAI, license: proprietary)
- **Data sources**: PR diffs and repo metadata (no external PII)
- **PII / Privacy concerns**: None expected; any identified PII should be redacted and reported

## Security & Access
- **Permissions required**: read-only GitHub token to fetch PR diffs
- **Secrets handling plan**: Token via GitHub Actions secrets or Vault; never stored in code
- **Network access**: Outbound calls to OpenAI only (via `GITHUB_ACTIONS` env) — documented and audited

## Safety & Mitigations
- **Known failure modes**: hallucinations, incorrect code suggestions
- **Adversarial scenarios to test**: malformed diffs, extremely large PRs
- **Mitigation controls**: max tokens, rate limits, human approval required before applying changes

## Testing & Acceptance
- **Unit tests**: parse diffs into sections, validate redaction of any email/PII
- **Integration tests**: run against known PR fixtures; ensure suggestions match expectations
- **Safety tests**: adversarial prompts that attempt to reveal secrets should be blocked
- **Acceptance criteria**: suggestions are relevant and non-actions are recommended only (no automatic code modifications)

## Deployment & Monitoring
- **Deployment targets**: staging via GitHub Actions, manual promotion to production
- **Canary / rollout plan**: 10% of CI runs enabled for the agent for two weeks
- **Monitoring metrics & alerts**: suggestion acceptance rate, false-positive rate, latency
- **Rollback plan**: disable agent in GitHub Actions or remove secret to stop inference

## Compliance & Approvals
- **Required approvals**: Security, Privacy

## Additional notes
- **Related issues/PRs**: #14 (agent guidelines)

---

**Checklist (final)**
- [x] Spec complete
- [x] Tests added
- [ ] Safety review completed
- [ ] CI passed
- [ ] Owner and rollback plan defined
