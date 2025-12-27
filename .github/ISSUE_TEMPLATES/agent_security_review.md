---
name: "Agent security review"
about: "Request a focused security & privacy review for agent changes"
title: "Security review: New/changed agent - confirm spec, access & mitigations"
labels: [security-review, agent]
---

**Summary**
Please perform a focused security & compliance review of the agent changes in this PR. Confirm that the agent has an attached, complete spec and that security, privacy, and operational mitigations are in place.

**Checklist (actionable items & acceptance criteria)**
- [ ] Spec attached & complete (Agent name, Owner, Permissions listed)
- [ ] trust_remote_code reviewed & justified (or removed)
- [ ] Secrets & credential usage verified (no committed secrets)
- [ ] Least privilege & permission scope documented
- [ ] Data retention & PII handling documented
- [ ] Logging & monitoring defined
- [ ] Safety/adversarial test plan included
- [ ] Incident response & rollback plan provided
- [ ] Dependency & license review completed
- [ ] Security & Privacy sign-off (comment/signature)

**Suggested reviewers:** @jimmyjdejesus-cmyk, @security-team (placeholder), @privacy-team (placeholder), @qa-team (placeholder)

**Priority:** High — initial review requested within 3 business days

**Reviewer feedback template**
- Risk areas (TRC, secrets, permissions, PII): [OK / Issue: <note>]
- Spec completeness: [OK / Missing: <fields>]
- Tests & monitoring: [OK / Missing: <items>]
- Approval: [Approve / Request changes]
