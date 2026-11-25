---
name: axiom-pr
description: Review and evidence checklist for AXIOM pull requests. Use before opening or reviewing a PR to confirm tests, references, claims, privacy and scope are in order.
---

# axiom-pr

Canonical rules are in `AGENTS.md`; this skill is only a checklist.

- Scope: matches the linked issue only; no unrelated features.
- Tests: ran `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`; report real exit statuses.
- Evidence: `REFERENCES.md`, ADRs, `docs/evidence/` updated where needed; claims carry a status and evidence link;
  no unmeasured success rates.
- Statistics: known-answer tests or an independent reference; a hash alone is not proof of accuracy.
- Privacy/licensing: no credentials, personal data, absolute local paths; data synthetic or licensed and attributed.
- Process: do not merge or approve your own PR; no force-push; wait for review.
