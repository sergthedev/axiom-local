---
name: axiom-source
description: Provenance workflow for AXIOM. Use when adding or relying on an external source, dataset, dependency, model or factual claim, so it is recorded in REFERENCES.md and the evidence files.
---

# axiom-source

Canonical rules are in `AGENTS.md`; this skill is only a checklist.

1. Actually consult the source (open it). Never cite from memory or planned reading.
2. Add a `REF-NNN` row to `REFERENCES.md`: author/publisher, title, canonical URL or DOI, version/revision,
   access date, the specific claim it supports, related issue/ADR. Note if you only saw a summary.
3. For dependencies, models or datasets: add license and attribution to `THIRD_PARTY_NOTICES.md`.
4. Link the claim in `docs/evidence/CLAIMS.md` with a status (proposed, implemented, measured, independently verified).
5. If it drives a decision, cite the REF ID in an ADR under `docs/decisions/`.
6. Log it in `docs/evidence/DEVELOPMENT_LOG.md`; record failed lookups too.
7. Keep private paths, chats and secrets out of all of the above.
