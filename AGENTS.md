# AGENTS.md — canonical engineering rules for AXIOM

This file is the single canonical source of rules for every contributor, human or AI.
`CLAUDE.md` only points here; do not duplicate rules in other files.

## Project in one paragraph

AXIOM is a small, offline, verifiable data-analysis workbench (macOS on Apple Silicon first, Windows later).
It pairs a deterministic statistical engine with an optional, locally fine-tuned open-weight intent router.
All numbers come from the engine; the model never calculates or alters results.

## Roles

| Role | Who |
|------|-----|
| Product owner (approves merges) | Repository owner |
| Technical lead, research lead, reviewer | ChatGPT |
| Implementation engineer | Claude Code |

GitHub is the single source of truth. Issues define scope; pull requests carry review.

## Workflow (mandatory)

1. Read the assigned issue in full, including comments and later acceptance criteria.
2. Read this file, relevant ADRs in `docs/decisions/`, and research in `docs/research/`.
3. Inspect the repository and `git status` before editing.
4. Implement only the approved issue. Do not expand scope.
5. Branch from `main`, make focused commits, run the real tests locally.
6. Update `REFERENCES.md`, ADRs and `docs/evidence/` when the change touches them.
7. Push the branch and open a pull request that links the issue. Fill in the PR template honestly.
8. Report results and blockers on the issue. Stop and wait for review.

Never merge or approve your own PR. Never force-push. Never bypass acceptance criteria.
Research gates apply before each substantive implementation (not to the empty-repository bootstrap).

## Scope lock

Capabilities: **Inspect** (descriptive statistics, data quality), **Relate** (Pearson, Spearman),
**Cluster** (reproducible K-means), **Compare** (group summaries). CSV first.

Out of scope: authentication, cloud services, telemetry, API keys, execution of model-generated code,
RAG, forecasting. The application runtime must work offline after initial dependency/model setup.

## Scientific integrity

- Determinism is not correctness. Validate against independent references and known-answer tests.
- Keep four statuses apart: **proposed**, **implemented**, **measured**, **independently verified**.
- Never fabricate benchmarks, training logs, citations, screenshots or results.
  Do not publish unmeasured success rates. A hash alone does not prove numerical accuracy.
- Record failures and limitations next to successes.
- Keep training, validation and held-out test data separated; prevent evaluation leakage.
- Demo datasets must be synthetic or explicitly licensed and attributed. No real personal records.

## Evidence and provenance

- Every significant technical decision needs an ADR (`docs/decisions/`).
- Every externally sourced factual claim needs a `REF-NNN` entry in `REFERENCES.md`
  (author/publisher, title, canonical URL or DOI, version/revision, access date, supported claim, related issue/ADR).
  Record only sources actually consulted. Planned reading is not a reference.
- Claims about the project go in `docs/evidence/CLAIMS.md` with a status and an evidence link.
- Experiments use the manifest template in `docs/evidence/experiments/`.
- Log actual AI/tool contributions and human checks in `docs/evidence/AI_CONTRIBUTIONS.md`.
- Chronology goes in `docs/evidence/DEVELOPMENT_LOG.md`.

## Security and privacy

- No credentials, personal data, private conversations, real payroll data or absolute local paths in the repository,
  commits, issues or PR text. Sanitize logs before committing.
- No telemetry, cloud inference or external APIs in the application runtime.
- Never execute model-generated code.
- Do not install third-party skills, agents or MCP servers without owner approval.

## Quality

- Prefer small, maintainable, well-tested code over feature count.
- Docs must describe the current implementation; mark unimplemented features as planned.
- Public documentation is written in professional English.
- Lint (`ruff`) and tests (`pytest`) must pass locally before pushing.

## Repository-owned skills

Only two minimal skills exist, under `.claude/skills/`: `axiom-source` (provenance workflow) and
`axiom-pr` (review/evidence checklist). Others are deferred until needed.
