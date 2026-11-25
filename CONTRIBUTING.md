# Contributing

Rules live in [AGENTS.md](AGENTS.md); this page is the short version.

1. Work only from an approved GitHub issue.
2. Branch from `main`; make focused commits.
3. Run `uv sync --locked`, `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`.
4. Update `REFERENCES.md`, ADRs and `docs/evidence/` when relevant. Claims need evidence.
5. Open a PR using the template and link the issue. Wait for technical review; the product owner approves merges.
6. Never include credentials, personal data, real payroll data or absolute local paths.

Datasets used in tests, examples or benchmarks must be synthetic or explicitly licensed, with attribution in
`THIRD_PARTY_NOTICES.md`.
