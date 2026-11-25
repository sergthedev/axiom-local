# AXIOM (axiom-local)

Local-first, deterministic and verifiable data intelligence powered by a specialized language model.

AXIOM is a small, offline data-analysis workbench. A deterministic Python statistical engine performs all
calculations; an optional, locally fine-tuned open-weight model (planned name: AX-01) only routes user intent
to typed analysis plans and never computes or alters results. Target platforms: macOS on Apple Silicon first,
Windows later.

## Status

**Bootstrap only.** This repository currently contains governance, documentation scaffolding, CI and an empty
Python package. **No analysis feature, model or UI is implemented yet.**

| Area | Status |
|------|--------|
| Package skeleton, tests, lint, CI | Implemented (Issue #1) |
| Inspect, Relate, Cluster, Compare engine | Planned |
| Typed plans / intent router (AX-01) | Planned |
| Desktop UI | Planned |
| Benchmarks, model training | Planned, no results exist |

See [docs/roadmap.md](docs/roadmap.md) and [docs/evidence/CLAIMS.md](docs/evidence/CLAIMS.md).

## Planned architecture

Typed plan → deterministic engine → verifiable result with provenance. Details and unvalidated assumptions:
[docs/architecture.md](docs/architecture.md) and [ADR 0001](docs/decisions/0001-initial-architecture.md).

## Install and test

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked      # create .venv from uv.lock (runtime: no third-party dependencies; dev: pytest, ruff)
uv run pytest         # tests
uv run ruff check .   # lint
uv run ruff format --check .
```

The exact commands, environment and exit statuses of the last verified run are recorded in
[docs/evidence/DEVELOPMENT_LOG.md](docs/evidence/DEVELOPMENT_LOG.md).

## Repository map

- `src/axiom/` – package (placeholder)
- `tests/` – tests
- `docs/` – architecture, roadmap, reproducibility, demo, ADRs, research, evidence
- `examples/`, `training/`, `benchmarks/` – placeholders, nothing implemented
- `AGENTS.md` – canonical contribution and engineering rules

## License and third-party notices

MIT, see [LICENSE](LICENSE). Dependency/model/data notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Sources: [REFERENCES.md](REFERENCES.md). Citation metadata: [CITATION.cff](CITATION.cff) (no release or DOI exists).
