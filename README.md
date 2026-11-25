# AXIOM (axiom-local)

Local-first, deterministic and verifiable data intelligence powered by a specialized language model.

AXIOM is a small, offline data-analysis workbench. A deterministic Python statistical engine performs all
calculations; an optional, locally fine-tuned open-weight model (planned name: AX-01) only routes user intent
to typed analysis plans and never computes or alters results. Target platforms: macOS on Apple Silicon first,
Windows later.

## Status

**Early development.** The repository contains governance, CI and the first version of the deterministic
statistical core (Issue #2, under review). **No language model, desktop UI or trained model exists yet.**

| Area | Status |
|------|--------|
| Package skeleton, tests, lint, CI | Implemented (Issue #1) |
| CSV input guard, typed plans, canonical results | Implemented (Issue #2), see [ADR 0003](docs/decisions/0003-input-guard-and-dataset-snapshot.md), [ADR 0005](docs/decisions/0005-typed-plans-binding-and-result-serialization.md) |
| Inspect, Relate (Pearson/Spearman), Compare, Cluster (K-means) | Implemented (Issue #2); measured on one macOS arm64 machine, see [EXP-001](docs/evidence/experiments/EXP-001-issue2-numerical-validation/MANIFEST.md) |
| Correlation matrix, outlier flags, p-values | Not implemented (deferred or out of scope) |
| Run passport, replay, cross-platform comparison | Planned (Issue #3) |
| Intent router (AX-01) | Planned, no model exists |
| Desktop UI | Planned |
| Model training | Planned, no results exist |

See [docs/roadmap.md](docs/roadmap.md) and [docs/evidence/CLAIMS.md](docs/evidence/CLAIMS.md).

## Architecture

Typed plan → deterministic engine → canonical result with identities (raw-byte hash, typed dataset hash, plan hash,
result hash). A hash shows identity and integrity, not correctness. Details and unvalidated assumptions:
[docs/architecture.md](docs/architecture.md) and the [ADRs](docs/decisions/README.md).

## Install and test

Requires Python 3.12 (3.11 is not supported, see [ADR 0002](docs/decisions/0002-dependency-stack-and-python-version.md))
and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked      # create .venv from uv.lock (runtime: numpy, pydantic, scikit-learn, threadpoolctl; dev: pytest, ruff)
uv run pytest         # tests
uv run ruff check .   # lint
uv run ruff format --check .
```

## Library example

```python
from axiom import ColumnSpec, CsvProfile, load_csv, parse_plan_json, run

profile = CsvProfile(
    columns=(ColumnSpec(name="region", kind="text"), ColumnSpec(name="income", kind="numeric"))
)
dataset = load_csv("data.csv", profile)  # explicit schema; nothing is guessed
plan = parse_plan_json(
    '{"schema_version":"axiom.plan/1","operation":"compare","group_by":"region","measure":"income"}'
)
result = run(dataset, plan)
print(result.canonical_json, result.result_sha256)
```

A complete synthetic example with one plan per operation is in [examples/](examples/README.md).

The exact commands, environment and exit statuses of the last verified run are recorded in
[docs/evidence/DEVELOPMENT_LOG.md](docs/evidence/DEVELOPMENT_LOG.md).

## Repository map

- `src/axiom/` – the deterministic core (`ingest`, `plan`, `kernels`, `ops_*`, `engine`, `canonical`)
- `tests/` – tests, exact-arithmetic oracles, vendored NIST reference data
- `benchmarks/` – scripts that produced the numbers in `docs/evidence/experiments/`
- `docs/` – architecture, roadmap, reproducibility, demo, ADRs, research, evidence
- `examples/` – synthetic example data and plans; `training/` – placeholder, nothing implemented
- `AGENTS.md` – canonical contribution and engineering rules

## License and third-party notices

MIT, see [LICENSE](LICENSE). Dependency/model/data notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Sources: [REFERENCES.md](REFERENCES.md). Citation metadata: [CITATION.cff](CITATION.cff) (no release or DOI exists).
