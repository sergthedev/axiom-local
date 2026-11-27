# AXIOM (axiom-local)

Open-source research prototype for local, bounded and reproducible statistical analysis; specialized bilingual intent
parsing under development.

AXIOM is a small, offline data-analysis workbench. A deterministic Python statistical engine performs all
calculations. A planned optional intent router (AX-01) may only propose one of four supported analyses for the person
to confirm; it never computes or alters results. **No model has been trained or evaluated for AXIOM yet.** Target
platforms: macOS on Apple Silicon first, Windows later.

## Status

**Research prototype, early development.** The repository contains governance, CI, the deterministic statistical
core (Issue #2) with verification and replay (Issue #3), and, under review for Issue #4, the router candidate
contract, the AXIOM Guard, the AXBENCH benchmark tooling and a rule baseline. **No trained model, no model
evaluation and no desktop UI exist yet.**

| Area | Status |
|------|--------|
| Package skeleton, tests, lint, CI | Implemented (Issue #1) |
| CSV input guard, typed plans, canonical results | Implemented (Issue #2), see [ADR 0003](docs/decisions/0003-input-guard-and-dataset-snapshot.md), [ADR 0005](docs/decisions/0005-typed-plans-binding-and-result-serialization.md) |
| Inspect, Relate (Pearson/Spearman), Compare, Cluster (K-means) | Implemented (Issue #2); measured on one macOS arm64 machine, see [EXP-001](docs/evidence/experiments/EXP-001-issue2-numerical-validation/MANIFEST.md) |
| Correlation matrix, outlier flags, p-values | Not implemented (deferred or out of scope) |
| Analysis Passport, fresh-process replay, cross-platform comparison | Implemented (Issue #3, accepted; ADRs 0007 and 0008 ratified), see [ADR 0007](docs/decisions/0007-analysis-passport-and-environment-capture.md), [ADR 0008](docs/decisions/0008-replay-mismatch-classification-and-comparison-criteria.md), [EXP-004](docs/evidence/experiments/EXP-004-replay-benchmark/MANIFEST.md), [EXP-005](docs/evidence/experiments/EXP-005-cross-platform/MANIFEST.md) |
| Router candidate contract, schema projection, AXIOM Guard (confirmation before any dispatch) | Implemented, under review (Issue #4), see [ADR 0009](docs/decisions/0009-ax01-candidate-contract-and-schema-projection.md), [ADR 0010](docs/decisions/0010-axiom-guard-confirmation-and-dispatch.md) |
| AXBENCH bilingual routing benchmark (synthetic draft, unreviewed) and rule baseline | Draft tooling under review (Issue #4), see [dataset card](data/axbench/DATASET_CARD.md), [ADR 0011](docs/decisions/0011-axbench-benchmark-metrics-and-sealing.md); rule baseline measured on validation only ([EXP-008](docs/evidence/experiments/EXP-008-axbench-rule-baseline/MANIFEST.md)) |
| AXBENCH-TR v0.2 Turkey-first data acquisition and draft corpus (unreviewed; nothing trained) | Draft under review, see [dataset card](data/axbench-tr/DATASET_CARD.md), [DATA_SOURCES.md](data/axbench-tr/DATA_SOURCES.md), [EXP-TR01](docs/evidence/experiments/EXP-TR01-axbench-tr-acquisition/MANIFEST.md) |
| AX-01 language model | Not trained; candidate checkpoints and an optional Apple Silicon environment prepared ([ADR 0012](docs/decisions/0012-optional-model-environment.md)) |
| Desktop UI | Planned, not started |
| Model training | Planned, protocols only; no results exist |

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

## Verify and replay

```bash
uv run python -m axiom analyze --data examples/synthetic_regions.csv \
  --profile examples/profiles/synthetic_regions.json --plan examples/plans/compare.json --out passport.json
uv run python -m axiom replay --passport passport.json --data examples/synthetic_regions.csv --runs 5
```

The passport records the identities, inputs, full-precision result and environment; replay recomputes in fresh processes
and reports `exact_match`, `numeric_agreement`, `numerical_discrepancy`, `execution_failure` or `integrity_mismatch`.
Repeatability and agreement are not correctness ([docs/reproducibility.md](docs/reproducibility.md)).

The exact commands, environment and exit statuses of the last verified run are recorded in
[docs/evidence/DEVELOPMENT_LOG.md](docs/evidence/DEVELOPMENT_LOG.md).

## Repository map

- `src/axiom/` – the deterministic core (`ingest`, `plan`, `kernels`, `ops_*`, `engine`, `canonical`) and verification
  (`environment`, `passport`, `replay`, `agreement`, `partition`; CLI `python -m axiom`)
- `src/axiom_router/` – router candidate contract, schema projection, semantic validation, Guard, prompt, rule router
  (separate from `axiom`, so Core identities do not change)
- `src/axbench/` – AXBENCH generator, audits, scoring, statistics, sealing (CLI `python -m axbench`);
- `src/axbench_tr/` – AXBENCH-TR acquisition, real-schema inventory, corpus builder and validator (CLI `python -m axbench_tr`; raw data stays in the ignored `data/axbench-tr/_raw/`; `pyarrow==26.0.1` is needed only to rebuild the Cognipeer part and is not in the lockfile);
  `data/axbench/` – draft corpus and dataset card (no final test split)
- `envs/model/` – optional Apple Silicon MLX-LM environment (separate lock, not needed for Core or CI)
- `tests/` – tests, exact-arithmetic oracles, vendored NIST reference data
- `benchmarks/` – scripts that produced the numbers in `docs/evidence/experiments/`
- `docs/` – architecture, roadmap, reproducibility, demo, ADRs, research, evidence
- `examples/` – synthetic example data and plans; `training/` – protocols only, no training run
- `AGENTS.md` – canonical contribution and engineering rules

## License and third-party notices

MIT, see [LICENSE](LICENSE). Dependency/model/data notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Sources: [REFERENCES.md](REFERENCES.md). Citation metadata: [CITATION.cff](CITATION.cff) (no release or DOI exists).
