# EXP-005 pre-registration: macOS arm64 reference versus GitHub Actions macOS and Windows x64

- Issue: #3. Written and committed **before** any cross-platform comparison was evaluated.
- Scenarios: the same ten as EXP-004 (`benchmarks/issue3_common.py`).
- Reference: passports written by `benchmarks/issue3_cross_platform.py --write-reference` on the development Mac (Apple
  M5, macOS 27.0.1 arm64, CPython 3.12.16, locked environment) into `reference/` in this directory, at the commit
  recorded in each passport.
- Observed: the CI step "Cross-platform numerical comparison (report only)" runs
  `benchmarks/issue3_cross_platform.py --compare` on each `core-tests` runner (`macos-latest`, `windows-latest`) of the
  pull request. The JSON report is read from the job log. The step is report-only: it does not fail on a discrepancy,
  so a green check is **not** evidence of agreement.

## Criteria (fixed now)

- Comparison: `axiom.replay.compare_passports`, criteria `axiom.agreement/1` as written in ADR 0008. Inputs (raw bytes,
  typed dataset, plan digests) must be identical; otherwise the scenario is an integrity mismatch and not compared.
- Per scenario the verdict is recorded as observed: `exact_match`, `numeric_agreement`, `numerical_discrepancy`,
  `integrity_mismatch` or `execution_failure`. Every float field that differs is listed with its ulp distance and
  absolute difference, and cluster partitions are compared by membership sets (ARI auxiliary, iteration counts diagnostic).
- Expectations (predictions, not criteria): Inspect and Compare byte-identical on all runners (exact rational kernels);
  Relate within criteria; Cluster same partition. A failed prediction is reported as such.
- No tolerance will be changed after seeing the results. If a criterion fails, the failure is recorded and escalated.

## Scope limits

- One CI run per runner; GitHub-hosted runner hardware is whatever the run reports. This is not packaged Windows support,
  not Windows ARM64, and not independent human replication.
