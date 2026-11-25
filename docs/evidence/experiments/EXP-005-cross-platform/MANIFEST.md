# Experiment manifest: EXP-005 — cross-platform numerical comparison

- Issue / PR: Issue #3 (pull request linked from the issue)
- Date: 2026-11-25
- Status: run (one CI run per runner; all ten scenarios byte-identical to the reference on both runners; every expectation held)
- Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md), committed in `f163b41` before any comparison

## Provenance
- Reference: `reference/*.passport.json`, written by `uv run python benchmarks/issue3_cross_platform.py --write-reference docs/evidence/experiments/EXP-005-cross-platform/reference`
  at commit `f163b41f437cbb7dbe40b5d42b78cea5b086fd22` (clean) on Apple M5, macOS 27.0.1 arm64, CPython 3.12.16, locked environment.
  Their result hashes are identical to the EXP-004 reference hashes of the same scenarios.
- Observed: the `core-tests` workflow step "Cross-platform numerical comparison (report only, EXP-005)" on `macos-latest` and
  `windows-latest`.

## Execution
- GitHub Actions run [45677769454](https://github.com/sergthedev/axiom-local/actions/runs/45677769454) for PR head `031360b`
  (tested merge commit `ef0b84bdaaca3719ca708b003fb35459af008bec`, clean): jobs `test (windows-latest)` 139429784851 and
  `test (macos-latest)` 139429785549, both successful (327 tests each). The JSON report printed by the step was copied verbatim from
  each job log into `ci_windows_x64.json` and `ci_macos_arm64.json` (no paths; checked with grep). Observed by the implementer
  from the logs; not yet re-verified by the reviewer.
- The first PR run (45675071394) never reached this step: a test defect failed the Windows job and the macOS job was cancelled
  before a runner was assigned (development log).

## Results (measured only)

| Runner | Python | OS / CPU | NumPy BLAS | Native pools in the worker | Verdicts (10 scenarios) |
|---|---|---|---|---|---|
| Reference (development Mac) | CPython 3.12.16 (Clang 21) | macOS 27.0.1 arm64, Apple M5 | Accelerate | libomp 1 thread; Accelerate unverified | n/a |
| `windows-latest` | CPython 3.12.10 (MSC v.1943, AMD64) | Windows Server 2025 (10.0.26100) x64, AMD Family 25 Model 17, 4 logical CPUs | scipy-openblas 0.3.34 | two OpenBLAS pools and vcomp OpenMP, all 1 thread (verified) | **10 exact_match** |
| `macos-latest` | CPython 3.12.10 (Clang 13) | macOS 26.7.2 arm64, 3 logical CPUs | Accelerate | libomp 1 thread; Accelerate unverified | **10 exact_match** |

- Every scenario on both runners: canonical result **byte-identical** to the reference (0 differing fields out of 1 to 57 float fields per scenario,
  max ulp distance 0, 0 gating failures); both K-means scenarios had the same partition and the same iteration count (8 and 2).
  Because nothing differed, the numeric tolerances of `axiom.agreement/1` were not exercised by this experiment.
- Comparison mode was `cross_environment_numeric` for every scenario (the environments differ), so byte identity is reported
  as an observation, and agreement under the criteria follows trivially.
- Environment fields recorded as different: Windows: Python version and compiler, OS system/release/machine, NumPy backend,
  commit, `uv.lock` digest and engine file digest; macOS runner: Python version and compiler, OS release, commit.
- **Finding (limitation of ADR 0007):** on Windows the `uv.lock` digest and the engine file digest differ from the reference
  although the tested commit contains the same files; on the macOS runner they are equal. The most likely cause is line-ending
  conversion of text files at checkout on Windows (Git `core.autocrlf`); this was **not verified**. Consequence: these digests
  identify checked-out bytes, not repository content, so identical code can appear as a different environment (a conservative
  error: it can only switch a replay to cross-environment mode, never make different code look identical).

## Predictions versus outcome
- Inspect and Compare byte-identical on all runners: held. Relate within criteria: held (byte-identical). Cluster same partition:
  held (byte-identical, same iteration counts).

## Limitations and failures
- One CI run per runner, ten scenarios on two synthetic datasets, one dataset scale (up to 5,000 rows). This shows agreement for
  these inputs on these runners only; it is not a general portability guarantee, not packaged Windows support, not Windows ARM64
  and not independent human replication.
- Byte identity across platforms is an observation here, not a promise; the criteria remain the contract for other inputs.
- The step is report-only; its green status alone says nothing about agreement.

## Notes added after technical review (2026-11-25; results unchanged)

- **Commit versus content.** The reference passports were written at `f163b41`; CI tested the merge commit `ef0b84b`
  (PR head `031360b` merged into `main`). `git diff f163b41 031360b -- src uv.lock pyproject.toml` is empty, so the
  engine source and lock file content were the same; the commit field differs by construction, which alone makes every
  comparison cross-environment.
- **Windows digests: cause confirmed in EXP-006.** The hypothesis above was tested in
  [EXP-006](../EXP-006-review-confirmation/MANIFEST.md) part D: the Windows runner's Git uses `core.autocrlf=true`, and
  every hashed file differed from its committed blob only by LF -> CRLF conversion. `.gitattributes` now pins LF for
  `*.py` and `uv.lock`.
- The comparison in this experiment treated unknown environment fields as differences (the code under test at the time).
  The revised classification (ADR 0008) would also have reported these environments as `different` (Python version,
  OS and commit differ), so the verdicts are unaffected.
