# Development log

Chronological, factual record. Failures and tradeoffs are included. Paths are sanitized.

## 2026-11-25 — Issue #1: bootstrap

### Starting state
- The remote repository had a single branch, `idk`, with one seed commit (`.gitignore`, `LICENSE`, `README.md`).
  There was no `main`. `main` was created from that seed commit and the foundation commit added on top
  (no force push; `idk` was left untouched).
- The seed `LICENSE` named a former GitHub username as copyright holder. The owner confirmed the current
  handle (`sergthedev`, same account) and asked for it to be updated; done in the foundation commit.

### Environment findings (failures/blockers encountered)
- The Mac initially had only system Python 3.9.6 (below the required 3.11/3.12), no `uv`, and no GitHub CLI.
  With the owner's explicit approval, `brew install python@3.12 uv gh` was run (Homebrew).
- Installed versions: Python 3.12.16, uv 0.13.10, GitHub CLI 2.105.0.

### Governance addendum and tradeoffs (approved 2026-11-25)
- Evidence-first files (REFERENCES, CLAIMS, ADRs, research register, experiment template) were created before any
  engine code. Tradeoff: more up-front documentation for a placeholder package; accepted to fix the habits early.
- Only two repository skills were added (`axiom-source`, `axiom-pr`); no third-party skills, agents or MCP servers.
  The skill format was checked against the official documentation (REF-001); part of that page was read through a
  summarized fetch rather than verbatim.
- Model/effort identifiers are not recorded in repository files by the implementing tool's policy; see
  `AI_CONTRIBUTIONS.md`.
- Runtime has zero third-party dependencies; dev tools pinned exactly in `pyproject.toml` and locked in `uv.lock`
  (tool and package versions chosen from PyPI metadata, REF-004).

### Local verification run (measured, single run)
- Hardware/OS: Apple Silicon (`arm64`), macOS 27.0.1.
- Python 3.12.16 (Homebrew), uv 0.13.10. Fresh virtual environment (previous `.venv` removed).

| Command | Exit status | Result |
|---------|-------------|--------|
| `uv sync --locked --python 3.12` | 0 | 7 packages installed (axiom-local, iniconfig 2.3.1, packaging 26.4, pluggy 1.7.0, pygments 2.21.0, pytest 9.2.0, ruff 0.17.6) |
| `uv run pytest` | 0 | 1 passed |
| `uv run ruff check .` | 0 | All checks passed |
| `uv run ruff format --check .` | 0 | Files already formatted |

Limitations: one machine, one run, not independently verified; Windows and CI results are reported on the issue
after the workflow actually runs, not asserted here. The single smoke test proves only that the package imports.

## 2026-11-25 — Issue #2: deterministic statistical core

### Scope and starting state
- Authorized by the owner and the technical lead ([decision](https://github.com/sergthedev/axiom-local/issues/2#issuecomment-7167523230),
  updated Issue #3 requirements read for compatibility only; nothing from Issue #3 was implemented).
- Starting point: `main` at `1d3b646`; branch `feat/issue-2-deterministic-core`; one pull request, sequential focused commits.
- Not implemented on purpose: UI, language model, training, replay/run passport (Issue #3), correlation matrix and outlier
  flags (deferred P1), p-values, confidence intervals.

### Issue #1 review follow-ups applied
- `C-003`, this log and ADR 0001 now cite CI run 45644347806 for commit `1d3b646`. That run was inspected independently
  by the technical lead (reviewer-observed); the earlier local run on the Mac remains implementer-reported.
- The technical lead's review is recorded in `AI_CONTRIBUTIONS.md` as an AI review, not a human check; the Issue #1
  configuration report is linked there instead of writing a model identifier.
- Python 3.11 support was an untested claim; it is removed (the stack requires Python 3.12, ADR 0002).
- GitHub Actions are pinned to commit SHAs (`actions/checkout` and `actions/setup-python`, both tag v7, resolved with the
  GitHub API); `uv` is pinned by version in the workflow. Build-tool pinning beyond that remains future hardening.
- Branch protection could not be read through the integration (HTTP 403 in the lead's review). It was not checked or
  changed here; the owner should verify it in repository settings.

### Environment
- Apple M5 (arm64), 16 GiB, macOS 27.0.1, CPython 3.12.16 (Homebrew), uv 0.13.10.
- Locked runtime stack: numpy 2.5.5, pydantic 2.14.4, scikit-learn 1.9.1, threadpoolctl 3.8.0 (transitive scipy 1.18.1, joblib 1.6.0
  and others in `uv.lock`). cp312 wheels for macOS arm64 and Windows amd64 exist per PyPI metadata (REF-019).
- NIST data and certified-value pages were downloaded with `curl` and cross-checked; license basis in `THIRD_PARTY_NOTICES.md`.

### Implementation order and commits
1. Dependencies, input guard, canonical JSON, typed plans and binding (ADRs 0002, 0003, 0005).
2. Inspect with NIST and exact-oracle validation (ADR 0004).
3. Relate, then Compare, then Cluster (ADRs 0004 and 0006), each with its own tests.
4. Engine-level tests and a fully synthetic economics example; evidence scripts; documentation.

### Failures, findings and deviations (recorded, not hidden)
- **Quartile accuracy criterion failed once.** First criterion: quartile within 2 ulp of the result. On 300 seeded random datasets the
  worst error was 4.0 ulp. Analysis of the worst case showed cancellation in `a + t*(b-a)` (0.125 ulp at the operand scale; all
  600 quartiles within 0.5 operand-scale ulp). The criterion was replaced by an operand-scaled one. This revision was made after seeing
  the failure; it is derived from the formula, not tuned to a passing value, but it is post hoc and is flagged as such in ADR 0004 and EXP-001.
- **Standard-library `statistics.correlation` is wrong at extreme scales** (3.12.16): `0.0` for data near 1e200, a false "constant"
  error near 1e-200. A range guard (exact power-of-two scaling) was added; SciPy was not added as a direct dependency (ADR 0004).
- **Default multi-threaded K-means was not bit-reproducible** in a 20-fit experiment (7 distinct outputs in the first run, 9 in the final run); single-threaded fits gave 1 both times.
  The engine fits single-threaded (ADR 0006). One machine only.
- **The near-constant Pearson rule is conservative** against algorithmic error (EXP-001, section 5); kept because it is justified by input
  representation uncertainty, and documented as a limitation.
- **Test-authoring mistakes caught by the tests:** a hand-calculated group variance in a Compare test was first written as 13 instead of
  7 ((4+1+9)/2); the test, not the engine, was wrong. A malformed schema in a record-width test was also caught.
- `ruff format` rewrote the README code example (the installed ruff formats Markdown code blocks).
- The first evidence run was made at commit `eb111b7` with an unrelated dirty tree; the evidence scripts were re-run on a clean tree at the
  final code commit and those results are the ones recorded.

### Local verification run (measured, one run)
| Command | Exit status | Result |
|---------|-------------|--------|
| `uv sync --locked` | 0 | environment from `uv.lock` |
| `uv run pytest -q` | 0 | 206 passed (about 17-19 s) |
| `uv run ruff check .` | 0 | All checks passed |
| `uv run ruff format --check .` | 0 | formatted |
| `uv run python benchmarks/issue2_candidate_smoke.py` | 0 | EXP-001 section 4 |
| `uv run python benchmarks/issue2_evidence.py` | 0 | EXP-001 sections 1-3, 5-8 |

Limitations: one machine; the author of the engine also wrote the tests and oracles; no independent reviewer has reproduced
these results; Windows numerics are not measured (see `EXP-001`).

### CI for the pull request
- GitHub Actions run [45656746267](https://github.com/sergthedev/axiom-local/actions/runs/45656746267) on PR commit `1a8c63d`: `test (macos-latest)` passed (14 s) and `test (windows-latest)` passed (43 s). Observed by the implementer; the workflow runs `uv sync --locked`, ruff check/format check and pytest only. This is not a numerical comparison between platforms.

## 2026-11-25 — Issue #2, PR #6: technical review response (four merge blockers)

Review: [technical lead review of head `1a8c63d`](https://github.com/sergthedev/axiom-local/pull/6#issuecomment-7168446160) (an AI-assisted static review; not independent human
sign-off). Addressed on the same branch with focused commits; no history was rewritten, nothing was merged, Issue #3 was not started.

### CI observation record (reviewer-observed, distinct from local runs)
- The technical lead verified run [45656746267](https://github.com/sergthedev/axiom-local/actions/runs/45656746267) directly: both jobs succeeded, 206 tests passed on
  Windows x64 (CPython 3.12.10, 9.44 s) and on macOS arm64 (CPython 3.12.10, 6.36 s); checkout merge SHA `61f14f6588d9ce4705fa20817ec2bf76c7378d4b`; the head
  commit was `1a8c63d24aa8681d7512241a4e949392596b3a81` and the diff from the final code commit `f92117a` was documentation/evidence only.
- The implementer's local runs (Apple M5, macOS 27.0.1, CPython 3.12.16, locked environment) are separate and implementer-reported. CI and local runs used different
  CPython patch versions. Neither establishes numerical agreement between platforms (not claimed).

### Blocker 1: quartile and IQR extreme-range correctness
- Finding: the first quartile implementation (NumPy `linear`) reported IQR overflow for `[-M, M]`; the exact answer is representable (q1 = -M/2, q3 = M/2, IQR = M). The test I
  had written asserted the overflow, so it encoded the defect. The test was replaced, not the evidence: EXP-001 and its recorded quartile failure are untouched.
- Fix: median and type-7 quartiles are exact rationals rounded once; true overflow (exact value beyond binary64) and exact-nonzero underflow are distinguished and reported as null
  with a reason for every statistic (ADR 0004, amendment 1). New regressions: `[-M, M]` known answers, `[-M, -M, M, M]` true IQR overflow, cancellation case, subnormal cases,
  mean/median/quantile underflow policy, NumPy as cross-check only.

### Blocker 2: confirmation set for the post-hoc quartile revision
- EXP-002 was pre-registered (criteria, seeds, families, denominators, oracle formulation, derived NumPy bound) and committed before any data were generated, then run on a clean tree.
- Results: production 0 value and 0 classification mismatches in 8,000 evaluations. NumPy: the EXP-001 revised criterion was **falsified** on held-back data (1.125 operand-scale ulp), the
  newly *derived* first-order bound (4 operand-scale ulp) held, the original result-ulp criterion failed again (up to 1,048,576 ulp under cancellation), and 69 representable results were non-finite.
  The EXP-001 "<= 1 operand ulp" bound was not derived tightly; this is recorded rather than adjusted.

### Blocker 3: validated-plan boundary and resource guards
- Revalidation from field values at `bind_plan` and `execute`; limits supplied at execution are re-checked; bounded plan JSON (bytes, nesting, value count) with structured errors for
  pathological input; BoundPlan documented as data, not a signature; result bytes fixed at construction (`core` is a copy). Tests in `tests/test_plan_boundary.py` (ADR 0005, amendment 1).

### Blocker 4: K-means resource and verification contract
- v1 solver ranges and a joint workload cap checked before any fit; solver output validation; warnings and FloatingPoint exceptions guarded across scaler, fit and checks; exact power-of-two
  feature scaling so tiny/huge magnitudes are valid; `centroid_fixed_point` renamed `centroid_consistency_approx` with a separate Lloyd label fixed-point diagnostic; EXP-003 measured throughput for the cap
  (ADR 0006, amendment 1). The in-process repeat test was renamed `test_twenty_in_process_repeats_are_byte_identical` (subprocess replay evidence is EXP-001 section 8 and `test_fresh_processes_...`).
- Behavior change to note: a feature with magnitude above 2^500 was previously rejected (`unsafe_scale`); it is now valid because of the exact scaling.

### Other review items
- `REFERENCES.md` REF-009 and ADR/notice text updated so they no longer say production quartiles use NumPy; NIST attribution and EXP-001 are unchanged.
- ADRs 0002-0006 remain Proposed until the owner decides.
- Branch protection: the review observed `protected=false` and no rulesets. Not changed from this branch; the owner must verify/enable suitable protection and status checks before merge.
- The PR header contains the automated project-thread attribution link required by this workflow; whether to keep the public link is the owner's decision. No history was rewritten.

### Local verification after the fixes (measured, one run; Apple M5, macOS 27.0.1, CPython 3.12.16, locked environment)
| Command | Exit status | Result |
|---------|-------------|--------|
| `uv sync --locked` | 0 | environment from `uv.lock` |
| `uv run pytest -q` | 0 | 259 passed (about 17 s), including the NIST and example tests on the new code |
| `uv run ruff check .` | 0 | All checks passed |
| `uv run ruff format --check .` | 0 | 75 files already formatted |
| `uv run python benchmarks/issue2_quartile_confirmation.py` | 0 | EXP-002 |
| `uv run python benchmarks/issue2_cluster_workload.py` | 0 | EXP-003 |

EXP-001 describes the code at commit `f92117a` (before these fixes) and is deliberately not re-run or edited; the NIST comparisons are re-asserted by `tests/test_nist.py` on the current code.
The pull request CI for the fixed head follows.

### CI for the fixed head (implementer-observed)
- GitHub Actions run [45661119280](https://github.com/sergthedev/axiom-local/actions/runs/45661119280) on head `972d733`: `test (macos-latest)` passed (259 tests in 12.11 s, CPython 3.12.10, darwin) and
  `test (windows-latest)` passed (259 tests in 9.43 s, CPython 3.12.10, win32). Locked install, lint, format check and pytest only; observed by the implementer from the run logs, not yet re-verified by the reviewer.
  The head after this entry is a documentation-only commit with its own CI run.

## 2026-11-25 — Issue #2, PR #6: final technical re-review (original-unit numeric range)

Review: [technical re-review of head `cc70cfa`](https://github.com/sergthedev/axiom-local/pull/6#issuecomment-7169191502) (AI-assisted static re-review; blockers 1-3 accepted, one narrow issue left in blocker 4).

- **Reproduced end to end** (before any change): `x = [0, 5e-324, 0, 5e-324]`, k = 2 returned `scaler.scale = [0.0]` for a varying feature. Neighbouring probe `x = [0, 5e-324, 1e-323, 0, 5e-324, 1e-323]` showed a second defect of the same kind: a cluster mean reported as `0.0` for the exact value 2.5e-324 (a false zero). Max-range and mixed-sign subnormal inputs with representable results were fine.
- **Cause:** the inverse power-of-two conversion of the scaler and the original-unit cluster means ran outside the guarded block and without an exactness check; pre-scaling could also flush tiny entries (the previous test even allowed k=2 on such data).
- **Fix (conservative rejection, no schema change; ADR 0006 amendment 2):** exact round-trip required for original-unit scaler statistics and, via the shared Inspect mean rule, for cluster means (`original_units_not_representable`); lossless pre-scaling required (`prescaling_underflow`). The earlier permissive collapse policy is revised and documented.
- **Regression tests:** the reported counterexample, a scaler that would round in original units, a cluster mean that underflows (found by exhaustive search of small subnormal fixtures), lossy pre-scaling (named features), valid cases (mixed-sign subnormals, max range, wide lossless ranges) and a non-finite conversion. Five of the new tests were confirmed to fail on the previous source.
- **Wording corrected:** W (rows x k x features x n_init x max_iter) is a conservative iteration-work proxy, not a measured count of distance evaluations and not a wall-time guarantee; the "30-35 s" statement in ADR 0006 and the EXP-003 manifest was withdrawn (the EXP-003 table and `results.json` are unchanged; the correction is marked in the manifest).
- Preserved: EXP-001, EXP-002 and the EXP-003 data. No Issue #3 work, no merge.

### Local verification (measured, one run; Apple M5, macOS 27.0.1, CPython 3.12.16, locked environment)
| Command | Exit status | Result |
|---------|-------------|--------|
| `uv sync --locked` | 0 | environment from `uv.lock` |
| `uv run pytest -q` | 0 | 266 passed (about 14-17 s) |
| `uv run ruff check .` | 0 | All checks passed |
| `uv run ruff format --check .` | 0 | 75 files already formatted |

No experiment scripts were re-run: the change affects reporting edge cases only, and EXP-001/002/003 data are preserved unchanged. CI for the final head follows once observed.

### CI for the numeric-range fix (implementer-observed)
- Run [45664213878](https://github.com/sergthedev/axiom-local/actions/runs/45664213878) on head `afaabf8`: `test (macos-latest)` passed (266 tests in 16.96 s) and `test (windows-latest)` passed (266 tests in 10.57 s),
  CPython 3.12.x on both runners; locked install, lint, format check and pytest only. Observed from the run logs by the implementer, not yet re-verified by the reviewer; the entry's own commit is documentation-only with its own CI run.

## 2026-11-25 — Issue #3: verification engine (Analysis Passport and replay)

### Starting state and authorization
- PR #6 (Issue #2) was accepted by the technical lead at head `0b52b44c1532ea90f69fe17f902817da32f5f7c4`
  ([final acceptance](https://github.com/sergthedev/axiom-local/pull/6#issuecomment-7169639125)) and merged by the owner as
  `323620d0d8fe6a989a1d9c2cb18e549fd565aa83`. Reviewer-observed CI: run
  [45665324679](https://github.com/sergthedev/axiom-local/actions/runs/45665324679) on merge ref `12b6d69`, macOS job 139385821828
  (266 passed, 13.84 s) and Windows job 139385822124 (266 passed, 9.38 s), CPython 3.12.10, lint and format passed; the lead also
  reported `main` as protected with both required checks. Issue #2 is closed. (Recorded here as reported by the lead; not
  re-inspected by the implementer.)
- The owner authorized Issue #3 in the project thread. Branch `feat/issue-3-verification-engine` from `323620d`.
- Read: Issue #3 body, [DR-001 note](https://github.com/sergthedev/axiom-local/issues/3#issuecomment-7166468157),
  [DR-001 + DR-002 decision](https://github.com/sergthedev/axiom-local/issues/3#issuecomment-7167524679), the PR #6 acceptance,
  and the [Issue #5 research positioning](https://github.com/sergthedev/axiom-local/issues/5#issuecomment-7170027058).
- Not done on purpose: AX-01, AXBENCH, training, Studio UI, regression, forecasting, PROV/RO-Crate, offline (network-disabled)
  rehearsal, Windows packaging. No Core contract change: the four fingerprints, `execute`, `AnalysisResult` and
  `environment_info()` are reused unchanged.

### Implementation (commit `26eb769`)
- `environment` (manifest `axiom.environment/1`), `passport` (`axiom.passport/1`), `_worker` + `replay` (fresh-process analysis,
  replay, classification, passport comparison), `agreement` (criteria `axiom.agreement/1`), exact rational ARI in `partition`,
  CLI `python -m axiom env|analyze|replay|compare`, example profile JSON. ADRs 0007 and 0008.
- Defects found while testing and fixed before commit: (1) `build_passport` aliased the plan object between `inputs` and the result
  core, so a test edit of one changed both; fixed with independent copies. (2) The replay summary first reported `plan_identity: pass`
  when the engine had rejected the data before reaching the plan check; it is now `not_run` when not reached.
- Observation: on the development Mac NumPy uses Apple Accelerate, which threadpoolctl does not list; the manifest reports BLAS
  thread control as unverified instead of single-threaded.

### Pre-registration (commit `f163b41`) and experiments
- EXP-004 and EXP-005 pre-registrations, ADR 0008 criteria and the scripts were committed before the measured runs. A one-replay
  smoke run of the EXP-004 script preceded the pre-registration (stated in it); its output was not kept.
- **EXP-004** at `f163b41`: 200 of 200 fresh-process replays byte-identical (ten plans x 20), every mutation and malformed-input
  criterion met; 0 failures. See the manifest.
- **EXP-005**: reference passports written at `f163b41` on the Mac (their result hashes equal the EXP-004 references); a local
  self-comparison against them gave 10 of 10 `exact_match` in the same environment (a sanity check, not part of EXP-005). The
  cross-platform result comes from the pull request CI and is recorded in the EXP-005 manifest.
- The new CI step is report-only: it prints the comparison and exits non-zero only if a scenario cannot be run.

### Documentation carry-forward from the Issue #2 acceptance
- ADR 0006 amendment 3: representability rules are a conservative reporting policy, not exact statistics; a power-of-two round trip
  does not make downstream floating-point operations exact. ADR statuses remain Proposed until owner acceptance.
- The initial PR #6 summary mentioned NumPy quartiles and unverified branch protection; both are outdated (quartiles are exact
  rationals since amendment 1 of ADR 0004; the lead later observed `main` protected). The merged PR text was not edited.
- The README tagline ("powered by a specialized language model") may read as if a trained model exists. The lead proposed
  owner-approved wording in the Issue #5 comment; it was **not** changed here because it needs the owner's decision.
- Experiments index now lists EXP-002 and EXP-003, which were missing.

### Local verification (measured, one run; Apple M5, macOS 27.0.1, CPython 3.12.16, locked environment)
| Command | Exit status | Result |
|---------|-------------|--------|
| `uv sync --locked` | 0 | environment from `uv.lock` (unchanged; no new dependency) |
| `uv run pytest -q` | 0 | 327 passed in about 40 s (266 existing + 61 new; the fresh-process tests start about 30 interpreters) |
| `uv run ruff check .` | 0 | All checks passed |
| `uv run ruff format --check .` | 0 | 95 files already formatted |
| `uv run python benchmarks/issue3_replay_benchmark.py --runs 20` | 0 | EXP-004 |
| `uv run python benchmarks/issue3_cross_platform.py --write-reference ...` | 0 | EXP-005 reference |

### First PR CI run (implementer-observed)
- Run [45675071394](https://github.com/sergthedev/axiom-local/actions/runs/45675071394) on head `8b4d84b`: `test (windows-latest)` **failed**: 326 passed, 1 failed
  (`test_compare_passports`). Cause: a **test defect**: the test simulated "another environment" by setting `os.system` to `"Windows"`,
  which is the real value on the Windows runner, so the environments were equal and the comparison mode was correctly
  `same_environment_exact`. The test now uses a value that differs on every runner; the engine was not changed. Because the
  tests failed, the EXP-005 step did not run on that job.
- `test (macos-latest)` was **cancelled** without being assigned a runner (no steps ran; started 18:25:52Z, cancelled 18:40:54Z).
  Not caused by the change; recorded as an infrastructure event.

### Second PR CI run and EXP-005 (implementer-observed)
- Run [45677769454](https://github.com/sergthedev/axiom-local/actions/runs/45677769454) on head `031360b` (merge ref `ef0b84b`): `test (windows-latest)` passed
  (327 tests in 48.31 s, CPython 3.12.10, win32) and `test (macos-latest)` passed (327 tests in 81.48 s, CPython 3.12.10, darwin); lint and format
  passed on both (95 files).
- EXP-005 step: **10 of 10 scenarios byte-identical** to the Mac reference on both runners. Finding: on Windows the `uv.lock` and engine-file digests
  differ from the reference (most likely CRLF conversion at checkout; not verified), so the manifest reports a different environment for identical
  code. Recorded as a limitation of ADR 0007; not changed in this PR, to keep the measured code fixed after the experiment.
- CI run [45679448835](https://github.com/sergthedev/axiom-local/actions/runs/45679448835) on the evidence-only head `3172cdc` also
  succeeded on both runners (observed after the review was posted).

## 2026-11-25 — Issue #3, PR #7: technical review response (B1-B3, Windows digests)

### Review and reproduction
- [Technical review](https://github.com/sergthedev/axiom-local/pull/7#issuecomment-7171558913) of head `3172cdc`: changes
  required (B1 resource policy, B2 strict structure and structured errors, B3 environment classification and evidence
  precision; Windows checkout digests). Static counterexamples, not executed by the reviewer.
- Reproduced on `3172cdc` before any change, with a scratch script (not committed) using the test helpers:
  - B1: a re-sealed passport with `max_bytes` x1000 and `max_cluster_work` 1e30 passed all integrity checks, was
    dispatched to a worker with `max_bytes` 33,554,432,000 and replayed as `exact_match`; `timeout=inf` crashed with
    an unstructured `OverflowError`.
  - B2: a re-sealed passport with an invented verification entry, `dataset_format` `axiom.dataset/999`, a non-hex
    `raw_sha256` and `n_rows` -5 loaded and passed all five integrity checks; two identical passports with result block
    `{"nonsense": 1}` compared as `exact_match`; `1e999` parsed to `inf`; a passport with `result: []` raised
    `AttributeError` in `replay`; opposite largest floats gave `abs_diff = inf` and a report that `allow_nan=False`
    could not serialize.
  - B3: equal `unavailable` package markers plus two `unverified` probes gave no environment difference; pools at 8
    threads versus 1 gave no difference; Accelerate reported `unverified` while the probe read `verified_single_thread`.
  - All counterexamples were confirmed.

### Corrections (commit `fd61eb6`)
- B1: `replay(..., policy=DEFAULT_LIMITS)` rejects recorded limits above the trusted policy (`limits_exceed_policy`)
  before any worker; runs in [1, 100], timeout finite in (0, 3600] s, `Limits` validated; passport text bounded at
  64 MiB before parsing (`VerifyError` / `PassportError`, CLI exit 2).
- B2: new `src/axiom/passport_schema.py` (full nested validation, depth, finiteness, vocabulary, per-operation result
  shape); `integrity_checks` adds `result_consistency` and `verification_record` and checks versions, row count and
  operation/plan agreement; `abs_diff` is `null` beyond the binary64 range.
- B3: `compare_environments` (verified_same / unverified / different; nested unknowns, `None`/`"unknown"` versions and
  `unverified` statuses are unknown; thread variables, pools, probe and BLAS visibility compared);
  `exact_environment_unverified` mode keeps byte identity; per-run run id, environment status and worker thread
  evidence; `native_thread_policy` check; partial runs never pass; the CI comparison script reports completeness and
  exits 1 if any scenario was not compared.
- Tests: `tests/test_review_regressions.py` (55 tests), four existing tests adapted to the new semantics (environment
  modes; one forged-plan test split into an integrity failure and a consistent-forgery test). The Core contract and
  `axiom.agreement/1` are unchanged.
- Local: `uv sync --locked`; `uv run pytest -q` 384 passed (68 s); ruff check and format clean (100 files).

### EXP-006 (pre-registered in `fd61eb6`)
- Part R (local, clean `fd61eb6`): 200 of 200 `exact_match`, 200 distinct stored run ids, per-worker evidence; mutations,
  malformed input and A1-A3 met. One smoke run of the uncommitted script preceded the pre-registration (stated there).
- CI run [45683518798](https://github.com/sergthedev/axiom-local/actions/runs/45683518798) (merge `6074227`): 384 passed on
  Windows (77.88 s) and macOS (90.12 s). D1 confirmed the CRLF hypothesis (`core.autocrlf=true` on Windows; 28 files
  differing only by LF -> CRLF; blob digests equal to macOS). Part X: 10 of 10 `exact_match` on both runners against the
  EXP-005 reference, reports complete.
- `.gitattributes`: `*.py text eol=lf`, `uv.lock text eol=lf` (pre-registered decision after D1). No blob changed
  (all were LF already).
- Historical evidence kept: EXP-004 and EXP-005 results unchanged; a correction note narrows the EXP-004 per-worker
  thread statement (C-025); EXP-005 notes the commit/content distinction and the confirmed cause.
- CI run [45685214061](https://github.com/sergthedev/axiom-local/actions/runs/45685214061) (head `8310baa`, merge `efa20ba`):
  384 passed on Windows (86.24 s) and macOS (103.28 s), lint and format passed. EXP-006 D2 met: no hashed file differs from
  its blob on either runner; engine digest `4b904d42…` and lock digest `0591cb58…` identical on both. Part X again 10 of 10
  `exact_match` per runner.

## 2026-11-25 — PR #7 final review: residual B1 and final audit (Issue #3)

Input: the [final technical review](https://github.com/sergthedev/axiom-local/pull/7#issuecomment-7172653455) of head
`e8a9f57` (accepts B3 and the Windows fix; B2 substantially addressed; one blocker: residual B1) and the owner's request
for a comprehensive final audit of AXIOM Verify.

### Reproduction on `e8a9f57` (before any change; safe scale only, no OOM-sized allocation)
- Residual B1: re-sealed example Cluster passports claiming `k` (result only) gave a peak traced allocation in
  `integrity_checks` of 0.02 / 3.99 / 39.99 / 79.99 MB for `k` = 10 / 10^5 / 10^6 / 2 x 10^6. With `max_k` also raised,
  `replay` reached 40.0 MB before returning `integrity_mismatch` (integrity ran before the policy) and
  `compare_passports` 40.0 MB; no worker was started. Confirmed.
- Other findings of the audit, each confirmed: `replay(timeout=10**400)` raised `OverflowError`; a passport whose text
  contains the escape `\ud800` loaded and then raised `UnicodeEncodeError` in `integrity_checks`; integers of 5000
  digits in API objects raised `ValueError` when hashed; a re-sealed passport with an added empty cluster passed all
  seven integrity checks and `compare_passports` raised `KeyError` while realigning labels; error messages quoted whole
  rejected values; with `PYTHONIOENCODING=cp1252` (the locale encoding of a redirected stdout on a Windows runner)
  `python -m axiom analyze` with a column named `温度` exited 2 with `UnicodeEncodeError`; status `ok` worker responses
  missing fields, worker errors given as text, and unreadable result text would raise in the parent; the worker's
  `git status` may refresh the repository index; `None` or NUL data paths raised `TypeError`/`ValueError`.
- Audited without a finding: the worker timeout kills the child and `subprocess.run` waits for it (now tested with a
  real worker); plan text is already bounded and rejects surrogates (Core); environment classification never treats
  unknown as equal; agreement tolerances are inclusive as documented (now tested at the boundaries); report sizes are
  linear in the documents.

### Corrections (commit `ab33266`)
- `passport.py`: cluster checks enumerate existing entries, `k` against the validated plan and recorded `max_k`, labels
  counted in a dictionary, features compared once per cluster against one set; result checks `not_run` when the plan
  or limits are invalid; empty clusters and false postconditions are failures; problem messages bounded; passport text
  must be str/bytes and valid Unicode.
- `passport_schema.py`: integers in the signed 64-bit range, no unpaired surrogates in strings or keys, quoted values
  and paths shortened to 64 characters.
- `replay.py`: trusted policy before integrity; overflow-free timeout check; data path validation; worker response
  shape check (`bad_output` / `crashed`); unreadable result text is a failed run.
- `agreement.py`: realignment only if every observed cluster id is a label; compact shape differences.
- `__main__.py`: UTF-8 bytes with LF to stdout and `--out`. `environment.py`: `git --no-optional-locks`.
- Tests: `tests/test_final_audit.py` (73 tests; 53 failed when run against the `e8a9f57` sources, the rest are
  boundary and termination tests that passed before too); one existing test changed (`10**30` is now outside the
  64-bit range, so the oversized-limits test uses `2**62` and asserts that `10**30` is rejected structurally).
- Local: `uv run pytest -q` 457 passed (47 s); ruff check and format clean (104 files).

### EXP-007 (pre-registered in `ab33266`, run once on the clean commit)
- S: 30 of 30 `exact_match`, all four families, distinct run ids, per-run evidence. B: peak traced allocation at most
  19,698 bytes for claimed `k` up to 2^63 - 1, no worker started, policy before integrity. P: parse memory linear; at
  most 25.2 times the text (empty objects); 1.6 GiB at the 64 MiB cap is an extrapolation and an open limitation.
- Negative result kept: the parse amplification is not reduced by this change.
- CI run [45692893944](https://github.com/sergthedev/axiom-local/actions/runs/45692893944) (head `c597e7a`, merge
  `9391082`): 457 passed on macOS (131.73 s) and Windows (79.97 s); lint and format passed (105 files); engine digest
  `b0e2c5e4…` and lock digest `0591cb58…` on both runners; no hashed file differs from its blob; cross-platform report
  complete with 10 of 10 `exact_match` on each runner. EXP-007 part C met.

## 2026-11-25/26 — Issue #4, Phase 3A: router contract, Guard, AXBENCH, rule baseline, model environment

### Authorization and starting state
- The owner authorized the start of Issue #4 and ratified ADRs 0007 and 0008 in a project-thread message on
  2026-11-25 (not a GitHub comment). Statuses updated in both ADRs and the ADR index; their text is otherwise
  unchanged.
- Baseline: accepted `main` `d7c4c8e882f9d45155a118b36dface2b2366ca94` (PR #7 merged). Branch
  `feat/issue-4-axbench-guard`. Read: Issue #4 with the DR-003, DR-004 and DR-005 reviews and the Verify acceptance
  (REF-031), `AGENTS.md`, ADRs 0001-0008, evidence files.

### Design choices (details in ADRs 0009-0012)
- New code lives in `src/axiom_router` and `src/axbench`, not in `axiom`, because the engine digest (ADR 0007) is
  computed over every `.py` file of `axiom`; Core identities and the Core contract are unchanged. A test checks that
  importing `axiom` loads neither package and that neither imports an ML library.
- The model environment is a separate uv project (`envs/model`) instead of a dependency group, so the Core `uv.lock`
  digest recorded in passports and the Windows CI do not change.

### Failures and corrections during development (kept for the record)
- Two `.DS_Store` files were committed in the first local commit; the unpushed commit was amended to remove them and
  `.DS_Store` was added to `.gitignore` (no pushed history was rewritten).
- Generator: column-free templates produced duplicates (fixed by a per-split (text, instance) key and global
  cross-split text uniqueness); mixed-language rendering was infeasible for some lineages (allowed cross-language
  generic mentions); hierarchical rounding gave the validation split only 2 REJECT examples of one reason (replaced by
  exact interleaved marginal allocation); a capitalization rule broke Turkish dotted/dotless i (removed).
- Rule router contamination: an audit found 81 rule cues that occurred only in validation or test templates, plus a
  few word-list entries; they were removed and the audit (`axbench.contamination`) is now a test that must return no
  finding. Rule development then continued on the train split only (final train result 400 of 400). The held-out
  template words were read by the audit; no router was scored on held-out examples before EXP-008.
- The Wilson expectation first written in a unit test (83/100) was wrong by hand; the test now uses SciPy
  `binomtest(...).proportion_ci(method="wilson")` as the reference. `audit` crashed on an empty split (`max()` of an
  empty sequence); fixed.

### EXP-008 (pre-registered in `409b46c`, run once on `857d4ee`)
- All criteria met: corpus bytes regenerate exactly; audit passed with zero cross-split components and maximum
  cross-split Jaccard 0.4833; no contamination finding; 160 of 160 valid rule outputs; 0 dispatches.
- Rule baseline on validation: 72 of 160 semantic exact match (45.0 %; cluster bootstrap 28.5-62.5 %); 0 of 21
  Compare requests correct (every validation Compare phrasing was unknown to the train-only cues); 26 wrong executable
  proposals, all held at confirmation. Train (development reference): 400 of 400.

### Model environment investigation (2026-11-26)
- Hugging Face API and pinned files of Qwen3-0.6B and Qwen3-1.7B fetched with `curl` into a scratch directory
  outside the repository (configs, tokenizer configuration, licence, model card; no weights): REF-032, REF-033.
- PyPI metadata of `mlx`, `mlx-lm`, `mlx-metal`: REF-034. Pins: mlx 0.32.4 and mlx-lm 0.32.2 (the newest mlx, 0.32.5,
  was one day old and was not chosen). The model lock uses `exclude-newer = 2026-11-12` (two weeks) except for the
  Core pins, which are reviewed in the main lock; the first lock without that cut-off had picked a one-day-old
  `regex` and `transformers` 5.24.1.

### First PR CI run (implementer-observed)
- Run [45704135434](https://github.com/sergthedev/axiom-local/actions/runs/45704135434) on `d47364b`: macOS passed;
  Windows failed 1 of 599 tests: `test_seal_writes_verifiable_test_outside_repo`. Cause: `seal` wrote `test.jsonl`
  in text mode, so Windows translated LF to CRLF and the file no longer matched the hash of the LF text in `SEAL.json`.
  A real defect that would have made a sealed test unverifiable on Windows. Fix: `test.jsonl`, `SEAL.json` and
  `PROVENANCE.json` are written as bytes.

### Model environment install and source reading (2026-11-26)
- `uv sync --locked` in `envs/model` took more than 20 minutes on the home connection (the 78 MB `mlx-metal` wheel was the
  slowest download); exit 0. Import probe: mlx 0.32.4, mlx-lm 0.32.2, transformers 5.22.2; `Device(gpu, 0)`, Metal
  available, a 3-element computation returned 14.0; `mlx_lm.models.qwen3` present.
- Reading the installed MLX-LM source (REF-035) found two template issues for the protocols: inference must pass
  `enable_thinking=False` explicitly, and `--mask-prompt` puts the empty think block into the trained completion.
  Both are recorded in ADR 0012 and EXP-010; neither has been measured with the tokenizer yet.
- CI run [45705966013](https://github.com/sergthedev/axiom-local/actions/runs/45705966013) on `e0c91bd`: macOS and
  Windows passed (tests, lint, format, report-only cross-platform step).

### PR #8 review response (2026-11-26)
- Review of head `51a6a3a` by the technical lead
  ([comment](https://github.com/sergthedev/axiom-local/pull/8#issuecomment-7175525079)). Each finding was reproduced
  on `51a6a3a` with a scratch script outside the repository before any fix. Observed: a candidate produced for one
  dataset and reviewed after `set_dataset` to another opened `awaiting_confirmation` with the new dataset's columns
  (`debt`, `score`); the same happened after a same-schema data change; a non-ASCII confirmation digest raised
  `TypeError` from `hmac.compare_digest`; a CLARIFY with the right reason and a nonexistent column id scored
  `exact_match` true with `semantic_valid` false; a duplicate prediction id silently replaced the first prediction
  and an unknown id was ignored; a REJECT class with support 1 and no predictions had F1 `None` and was left out of the
  macro F1.
- Fixes: Guard policy 2 with request contexts (ADR 0010 revision 2); scorer and harness 0.2; corpus versions,
  `freeze` and seal 0.2 (ADR 0011 revision 2). One existing Guard test assertion changed deliberately: an empty digest
  and an over-long token are now `malformed_confirmation` instead of `digest_mismatch` / `unknown_token`.
- During test writing: an evaluator flow-through assertion was too weak and relied on a key absent from CLARIFY
  predictions (KeyError); replaced by scoring the corrected and the original gold directly. An owner-lineage test
  template used the slot name `{features}` instead of `{feats}`; fixed in the test.
- The audit command now reads the committed files; its output is byte-identical to the committed `AUDIT.json`.
- EXP-011 (pre-registered in `5519973`, run once): the 160 EXP-008 predictions under scorer 0.2 gave identical
  metrics; the frozen rules reproduced all 160 raw outputs. EXP-008 files were only read.
- EXP-012 (exploratory, train only, run on `a2c50e1`): the Compare cue vocabulary of the rules is lineage-specific.
  The earlier log line on EXP-008 ("every validation Compare phrasing was unknown to the train-only cues") was a
  reading of validation results; EXP-012 gives the train-only mechanism and does not confirm it on validation.
- EXP-009 protocol revision 2 (still before any model prediction): safety stop, eligibility, 5-point margin with a
  paired cluster-bootstrap condition, 20 % informativeness floors and a default fallback to Qwen3-0.6B.
- The lead's AI pass over the Turkish batch is recorded as AI review only; no example was changed. No model was
  downloaded, loaded or run; nothing was trained; the final test was not generated, read or scored.

### Annotation decision and first frozen corpus (2026-11-26)
- PR #8 was merged by the owner (`ba05b15`) after the lead's technical acceptance of the infrastructure. The owner then
  delegated the annotation review to the technical lead AI; its decision
  ([comment](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7185793594)) asked for an explicit
  AI-review provenance path instead of `--human-review`. Work continues on a new branch from `ba05b15`.
- `freeze` 0.2 gained `--ai-review-authorization`; corrected requests now require reviewed `corrected_tags`. Existing
  tests were adapted to the manifest key `review` (previously `human_review`).
- Row 24 (`validation-0025`): Core Inspect returns only `n_total`, `n_valid` and `n_missing` for text columns, so the
  decision's fallback wording without category counts was chosen before the freeze. Row 9 (`train-0235`): the new
  mention `deneyim yılı` keeps its diacritics, so the `diacritics_stripped` tag was removed; the other tags of the
  three corrected examples still apply.
- The transcribed requests were compared programmatically with the decision text (exact matches; row 24 equals the
  first wording with the fallback clause substituted).
- Freeze passed; full audit (with the in-memory development test split) passed; a record-by-record comparison with the
  draft found content changes only in the three corrected examples, identical gold everywhere and
  `human_review: "none"` on all 560 records; the contamination audit returned no finding.
- EXP-013 (pre-registered in `b9f5d2e`, run once): rules 73 / 160 on the frozen validation split; only
  `validation-0037` changed outcome. No model was downloaded, loaded or run; nothing was trained; no test split was
  sealed or read. Sealing is left to the owner.

### Phase 3B: EXP-009 preparation (2026-11-26)
- The owner sealed the final test privately ([seal record](https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7186347486))
  and authorized downloading both pinned checkpoints and executing EXP-009 only. New branch from `6d887f9`.
- Repository, corpus hashes and the Core lock verified; `envs/model` reported outdated only because the editable
  `axiom-local` needed reinstalling after main moved; resynced with `uv sync --locked`.
- Both checkpoints provisioned with `provision.py` (0.6B in 1 min 16 s, 1.7B in 3 min 6 s); every published hash
  matched and an independent re-hash agreed.
- Wrote the runner (`envs/model/exp009_run.py`), the analysis (`benchmarks/issue4_exp009_analyze.py`), the selection
  rule as code (`axbench.selection`) and stop-rule bookkeeping (`axbench.runwatch`), with tests.
- Train-only smoke runs (3 train examples per model, battery power, harness check only): the untuned 0.6B wrapped
  its JSON in a code fence and produced non-contract slots on these train examples; the 1.7B decoded at about
  30 tokens/s on battery. These observations are not results and did not change any rule.
- Reading the installed MLX-LM showed that generation stops on the EOS set loaded from the checkpoint
  (`<|im_end|>` and `<|endoftext|>`), not on the end-of-turn token alone; recorded as operational definition D1.
- The protocol leaves 16 measurement details open (stop tokens, latency, warm-ups, sampling of swap and pressure,
  slowdown window, repeatability subset, run order); they are fixed in `PRERUN.md` before any validation prediction
  and submitted for review.

### Phase 3B: EXP-009 execution (2026-11-26)
- Before any validation prediction, the lead's pre-run review
  ([comment](https://github.com/sergthedev/axiom-local/pull/10#issuecomment-7186773178)) asked for five changes,
  all implemented in `995ae3f` with regression tests and PRERUN revision 1:
  - swap growth measured from one baseline per process taken before loading, with checks after load, every warm-up
    and every request;
  - mains power and normal pressure required before every process, and no battery override;
  - `MemoryError` stops the run whatever its message;
  - eligibility judged over the main and all five fresh processes;
  - strict record validation and integrity checks before the selection rule.
- A train-only `check` of the full process frame showed that `getrusage` RSS does not track MLX's Metal memory: the
  1.7B had 1.09 GB RSS against a 3.44 GB MLX load peak. This is reported, not corrected, because the protocol names
  RSS.
- The owner restarted the Mac and closed applications. Preflight passed: mains power, 0 bytes swap, pressure level 1.
- Attempt 1 (08:49-09:50 UTC): the 0.6B main and fresh processes completed. The 1.7B main was killed after 598 of 600
  repeat records. No stop rule had fired and the Mac had not rebooted. The likely cause is that the background job
  ended with the Claude Code session that owned it. The attempt was archived with file hashes and an interruption
  record (`ATTEMPTS.md`).
- The lead approved one full 1.7B rerun under D19
  ([comment](https://github.com/sergthedev/axiom-local/pull/10#issuecomment-7189459615)). Attempt 2 ran detached
  (new session, `nohup`, `caffeinate -i`), from 10:32 to 11:22 UTC, with exit status 0 and all six processes
  completed.
- Raw outputs were committed unscored (`6d4da72`). The analysis then gained accepted and archived attempt inputs
  (`b2586da`, no scoring change) and ran once.
- Results: exact match 0.6B 0, 1.7B 12, rules 73 of 160; schema-valid 2 and 53; 0 dispatches. Qwen3-1.7B was selected
  by rule 5 (+31.9 pp schema validity, interval +20.7 to +44.9). The archived 1.7B attempt's validation outputs are
  byte-identical to attempt 2 on all 160 requests.
- The final test was not accessed. Nothing was trained.

## 2026-11-26 — Issue #4, Phase 3C: EXP-010 LoRA training preparation (nothing trained)

- Started from accepted main `fbd4adc` (PR #10 merged). Branch `feat/issue-4-exp010-lora-prerun`.
- Verified: clean tree, 698 Core tests, model environment versions (mlx 0.32.4, mlx-lm 0.32.2), the Qwen3-1.7B
  checkpoint (10 files re-hashed against pinned and recorded hashes), the frozen corpus manifest and split hashes.
- Exported train and validation outside the repository and recorded their hashes; a second export was
  byte-identical, and a new Core test re-exports and checks the hashes on every platform.
- Read the installed MLX-LM training path (REF-036) and measured it with the pinned tokenizer. Findings: micro-step
  counting, built-in evaluation between updates (it would report 24 and 199 updates as 25 and 200), an inclusive
  loss-mask end bound that trains one padding target, a chat mask boundary that trains the empty think block and the
  template newline (6 extra targets per example on all 560), a shared NumPy RNG between evaluation and data order,
  silent truncation.
- Measured token lengths (longest 529, so `max_seq_length` 1024 by the rule) and the trainable parameter count
  (458,752, lazy load, no weights evaluated).
- Wrote the harness (`envs/model/exp010_train.py`: tokens, params, train, reload, evaluate, selftest), the detached
  launcher and 30-second monitor (`envs/model/exp010_launch.py`), and the pure logic with tests
  (`axbench.trainwatch`, `axbench.pilotgates`, `tests/test_exp010_prerun.py`). The harness renders the completion
  boundary itself (`enable_thinking=False`), uses the installed loss with an exclusive end bound, evaluates and saves at
  update boundaries, protects the data-order RNG, and pins export hashes, `max_seq_length` and the parameter count.
- Harness self-test on a tiny random Qwen3-architecture model with synthetic tokens (no checkpoint, no corpus): all
  checks passed. The monitor's sampling and exit were checked against a short-lived process. The real `train`,
  `reload`, `evaluate` and the detached launcher were not run.
- Pre-registered in PRERUN D13/D14 before any adapter exists: the pilot criterion (EXP-009 rule 3 on exact match vs the
  untuned base) and the DR-005 deployment-quality gates as exact count rules, descriptive only on this corpus.
- No training, no adapter, no adapter inference, no final-test access. Stopped for technical review.

### Pre-run revision 1 (lead review of `ea1d343`, corrections A-D)

- The [lead's pre-run review](https://github.com/sergthedev/axiom-local/pull/11#issuecomment-7191912956) accepted
  decisions 1-11 with conditions and required four corrections before the smoke. No training setting, data, base
  checkpoint or endpoint changed.
- A: new `axbench.supervise` (common pre-load baseline and preflight, monitor-ready gate, supervision of the monitor
  and every step, fail-closed sampling, stale heartbeat, stop marker overriding success, SIGTERM then kill after a
  grace period, exit status written in every case); `exp010_launch.py` is now a thin wrapper and the smoke reload is
  a supervised step. 27 dummy-process tests (`tests/test_exp010_supervise.py`), including the detached path end to end.
- B: `reload` verifies the smoke adapter's provenance from the training records, runs exactly the 5 warm-up train
  examples with stop rules after load and every request, re-reads the persisted records and exits non-zero on any
  missing or error record. C: `train` sets `completed` only after the record, optimizer-step (`optimizer.step`), file,
  hash and tensor checks and a final resource sample; finalization errors never replace a stop. D: lineage and
  denominator checks, not-assessable semantics and independent dispatch reporting in `pilotgates`
  (`exp010.pilot-gates/2`); G12 MLX peak over load, warm-up and validation; EXP-009 comparator hashes pinned;
  execution authorization (D17) and the analysis integrity plan (D18) pre-registered.
- Confirmed in the MLX-LM source and by the self-test that the trainer's update timing excludes callback work
  (C-057).
- Self-test revision 1 on the tiny random model and synthetic fixtures: 15 of 15 checks passed
  (`prerun/harness_selftest_rev1.json`). Core tests: 827 passed locally on macOS before the commit (737 at revision 0).
- Not run: real training, the real `reload` and `evaluate`, the macOS sampler inside the supervised sequence. No
  validation predictions, no final-test access. Stopped for re-review and the owner's smoke decision.
- Re-review of `c4387a1`: Windows CI failed the two new comparator-hash tests (809 passed, 2 failed) because Git
  converted the comparator files to CRLF on checkout; fixed with a scoped `-text` rule for byte-hashed experiment
  evidence, Git blob IDs verified unchanged, no evidence or pinned hash touched. The lead also found a narrow
  supervisor race (a monitor ending cleanly as a step finished was accepted); the monitor's health is now checked
  before and after every step and any monitor exit before the sequence is done fails, with 7 new deterministic
  dummy regressions. Core tests: 835 passed locally on macOS.
