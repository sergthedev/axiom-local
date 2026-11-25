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
