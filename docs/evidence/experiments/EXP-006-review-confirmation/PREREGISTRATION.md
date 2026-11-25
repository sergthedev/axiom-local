# EXP-006 pre-registration: confirmation of the PR #7 review corrections

- Issue: #3; [technical review](https://github.com/sergthedev/axiom-local/pull/7#issuecomment-7171558913) of head
  `3172cdc` (blockers B1, B2, B3 and the Windows checkout-digest finding).
- Written and committed **before** any of the runs below. A one-replay smoke run of
  `benchmarks/issue3_confirmation.py` on the uncommitted code was made to check that the script executes; its output was
  not kept and is not part of the results. The reviewer's static counterexamples were reproduced on `3172cdc` before
  any change (development log); that reproduction is not part of this experiment.
- Code under test: the commit that contains this file (recorded by the scripts as `git.commit`). If part D below
  leads to a `.gitattributes` change, that later commit changes no engine file; the engine file digest recorded by
  every run identifies the engine code.
- EXP-004 and EXP-005 are not rerun or edited; their results stand as recorded.

## Part R: fresh-process replay on the revised code (development Mac, one run)

Command (repository root, clean tree, after `uv sync --locked`):
`uv run python benchmarks/issue3_confirmation.py --runs 20 > docs/evidence/experiments/EXP-006-review-confirmation/results.json`

Same ten scenarios as EXP-004 (`benchmarks/issue3_common.py`).

| ID | Criterion |
|----|-----------|
| R1 | All 10 x 20 = 200 replays `exact_match`; per scenario one distinct result hash, equal to the reference passport's and to the in-process hash. |
| R2 | Each of the 200 runs stores its own run id; the 200 run ids are distinct (and distinct from the 10 reference run ids). |
| R3 | Each run stores its own worker observations: the thread variables (all `"1"`), every detected pool with `num_threads` 1, the thread-limit probe, the BLAS visibility and a derived native-thread status; plus its own environment status. |
| R4 | The replay is judged by byte identity in every run (comparison mode `same_environment_exact` or `exact_environment_unverified`, never `cross_environment_numeric`), since the worker runs the same checkout and interpreter. |
| M / X1 | The EXP-004 mutation (M-stale, M1-M4) and malformed-input (X1) criteria, unchanged, using the unchanged EXP-004 functions. |
| A1 | A re-sealed Inspect passport recording `max_bytes` 1000 times the default is rejected with `limits_exceed_policy`, and **no** worker process is started. |
| A2 | A re-sealed Inspect passport whose mean is moved by one ulp and whose `packages.numpy` is replaced by an unavailable marker: environment status `unverified`, comparison mode `exact_environment_unverified`, verdict `numerical_discrepancy` (the one-ulp difference is within the cross-environment criteria but must not be accepted when no environment difference was observed). |
| A3 | A re-sealed Cluster passport with one assignment swapped between clusters 0 and 1 (sizes unchanged) passes every integrity check (digests are not signatures) and is exposed by recomputation: verdict `numerical_discrepancy`, `same_partition` false. |

Predictions (not criteria): on this Mac every run's native-thread status is `unverified` (NumPy uses Apple Accelerate,
which threadpoolctl cannot see) and the environment status is `unverified` for the same reason and because threadpoolctl
reports no libomp version; so the comparison mode is `exact_environment_unverified`.

## Part D: Windows checkout digests (GitHub Actions, report-only step)

Diagnostic: `benchmarks/checkout_line_endings.py` (CI step "Checkout line-ending diagnostic"), run on `macos-latest`
and `windows-latest`. For the engine `.py` files, `uv.lock` and the text fixtures it compares the Git blob at `HEAD`
with the checked-out bytes (SHA-256, CRLF and LF counts, Git attributes, `core.autocrlf`) and recomputes the two
manifest digests from the committed blobs.

- **D1 (CI run of the commit containing this file, no `.gitattributes` change).** Hypothesis H: on `windows-latest`
  the checked-out `.py` files and `uv.lock` differ from their blobs only by LF -> CRLF conversion
  (`identical_after_crlf_to_lf` true for every differing file, blobs without CRLF), and the digests of the committed
  blobs equal those reported on `macos-latest`; on `macos-latest` no hashed file differs from its blob.
  - H confirmed: add `.gitattributes` rules `*.py text eol=lf` and `uv.lock text eol=lf` (only the files whose bytes
    the manifest hashes; the CSV and NIST fixtures are already `-text`). Raw byte hashing is not changed.
  - H not confirmed: no line-ending change; the actual discrepancy is documented and the conservative
    "different environment" classification is kept.
- **D2 (CI run after the change, only if H is confirmed).** On both runners no hashed file differs from its blob, and
  `package_files_sha256` and `lock_sha256` recorded by the manifest are equal on the two runners and equal to the
  digests of the committed blobs.

## Part X: cross-platform observation on the revised code (report only)

The existing CI step compares the revised code on both runners with the **EXP-005 reference passports** (written at
`f163b41`; not regenerated). The report must say `"complete": true` with 10 compared scenarios on each runner;
verdicts are recorded as observed. Prediction: 10 `exact_match` per runner, environment status `different` (the commit
and engine files differ from the reference by construction). This is an observation, not a correctness test.

## Not in scope

- The network-disabled rehearsal and the 100-run golden release target remain release gates (not run here).
- No tolerance, criterion or scenario is changed after seeing results; failures are recorded, not retried away.
