# ADR 0007: Analysis Passport v1 and environment capture

- Status: Proposed (awaiting technical review and owner approval). Revised after the
  [PR #7 review](https://github.com/sergthedev/axiom-local/pull/7#issuecomment-7171558913) (blockers B1-B3; see
  "Revision after technical review" below) and after the
  [final review](https://github.com/sergthedev/axiom-local/pull/7#issuecomment-7172653455) (residual B1 and final audit).
- Date: 2026-11-25
- Issue / PR: Issue #3 ([technical lead decision](https://github.com/sergthedev/axiom-local/issues/3#issuecomment-7167524679)); builds on ADR 0005

## Context

Issue #2 delivered four identities (raw bytes, typed dataset, bound plan, canonical result) and a minimal
`environment_info()`. Issue #3 requires a versioned, machine-readable record that lets a result be recomputed and
judged later: which inputs, which code, which runtime, which thread policy, and which verification was or was not
performed. The lead decision requires the scientific payload to stay free of time, path, host and process data,
unknown environment facts to be labelled rather than guessed, and `not_run` never to look like a pass.

## Decision

Implemented in `src/axiom/environment.py` and `src/axiom/passport.py`. The accepted Core contract is unchanged:
the four existing fingerprints, `execute`, `AnalysisResult` and `environment_info()` are reused as they are.

**Passport `axiom.passport/1`** (one JSON object, sorted keys, full-precision floats):

| Section | Content | Deterministic? |
|---------|---------|----------------|
| `identity` | `raw_sha256`, `dataset_sha256`, `plan_sha256`, `result_sha256` plus the format versions (`axiom.dataset/1`, `axiom.plan/1`, `axiom.policy/1`, `axiom.result/1`, `axiom.canonical-json/1`) | yes |
| `inputs` | explicit CSV profile, data size and row count, the explicit plan (defaults written out), the `Limits` used | yes |
| `result` | the complete canonical result core, unrounded (its canonical SHA-256 must equal `identity.result_sha256`) | yes |
| `environment` | `axiom.environment/1` manifest (below) | no |
| `execution` | run nonce (`uuid4`), mode (`fresh_process`), workspace description, thread variables set, UTC start/finish, timings | no |
| `verification` | checks in the shared vocabulary (below) | no |
| `passport_sha256` | SHA-256 of the canonical passport body | n/a |

- The data file is **not** embedded (privacy); only its fingerprints and size. No absolute paths or file names are
  stored; replay takes the data path as an argument.
- `passport_sha256` detects accidental or naive edits. It is **not a signature**: an adversary can recompute every
  digest consistently (a test does exactly that). Such a passport is exposed only by recomputation from the data
  (ADR 0008). Hashes show identity and integrity, never correctness or anonymization.
- Parsing is strict and bounded: text above 64 MiB is rejected before parsing; duplicate keys, `NaN`/`Infinity` and
  numbers outside the binary64 range (`1e999`) are rejected; then the **complete structure** is validated
  (`src/axiom/passport_schema.py`): exact keys and JSON types at every level, nesting depth at most 32, digest syntax,
  the format versions this code supports, the verification vocabulary, the environment and execution records, and the
  full result block of each operation; every integer within the signed 64-bit range and every string and key
  encodable as UTF-8 (no unpaired surrogates). Violations raise `PassportError` with a stable code and the offending
  field (values and path parts quoted in messages are shortened to 64 characters),
  at the API (`validate_passport`, `integrity_checks`, `replay`, `compare_passports`) and the CLI (exit 2).
  `build_passport` validates what it produces.
- `integrity_checks()` recomputes, without the data: the passport digest, the result digest, the plan digest from the
  explicit plan (implicit defaults are rejected; the plan must name the result's operation), identity consistency
  (digests, versions and row count against the result core), that the profile and limits can be reconstructed,
  **result consistency** (counts, row ordinals, plan fields, cluster ids, sizes and widths agree with each other and
  with the validated plan and recorded limits; no cluster is empty and no recorded postcondition is false; skipped as
  `not_run` when the plan or the limits are invalid; every check is linear in the document and none is sized by a
  count the document states) and
  that the recorded verification is exactly the creation record of `axiom.passport/1`. Recorded verification is a
  historical record: replay and comparison never report it as a check they performed.

**Verification vocabulary.** Every check has a `kind` (`exact`, `numeric`, `semantic`), a `status` (`pass`, `fail`,
`not_applicable`, `not_run`) and an `evidence` type (`integrity`, `environment`, `repeatability`,
`cross_environment`, `internal_consistency`, `independent_reference`). Only `independent_reference` would mean a
comparison with an independent method; a passport and a replay never perform one and record it as `not_run`. K-means
postconditions are recorded as `internal_consistency`. A newly created passport records replay and cross-environment
agreement as `not_run`.

**Environment manifest `axiom.environment/1`** (actual observations; anything unobservable is
`{"status": "unavailable", "reason": ...}`):

| Item | Source |
|------|--------|
| Python version, implementation, compiler, build, float repr style | `platform`, `sys` (REF-024) |
| OS system, release, version, platform string, machine, processor, logical CPUs, byte order | `platform`, `os` |
| Versions of axiom-local, numpy, scipy, scikit-learn, threadpoolctl, joblib, pydantic, pydantic-core | `importlib.metadata.version` (REF-024) |
| Engine code: SHA-256 over relative names and bytes of the installed `axiom/*.py` files; Git commit and dirty flag; SHA-256 of `uv.lock` | file system, `git` (the latter two only in a source checkout) |
| NumPy BLAS/LAPACK name and version, SIMD baseline/found/not found | `numpy.show_config(mode="dicts")` (REF-023) |
| Thread-control variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`), native pools seen by threadpoolctl (without file paths), and a probe of the pools inside `threadpool_limits(limits=1)`, the context of the K-means fit | `os.environ`, threadpoolctl (REF-025, REF-026) |

`fit_limit_probe` is `verified_single_thread` only if every detected pool reports one thread inside the limit. A BLAS
that threadpoolctl cannot see is reported, not assumed: on the development Mac NumPy is built against Apple
Accelerate, which threadpoolctl does not list, so `blas_thread_control` reads
`unverified (NumPy BLAS 'accelerate' is not visible to threadpoolctl)`. `native_thread_status()` derives the overall
status: `verified_single_thread` only if the probe passed **and** NumPy's BLAS is among the introspected pools;
otherwise `unverified` with the reason (or `multi_threaded`). The probe alone never stands for "all threads verified".

**Comparing two manifests** (`compare_environments`): the fields of `COMPARABLE_FIELDS` plus the thread-control
variables, the detected pools (with thread counts), the probe and BLAS visibility are compared recursively. Each is
`equal`, `different` (observed on both sides and unequal somewhere) or `unknown` (an `unavailable` marker anywhere in
it, a version reported as `None` or `"unknown"`, or a thread status starting with `unverified`; in the thread
variables `None` means "not set" and is an observation). The result is `verified_same`, `unverified` (nothing observed
differs, something is unknown) or `different`. Unknown is never equal; ADR 0008 states how each status is used.

## Alternatives considered

- **Embedding the data bytes** (rejected: privacy; the passport would become a data export).
- **Rounding floats or a quantized digest** for easier matching (rejected: would manufacture agreement; the lead
  decision forbids it). Tolerances are applied at comparison time instead, with the full values retained.
- **W3C PROV / RO-Crate serialization** (deferred by the lead decision; the sections above map onto their concepts
  but no interoperability is claimed).
- **Signing passports** (out of scope: no key management exists; documented as a limitation).
- **Git commit alone as the code identity** (insufficient outside a checkout and blind to uncommitted edits; the
  engine file digest was added).

## Tradeoffs and consequences

- A passport for a K-means result contains the full assignment list, so its size grows with the number of rows.
- Environment capture loads the solver's native libraries to probe thread pools; this costs import time.
- The engine file digest and the lock digest hash the bytes on disk (raw identities; deliberately not normalized), so
  they depend on how the files were checked out. EXP-005 observed different digests on the Windows runner for the same
  source; EXP-006 (part D) confirmed the cause: the runner's Git has `core.autocrlf=true`, and every hashed file
  differed from its committed blob only by LF -> CRLF conversion. `.gitattributes` now checks out `*.py` and `uv.lock`
  with LF everywhere. A checkout made another way (other Git settings, an editor converting line endings, an
  installed wheel) can still give different digests; the result is conservative (reported as a different
  environment), and a common commit never implies a common runtime.
- `platform.version()` and `processor()` are recorded verbatim; they describe the OS build and CPU, not the user.

## Unvalidated assumptions

- That the listed fields are sufficient to explain a cross-environment difference. They are what can be observed
  portably; CPU microcode, libm versions and BLAS internals are not captured.
- That `numpy.show_config` reports the BLAS actually used at run time (it reports the build configuration).
- That threadpoolctl's view of pools is complete; libraries it does not support are invisible to it (observed for
  Accelerate).

## Revision after technical review (2026-11-25, before acceptance)

The review of head `3172cdc` found, and the implementer reproduced, that (B2) only top-level keys were checked, so
re-sealed passports with invalid verification entries, inconsistent identity metadata or arbitrary result blocks passed
`integrity_checks`, `1e999` parsed to infinity, and malformed objects raised `KeyError`/`TypeError`; and (B3) whole
dictionaries were compared, so equal `unavailable` markers or two `unverified` probes counted as equal. The structural
validation, the added integrity checks, `compare_environments` and `native_thread_status` above are the corrections;
`tests/test_review_regressions.py` holds one regression test per counterexample. The Core contract (ADR 0005) is
unchanged. Validation is not authentication: a consistently re-sealed passport with a plausible result still passes
and is exposed only by recomputation (EXP-006 A3).

## Revision after the final technical review (2026-11-25, before acceptance)

The [final review](https://github.com/sergthedev/axiom-local/pull/7#issuecomment-7172653455) of head `e8a9f57` found,
and the implementer reproduced at a safe scale, that `result_problems` built `list(range(k))` and `[0] * k` from the
cluster count `k` stated in the result, a natural number without an upper bound, and that `integrity_checks` ran it even
when the plan check had failed: a re-sealed passport of a few kilobytes made the parent allocate about 40 MB per million
claimed clusters, through `replay` and `compare_passports`. The cluster checks now enumerate the entries that exist,
check `k` against the validated plan and the recorded `max_k`, and count labels in a dictionary; the result checks are
skipped when the plan or limits are invalid. The final audit found further gaps in the same layer, each reproduced on
`e8a9f57` before the fix: JSON `\ud800` escapes decode to strings that cannot be hashed (`UnicodeEncodeError` inside
`integrity_checks`); integers above 4300 digits in API objects passed validation and could not be hashed
(`ValueError`); error messages quoted whole rejected values; an empty cluster in a re-sealed passport passed every check
and crashed `compare_passports` with `KeyError`. The 64-bit integer range, the surrogate rule, bounded messages and the
empty-cluster and postcondition checks are the corrections (`tests/test_final_audit.py`; EXP-007 part B). They reject
only documents this engine cannot issue. The memory used while parsing is linear in the text but up to about 25 times
its size for adversarial shapes (EXP-007 part P); at the 64 MiB cap that is an extrapolated 1.6 GiB, an open limitation.

## References

REF-021, REF-022, REF-023, REF-024, REF-025, REF-026
