# ADR 0007: Analysis Passport v1 and environment capture

- Status: Proposed (awaiting technical review and owner approval)
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
- Parsing is strict: duplicate keys, `NaN`/`Infinity`, unknown or missing top-level fields and other versions are
  rejected. `integrity_checks()` recomputes, without the data: the passport digest, the result digest, the plan digest
  from the explicit plan (implicit defaults are rejected), identity consistency with the result core, and that the
  profile and limits can be reconstructed.

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
`unverified (NumPy BLAS 'accelerate' is not visible to threadpoolctl)`.

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
- `platform.version()` and `processor()` are recorded verbatim; they describe the OS build and CPU, not the user.

## Unvalidated assumptions

- That the listed fields are sufficient to explain a cross-environment difference. They are what can be observed
  portably; CPU microcode, libm versions and BLAS internals are not captured.
- That `numpy.show_config` reports the BLAS actually used at run time (it reports the build configuration).
- That threadpoolctl's view of pools is complete; libraries it does not support are invisible to it (observed for
  Accelerate).

## References

REF-021, REF-022, REF-023, REF-024, REF-025, REF-026
