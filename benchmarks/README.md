# Benchmarks and evidence scripts

- `issue2_evidence.py` – measures the Issue #2 engine against exact oracles and NIST certified values, thread
  determinism, single-run timings and 20 fresh-process replays. Output is JSON on stdout.
- `issue2_candidate_smoke.py` – compares candidate kernels (standard library, NumPy, SciPy) against exact oracles.

Results and the environment they were measured in are in `docs/evidence/experiments/EXP-001-issue2-numerical-validation/`.
Timings are single runs on one machine, not performance claims. No model or UI benchmark exists.
