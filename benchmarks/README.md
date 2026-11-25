# Benchmarks and evidence scripts

- `issue2_evidence.py` – measures the Issue #2 engine against exact oracles and NIST certified values, thread
  determinism, single-run timings and 20 fresh-process replays. Output is JSON on stdout.
- `issue2_candidate_smoke.py` – compares candidate kernels (standard library, NumPy, SciPy) against exact oracles.

- `issue2_quartile_confirmation.py`, `issue2_cluster_workload.py` – EXP-002 and EXP-003.
- `issue3_common.py` – the ten Issue #3 scenarios (example table and an integer-only 5,000-row synthetic table).
- `issue3_replay_benchmark.py` – EXP-004: 20 fresh-process replays per scenario, mutations, malformed input.
- `issue3_cross_platform.py` – EXP-005: writes reference passports or compares this machine against them (run in CI,
  report only).

Results and the environments they were measured in are in `docs/evidence/experiments/`.
Timings are single runs on one machine, not performance claims. No model or UI benchmark exists.
