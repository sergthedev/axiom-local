"""EXP-005: cross-platform numerical comparison against committed reference passports (Issue #3).

Two modes, run from the repository root:

* ``--write-reference DIR`` analyzes the ten scenarios of ``issue3_common`` in fresh processes and
  writes one passport per scenario into DIR (done once, on the reference machine).
* ``--compare DIR`` analyzes the same scenarios on *this* machine and compares each new passport
  with the reference using ``axiom.replay.compare_passports`` (criteria ``axiom.agreement/1``,
  fixed in ADR 0008 and the EXP-005 pre-registration before any comparison). Prints JSON.

``--compare`` is report-only: it exits 0 whenever every requested comparison was performed,
whatever the verdicts, and 1 if any scenario could not be compared (the report then says
``"complete": false`` and lists each scenario's state). Verdicts must be read from the report; a
green CI step is not numerical agreement.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from issue3_common import file_sha256, git_state, scenarios  # noqa: E402

from axiom.errors import AxiomError  # noqa: E402
from axiom.passport import dumps_passport, loads_passport  # noqa: E402
from axiom.replay import AnalysisFailed, analyze, compare_passports  # noqa: E402


def _name(scenario_id: str) -> str:
    return scenario_id.replace("/", "__") + ".passport.json"


def write_reference(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="axiom-exp005-") as tmp:
        for sc in scenarios(Path(tmp)):
            passport = analyze(sc["data"], sc["profile"], sc["plan"])
            (directory / _name(sc["id"])).write_text(dumps_passport(passport), encoding="utf-8")
            print(sc["id"], passport["identity"]["result_sha256"], file=sys.stderr)


def compare(directory: Path) -> int:
    rows = []
    observed_env = reference_env = None
    with tempfile.TemporaryDirectory(prefix="axiom-exp005-") as tmp:
        all_scenarios = scenarios(Path(tmp))
        for sc in all_scenarios:
            row = {"scenario": sc["id"], "data_sha256": file_sha256(sc["data"]), "compared": False}
            rows.append(row)
            try:
                reference = loads_passport((directory / _name(sc["id"])).read_bytes())
                observed = analyze(sc["data"], sc["profile"], sc["plan"])
                report = compare_passports(reference, observed)
            except AnalysisFailed as exc:
                row.update(verdict="execution_failure", error=exc.response.get("error"))
                continue
            except (AxiomError, OSError) as exc:
                row.update(
                    verdict="not_compared",
                    error={"type": type(exc).__name__, "code": getattr(exc, "code", None)},
                )
                continue
            observed_env, reference_env = observed["environment"], reference["environment"]
            comparison = report.get("comparison", {})
            environment = report.get("environment") or {}
            row.update(
                compared=True,
                verdict=report["verdict"],
                comparison_mode=report.get("comparison_mode"),
                byte_identical=report.get("byte_identical"),
                environment_status=environment.get("status"),
                environment_fields_differing=[
                    d["field"] for d in environment.get("differences", [])
                ],
                environment_fields_unverified=[
                    d["field"] for d in environment.get("unverified_fields", [])
                ],
                native_threads=report.get("native_threads"),
                n_float_fields=comparison.get("n_float_fields"),
                n_float_fields_different=comparison.get("n_float_fields_different"),
                max_ulp_distance=comparison.get("max_ulp_distance"),
                max_abs_diff=comparison.get("max_abs_diff"),
                n_gating_failures=comparison.get("n_gating_failures"),
                field_differences=comparison.get("field_differences", [])[:50],
                non_float_differences=comparison.get("non_float_differences", [])[:50],
                partition=comparison.get("partition"),
                reference_result_sha256=reference["identity"]["result_sha256"],
                observed_result_sha256=observed["identity"]["result_sha256"],
                observed_threads=observed["environment"]["threads"],
                observed_code=observed["environment"]["code"],
            )
    n_compared = sum(1 for r in rows if r["compared"])
    report = {
        "experiment": "EXP-005 comparison script (also used by EXP-006)",
        "git": git_state(),
        "observed_environment": observed_env,
        "reference_environment": reference_env,
        "scenarios": rows,
        # A partial run must never read as complete: counts of requested and compared scenarios.
        "scenarios_requested": len(all_scenarios),
        "scenarios_compared": n_compared,
        "complete": n_compared == len(all_scenarios),
        "verdict_counts": {
            v: sum(1 for r in rows if r["verdict"] == v)
            for v in sorted({r["verdict"] for r in rows})
        },
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    sys.stdout.write("\n")
    return 0 if report["complete"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write-reference", type=Path)
    group.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.write_reference:
        write_reference(args.write_reference)
        return 0
    return compare(args.compare)


if __name__ == "__main__":
    raise SystemExit(main())
