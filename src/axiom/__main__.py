"""Minimal verification CLI: ``python -m axiom {env,analyze,replay,compare}`` (ADR 0008).

Output is JSON on stdout (or ``--out``). Exit status: 0 for ``exact_match`` or
``numeric_agreement``, 1 for any other verdict (discrepancy, failure, mismatch), 2 for usage or
input errors. This is a thin wrapper over :mod:`axiom.replay` for scripts and a future Studio UI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .environment import capture_environment
from .errors import AxiomError
from .limits import DEFAULT_LIMITS
from .passport import dumps_passport, loads_passport, profile_from_dict
from .plan import parse_plan_json
from .replay import AnalysisFailed, analyze, compare_passports, replay

_OK = ("exact_match", "numeric_agreement")


def _write(obj: Any, out: str | None) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if out:
        Path(out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def _read_passport(path: str) -> dict[str, Any]:
    return loads_passport(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m axiom", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_env = sub.add_parser("env", help="print the environment manifest")
    p_env.add_argument("--out")
    p_an = sub.add_parser("analyze", help="run a plan in a fresh process and write a passport")
    p_an.add_argument("--data", required=True)
    p_an.add_argument("--profile", required=True, help="CSV profile JSON file")
    p_an.add_argument("--plan", required=True, help="plan JSON file")
    p_an.add_argument("--out")
    p_re = sub.add_parser("replay", help="replay a passport in fresh processes")
    p_re.add_argument("--passport", required=True)
    p_re.add_argument("--data", required=True)
    p_re.add_argument("--runs", type=int, default=1)
    p_re.add_argument("--timeout", type=float, default=600.0)
    p_re.add_argument("--out")
    p_cmp = sub.add_parser("compare", help="compare two passports under the documented criteria")
    p_cmp.add_argument("--reference", required=True)
    p_cmp.add_argument("--observed", required=True)
    p_cmp.add_argument("--out")
    args = parser.parse_args(argv)

    try:
        if args.command == "env":
            _write(capture_environment(), args.out)
            return 0
        if args.command == "analyze":
            profile = profile_from_dict(json.loads(Path(args.profile).read_text(encoding="utf-8")))
            plan = parse_plan_json(Path(args.plan).read_text(encoding="utf-8"), DEFAULT_LIMITS)
            try:
                passport = analyze(args.data, profile, plan)
            except AnalysisFailed as exc:
                _write({"verdict": "execution_failure", "worker": exc.response}, None)
                return 1
            text = dumps_passport(passport)
            if args.out:
                Path(args.out).write_text(text, encoding="utf-8")
            else:
                sys.stdout.write(text)
            return 0
        if args.command == "replay":
            if args.runs < 1:
                raise ValueError("--runs must be >= 1")
            report = replay(
                _read_passport(args.passport), args.data, runs=args.runs, timeout=args.timeout
            )
            _write(report, args.out)
            return 0 if report["verdict"] in _OK else 1
        if args.command == "compare":
            report = compare_passports(
                _read_passport(args.reference), _read_passport(args.observed)
            )
            _write(report, args.out)
            return 0 if report["verdict"] in _OK else 1
    except AxiomError as exc:
        _write({"verdict": "input_error", "error": exc.to_dict()}, None)
        return 2
    except (OSError, ValueError) as exc:
        _write({"verdict": "input_error", "error": {"type": type(exc).__name__}}, None)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
