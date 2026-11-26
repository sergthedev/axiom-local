"""EXP-009 analysis: score raw model outputs, rerun the rules, compare, apply the frozen rule.

Run from the repository root in the Core environment after the model runs have finished:

    EXP=docs/evidence/experiments/EXP-009-untuned-baselines
    uv run python benchmarks/issue4_exp009_analyze.py --runs $EXP/runs \
        --rules-predictions $EXP/rules_validation.jsonl > $EXP/results.json

Inputs are the raw outputs and measurement files written by ``envs/model/exp009_run.py`` (no MLX
needed here). Loads ``axbench/0.1-ai-reviewed`` through its verified manifest (hash pinned below),
reruns ``axiom.rules/0.1`` on the same frozen validation file, scores every system with harness
``axbench.evaluate/0.2`` and scorer ``axbench.score/0.2`` (Guard policy 2 in the after-Guard layer),
computes the paired comparisons (2x2, percentage-point delta, paired cluster bootstrap over
lineages, 2000 replicates, seed 2026; McNemar as IID illustration only), summarizes resources and
repeatability, and applies the base-selection rule frozen in protocol revision 3
(:mod:`axbench.selection`) with the eligibility scope fixed in PRERUN revision 1.

Before the rule is applied, the inputs are checked (PRERUN revision 1): one commit, clean trees and
the pinned runner version in every process; the pinned corpus and model revisions; exactly the
expected record keys in every output file (no duplicate, missing or unexpected record); rules rerun
identical to EXP-013. Any integrity issue blocks the selection; it is reported, not repaired.
Errors stay in every denominator. The final test split is not generated, read or scored.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from issue3_common import git_state  # noqa: E402

from axbench.corpus import load_corpus  # noqa: E402
from axbench.evaluate import HARNESS_VERSION, evaluate, rule_router, run_router  # noqa: E402
from axbench.runwatch import repeat_subset  # noqa: E402
from axbench.score import SCORE_VERSION, load_jsonl, score_one  # noqa: E402
from axbench.selection import LARGE, SELECTION_VERSION, SMALL, ModelRun, select_base  # noqa: E402
from axbench.stats import paired  # noqa: E402
from axiom_router import parse_candidate  # noqa: E402
from axiom_router.contract import CandidateError, candidate_to_dict  # noqa: E402
from axiom_router.rules import RULES_VERSION  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "axbench" / "v0.1-ai-reviewed"
MANIFEST_SHA256 = "b8b6d39f8cf942629d4abeff8e7d96f4b876c37ebf13f4082f7f5d47f516239c"
MODELS = {SMALL: "qwen3-0.6b", LARGE: "qwen3-1.7b"}
FRESH = range(1, 6)
REPEATS = range(1, 21)
RUNNER_VERSION = "exp009.runner/2"
EXP013 = ROOT / "docs" / "evidence" / "experiments" / "EXP-013-rules-frozen-ai-reviewed"


def _pinned_revisions() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("provision", ROOT / "envs/model/provision.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {name: c["revision"] for name, c in mod.CHECKPOINTS.items()}


def _jsonl(records) -> bytes:
    return "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records
    ).encode("utf-8")


def _quartiles(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    q = statistics.quantiles(values, n=4) if len(values) >= 2 else [values[0]] * 3
    return {
        "n": len(values),
        "median": statistics.median(values),
        "q1": q[0],
        "q3": q[2],
        "min": min(values),
        "max": max(values),
    }


def canonical(raw: str | None) -> str | None:
    """Canonical candidate JSON of a schema-valid output, else ``None``."""
    if raw is None:
        return None
    try:
        return json.dumps(candidate_to_dict(parse_candidate(raw)), sort_keys=True)
    except CandidateError:
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_jsonl(path: Path) -> list[dict[str, Any]] | None:
    return load_jsonl(path) if path.exists() else None


def check_records(
    records: list[dict[str, Any]] | None,
    expected: list[tuple],
    key: tuple[str, ...],
    system: str,
) -> dict[str, Any]:
    """Exact expected keys: no duplicate, missing or unexpected record; statuses counted."""
    if records is None:
        return {"present": False, "expected": len(expected), "observed": 0, "complete": False}
    keys = [tuple(r.get(k) for k in key) for r in records]
    counts = Counter(keys)
    exp = set(expected)
    duplicates = sorted(str(k) for k, c in counts.items() if c > 1)
    unexpected = sorted(str(k) for k in counts if k not in exp)
    missing = [k for k in expected if k not in counts]
    wrong_system = sum(r.get("system") != system for r in records)
    status = Counter(str(r.get("status")) for r in records)
    return {
        "present": True,
        "expected": len(expected),
        "observed": len(records),
        "missing": len(missing),
        "first_missing": [str(k) for k in missing[:5]],
        "duplicates": duplicates,
        "unexpected": unexpected,
        "wrong_system": wrong_system,
        "status": dict(sorted(status.items())),
        "complete": not missing and not duplicates and not unexpected and not wrong_system,
    }


def resources(meas: dict[str, Any] | None, preds: list[dict[str, Any]]) -> dict[str, Any] | None:
    if meas is None:
        return None
    gen = [p["generation_tokens"] for p in preds if p.get("generation_tokens") is not None]
    lat = [p["elapsed_ms"] for p in preds if isinstance(p.get("elapsed_ms"), (int, float))]
    return {
        "status": meas.get("status"),
        "stop_reason": meas.get("stop_reason"),
        "failure": meas.get("failure"),
        "preflight": meas.get("preflight"),
        "cold_load_seconds_one_observation": meas.get("cold_load_seconds"),
        "model_rehash_seconds_before_load": meas.get("model_rehash_seconds"),
        "warm_latency_ms_validation": _quartiles(lat),
        "generated_tokens_per_request": _quartiles([float(g) for g in gen]),
        "requests_hitting_256_token_cap": sum(p.get("finish_reason") == "length" for p in preds),
        "prompt_tokens_per_request": _quartiles(
            [float(p["prompt_tokens"]) for p in preds if p.get("prompt_tokens") is not None]
        ),
        "process_peak_rss_bytes": meas.get("rss_peak_process_bytes"),
        "process_peak_rss_after_load_bytes": meas.get("rss_peak_after_load_bytes"),
        "process_peak_rss_after_validation_bytes": meas.get("rss_peak_after_validation_bytes"),
        "mlx_peak_load_bytes": meas.get("load_mlx_peak_bytes"),
        "mlx_peak_validation_run_bytes": meas.get("validation_mlx_peak_bytes"),
        "mlx_peak_repeat_run_bytes": meas.get("repeat_mlx_peak_bytes"),
        "snapshots": {
            k: meas.get(k) for k in ("preflight", "after_warmup", "after_validation", "after")
        },
        "resource_monitor": meas.get("resource_monitor"),
        "validation_slowdown": meas.get("validation_slowdown"),
        "repeat_slowdown": meas.get("repeat_slowdown"),
        "note": "RSS, MLX peak, swap and pressure are separate measurements and are never summed",
    }


def _outcome(examples, p) -> dict[str, Any]:
    s = score_one(examples[p["example_id"]], p)
    raw = p.get("raw_output") if p.get("status") == "ok" else None
    return {
        "ok": p.get("status") == "ok",
        "text": raw,
        "cand": canonical(raw),
        "exact": s.exact_match,
    }


def repeatability(
    examples: dict[str, dict[str, Any]],
    rep_ids: list[str],
    reference: dict[str, dict[str, Any]],
    in_process: list[dict[str, Any]],
    fresh: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Per request: identical to the validation-run output? ``None`` when not assessable.

    Not assessable: the reference is missing or failed, an expected observation is missing, or an
    observation failed (failed requests are counted separately). Candidate identity is not
    assessed when the reference output is not a schema-valid candidate.
    """
    inproc = {(p["example_id"], p.get("repeat")): p for p in in_process}
    fr = {(p["example_id"], i): p for i, runs in fresh.items() for p in runs}
    per_id: dict[str, Any] = {}
    for i in rep_ids:
        ref = _outcome(examples, reference[i]) if i in reference else None
        row: dict[str, Any] = {"reference_ok": bool(ref and ref["ok"])}
        for label, store, expected in (
            ("in_process", inproc, [(i, r) for r in REPEATS]),
            ("fresh", fr, [(i, f) for f in FRESH]),
        ):
            obs = [_outcome(examples, store[k]) for k in expected if k in store]
            failed = sum(not o["ok"] for o in obs)
            assessable = row["reference_ok"] and len(obs) == len(expected) and not failed
            row[f"{label}_observed"] = len(obs)
            row[f"{label}_expected"] = len(expected)
            row[f"{label}_failed"] = failed
            for name, field in (("text", "text"), ("candidate", "cand"), ("correctness", "exact")):
                ok = assessable and not (name == "candidate" and ref["cand"] is None)
                row[f"{label}_{name}_identical"] = (
                    all(o[field] == ref[field] for o in obs) if ok else None
                )
            row[f"{label}_distinct_texts"] = len({o["text"] for o in obs if o["ok"]})
        per_id[i] = row
    summary = {}
    for label in ("in_process", "fresh"):
        for name in ("text", "candidate", "correctness"):
            vals = [r[f"{label}_{name}_identical"] for r in per_id.values()]
            summary[f"{label}_{name}"] = {
                "identical": sum(v is True for v in vals),
                "differs": sum(v is False for v in vals),
                "not_assessable": sum(v is None for v in vals),
                "n_requests": len(vals),
            }
        summary[f"{label}_failed_requests"] = sum(r[f"{label}_failed"] for r in per_id.values())
        summary[f"{label}_observations"] = {
            "observed": sum(r[f"{label}_observed"] for r in per_id.values()),
            "expected": sum(r[f"{label}_expected"] for r in per_id.values()),
        }
    return {
        "n_requests": len(rep_ids),
        "reference": "the request's output in the 160-request validation run of the main process",
        "summary": summary,
        "per_request": per_id,
    }


def model_section(
    system: str, d: Path, val: list[dict[str, Any]], rep_ids: list[str], pinned_revision: str
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Load, validate and summarize one model's processes. Returns (section, issues, raw)."""
    issues: list[str] = []
    by_id = {e["id"]: e for e in val}
    main = _read_json(d / "main_measurements.json")
    preds = _read_jsonl(d / "main_validation.jsonl")
    repeat = _read_jsonl(d / "main_repeat.jsonl")
    checks = {
        "validation": check_records(preds, [(e["id"],) for e in val], ("example_id",), system),
        "repeat": check_records(
            repeat, [(i, r) for r in REPEATS for i in rep_ids], ("example_id", "repeat"), system
        ),
    }
    fresh: dict[int, list[dict[str, Any]]] = {}
    fresh_meas: dict[int, Any] = {}
    for f in FRESH:
        recs = _read_jsonl(d / f"fresh_{f}_repeat.jsonl")
        fm = _read_json(d / f"fresh_{f}_measurements.json")
        checks[f"fresh_{f}"] = check_records(
            recs, [(i, f) for i in rep_ids], ("example_id", "fresh_process"), system
        )
        fresh_meas[f] = (
            None
            if fm is None
            else {
                k: fm.get(k)
                for k in (
                    "status",
                    "stop_reason",
                    "failure",
                    "cold_load_seconds",
                    "rss_peak_process_bytes",
                    "preflight",
                    "resource_monitor",
                    "after",
                )
            }
        )
        if recs:
            fresh[f] = recs
    processes = {"main": main, **{f"fresh_{f}": _read_json(d / f"fresh_{f}_measurements.json")
                                  for f in FRESH}}  # fmt: skip
    for name, m in processes.items():
        if m is None:
            continue
        env = m.get("environment", {})
        if m.get("experiment") != "EXP-009" or m.get("system") != system:
            issues.append(f"{system} {name}: experiment or system name differs")
        if m.get("corpus_manifest_sha256") != MANIFEST_SHA256:
            issues.append(f"{system} {name}: corpus manifest hash differs")
        if (m.get("model") or {}).get("revision") != pinned_revision:
            issues.append(f"{system} {name}: model revision differs from the pin")
        if env.get("runner_version") != RUNNER_VERSION:
            issues.append(f"{system} {name}: runner version differs")
        if (env.get("git") or {}).get("dirty") is not False:
            issues.append(f"{system} {name}: tree was not clean")
    for name, c in checks.items():
        if c["present"] and (c["duplicates"] or c["unexpected"] or c["wrong_system"]):
            issues.append(f"{system} {name}: unexpected, duplicate or mislabeled records")
    if preds is None:
        issues.append(f"{system}: no validation output")
    commits = {
        (m.get("environment") or {}).get("git", {}).get("commit")
        for m in processes.values()
        if m is not None
    }
    all_processes_completed = all(
        m is not None and m.get("status") == "completed" for m in processes.values()
    ) and all(c["complete"] for c in checks.values())
    rss = [m.get("rss_peak_process_bytes") if m else None for m in processes.values()]
    section = {
        "all_required_processes_completed": all_processes_completed,
        "max_process_peak_rss_bytes_all_processes": None if None in rss else max(rss),
        "validation_only": {
            "completed": bool(main and main.get("validation_completed"))
            and checks["validation"]["complete"],
            "process_peak_rss_after_validation_bytes": (main or {}).get(
                "rss_peak_after_validation_bytes"
            ),
        },
        "record_checks": checks,
        "process_statuses": {k: (m or {}).get("status") for k, m in processes.items()},
        "resources": resources(main, preds or []),
        "fresh_processes": fresh_meas,
        "environment": (main or {}).get("environment"),
        "model": (main or {}).get("model"),
        "eos_token_ids": (main or {}).get("eos_token_ids"),
        "warmup_ids": (main or {}).get("warmup_ids"),
    }
    ref = {p["example_id"]: p for p in preds or []}
    section["repeatability"] = repeatability(by_id, rep_ids, ref, repeat or [], fresh)
    raw = {"preds": preds or [], "commits": commits, "rss": rss}
    return section, issues, raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--rules-predictions", required=True)
    args = parser.parse_args()
    observed = hashlib.sha256((CORPUS / "MANIFEST.json").read_bytes()).hexdigest()
    if observed != MANIFEST_SHA256:
        raise SystemExit("frozen corpus manifest does not match the pre-registered hash")
    manifest, corpus = load_corpus(CORPUS)
    val = corpus["validation"]
    clusters = [e["lineage_id"] for e in val]
    rep_ids = repeat_subset(val)
    runs = Path(args.runs)
    pins = _pinned_revisions()
    issues: list[str] = []

    rules_preds = run_router(rule_router(), val, system="rules")
    Path(args.rules_predictions).write_bytes(_jsonl(rules_preds))
    reports = {
        "rules": evaluate(val, rules_preds, system=f"rules {RULES_VERSION}", split="validation")
    }
    exp013 = load_jsonl(EXP013 / "rules_validation.jsonl")
    rules_diff = sorted(
        a["example_id"]
        for a, b in zip(rules_preds, exp013, strict=True)
        if a["example_id"] != b["example_id"] or a["raw_output"] != b["raw_output"]
    )
    if rules_diff:
        issues.append(f"rules rerun differs from EXP-013 on {len(rules_diff)} examples")

    sections: dict[str, Any] = {}
    model_runs: dict[str, ModelRun] = {}
    commits: set = set()
    for system, folder in MODELS.items():
        section, model_issues, raw = model_section(
            system, runs / folder, val, rep_ids, pins[folder]
        )
        issues += model_issues
        commits |= raw["commits"]
        report = evaluate(val, raw["preds"], system=system, split="validation")
        report["memory"] = {
            "max_process_peak_rss_bytes_all_processes": section[
                "max_process_peak_rss_bytes_all_processes"
            ],
            "mlx_peak_validation_run_bytes": (section["resources"] or {}).get(
                "mlx_peak_validation_run_bytes"
            ),
        }
        report["memory_note"] = "process peak RSS (getrusage) and MLX peak, never summed"
        reports[system] = report
        sections[system] = section
        lat = (section["resources"] or {}).get("warm_latency_ms_validation") or {}
        per = report["per_example"]
        model_runs[system] = ModelRun(
            name=system,
            completed=section["all_required_processes_completed"],
            peak_rss_bytes=section["max_process_peak_rss_bytes_all_processes"],
            median_warm_latency_ms=lat.get("median") if lat.get("n") == len(val) else None,
            exact=[r["exact_match"] for r in per],
            schema_valid=[r["schema_valid"] for r in per],
            unauthorized_dispatches=report["after_guard"]["unauthorized_dispatch_count"],
        )
    if len(commits) != 1 or None in commits:
        issues.append(f"model processes ran on {len(commits)} different or unknown commits")

    def vec(system: str, field: str) -> list[bool]:
        return [r[field] for r in reports[system]["per_example"]]

    comparisons = {}
    for a, b in (("rules", SMALL), ("rules", LARGE), (SMALL, LARGE)):
        for field in ("exact_match", "schema_valid"):
            comparisons[f"{b} vs {a}: {field}"] = paired(
                vec(a, field), vec(b, field), clusters, reps=2000, seed=2026
            )
    if issues:
        selection: dict[str, Any] = {
            "selected": None,
            "basis": "integrity_failure",
            "by_evidence": False,
            "issues": issues,
            "trace": ["inputs failed the integrity checks; the rule was not applied"],
        }
    else:
        decision = select_base(
            model_runs[SMALL],
            model_runs[LARGE],
            clusters,
            reports["rules"]["after_guard"]["unauthorized_dispatch_count"],
        )
        selection = {
            "selected": decision.selected,
            "basis": decision.basis,
            "by_evidence": decision.by_evidence,
            "issues": [],
            "trace": decision.trace,
            "eligibility": decision.eligibility,
            "comparisons": decision.comparisons,
        }
    result = {
        "experiment": "EXP-009",
        "git": git_state(),
        "model_process_commits": sorted(c for c in commits if c),
        "corpus_version": manifest["corpus_version"],
        "corpus_manifest_sha256": observed,
        "corpus_files": manifest["files"],
        "corpus_status": "owner-authorized AI review, 0 human decisions; exploratory",
        "harness_version": HARNESS_VERSION,
        "score_version": SCORE_VERSION,
        "rules_version": RULES_VERSION,
        "selection_version": SELECTION_VERSION,
        "eligibility_scope": "PRERUN revision 1: main and all five fresh processes completed "
        "without a stop; maximum process-lifetime peak RSS across them; median warm latency "
        "over the 160 validation requests",
        "repeat_ids": rep_ids,
        "rules_rerun_consistency": {
            "compared_with": "EXP-013 rules_validation.jsonl (same rules, same frozen file)",
            "raw_output_differs": rules_diff,
        },
        "systems": reports,
        "models": sections,
        "paired_comparisons": comparisons,
        "selection": selection,
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
