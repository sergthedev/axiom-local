"""EXP-009 analysis: score raw model outputs, rerun the rules, compare, apply the frozen rule.

Run from the repository root in the Core environment after the model runs have finished:

    EXP=docs/evidence/experiments/EXP-009-untuned-baselines
    uv run python benchmarks/issue4_exp009_analyze.py --runs $EXP/runs \
        --rules-predictions $EXP/rules_validation.jsonl > $EXP/results.json

Inputs are the raw outputs written by ``envs/model/exp009_run.py`` (no MLX needed here). Loads
``axbench/0.1-ai-reviewed`` through its verified manifest (hash pinned below), reruns
``axiom.rules/0.1`` on the same frozen validation file, scores every system with harness
``axbench.evaluate/0.2`` and scorer ``axbench.score/0.2`` (Guard policy 2 in the after-Guard layer),
computes the paired comparisons (2x2, percentage-point delta, paired cluster bootstrap over
lineages, 2000 replicates, seed 2026; McNemar as IID illustration only), summarizes resources and
repeatability, and applies the base-selection rule frozen in protocol revision 3
(:mod:`axbench.selection`). The final test split is not generated, read or scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
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
EXP013 = ROOT / "docs" / "evidence" / "experiments" / "EXP-013-rules-frozen-ai-reviewed"


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


def resources(meas: dict[str, Any], preds: list[dict[str, Any]]) -> dict[str, Any]:
    gen = [p["generation_tokens"] for p in preds if p.get("generation_tokens") is not None]
    return {
        "status": meas["status"],
        "stop_reason": meas.get("stop_reason"),
        "power_source_before": meas["before"]["power_source"],
        "cold_load_seconds_one_observation": meas.get("cold_load_seconds"),
        "warm_latency_ms_160": _quartiles([p["elapsed_ms"] for p in preds if p.get("elapsed_ms")]),
        "generated_tokens_per_request": _quartiles([float(g) for g in gen]),
        "requests_hitting_256_token_cap": sum(p.get("finish_reason") == "length" for p in preds),
        "prompt_tokens_per_request": _quartiles(
            [float(p["prompt_tokens"]) for p in preds if p.get("prompt_tokens") is not None]
        ),
        "process_peak_rss_bytes": meas.get("rss_peak_process_bytes"),
        "process_peak_rss_after_validation_bytes": meas.get("rss_peak_after_validation_bytes"),
        "mlx_peak_load_bytes": meas.get("load_mlx_peak_bytes"),
        "mlx_peak_validation_run_bytes": meas.get("validation_mlx_peak_bytes"),
        "mlx_peak_repeat_run_bytes": meas.get("repeat_mlx_peak_bytes"),
        "swap_and_pressure": {
            k: meas.get(k) for k in ("before", "after_warmup", "after_validation", "after")
        },
        "validation_watch": meas.get("validation_watch"),
        "repeat_watch": meas.get("repeat_watch"),
        "note": "RSS, MLX peak, swap and pressure are separate measurements and are never summed",
    }


def repeatability(
    examples: dict[str, dict[str, Any]],
    rep_ids: list[str],
    reference: dict[str, dict[str, Any]],
    in_process: list[dict[str, Any]],
    fresh: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    def outcome(p: dict[str, Any]) -> tuple[str | None, str | None, bool]:
        s = score_one(examples[p["example_id"]], p)
        return p.get("raw_output"), canonical(p.get("raw_output")), s.exact_match

    per_id = {}
    for i in rep_ids:
        ref = outcome(reference[i])
        inproc = [outcome(p) for p in in_process if p["example_id"] == i]
        fr = [outcome(p) for runs in fresh.values() for p in runs if p["example_id"] == i]
        per_id[i] = {
            "in_process_n": len(inproc),
            "fresh_n": len(fr),
            "in_process_text_identical": all(o[0] == ref[0] for o in inproc),
            "in_process_candidate_identical": all(o[1] == ref[1] for o in inproc),
            "in_process_correctness_identical": all(o[2] == ref[2] for o in inproc),
            "fresh_text_identical": all(o[0] == ref[0] for o in fr),
            "fresh_candidate_identical": all(o[1] == ref[1] for o in fr),
            "fresh_correctness_identical": all(o[2] == ref[2] for o in fr),
            "distinct_texts_all": len({o[0] for o in [ref, *inproc, *fr]}),
        }
    keys = [k for k in next(iter(per_id.values())) if k.endswith("identical")]
    return {
        "n_requests": len(rep_ids),
        "reference": "the request's output in the 160-request validation run of the main process",
        "in_process_repeats": sorted({p["repeat"] for p in in_process}),
        "fresh_processes_completed": sorted(fresh),
        "requests_identical": {k: sum(v[k] for v in per_id.values()) for k in keys},
        "per_request": per_id,
    }


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
    by_id = {e["id"]: e for e in val}
    clusters = [e["lineage_id"] for e in val]
    rep_ids = repeat_subset(val)
    runs = Path(args.runs)

    rules_preds = run_router(rule_router(), val, system="rules")
    Path(args.rules_predictions).write_bytes(_jsonl(rules_preds))
    reports = {
        "rules": evaluate(val, rules_preds, system=f"rules {RULES_VERSION}", split="validation")
    }
    exp013 = load_jsonl(EXP013 / "rules_validation.jsonl")
    rules_consistency = {
        "compared_with": "EXP-013 rules_validation.jsonl (same rules, same frozen file)",
        "raw_output_differs": sorted(
            a["example_id"]
            for a, b in zip(rules_preds, exp013, strict=True)
            if a["example_id"] != b["example_id"] or a["raw_output"] != b["raw_output"]
        ),
    }
    model_runs: dict[str, ModelRun] = {}
    extra: dict[str, Any] = {}
    for system, folder in MODELS.items():
        d = runs / folder
        meas = json.loads((d / "main_measurements.json").read_text(encoding="utf-8"))
        preds = load_jsonl(d / "main_validation.jsonl")
        report = evaluate(val, preds, system=system, split="validation")
        res = resources(meas, preds)
        report["memory"] = {
            k: res[k] for k in ("process_peak_rss_bytes", "mlx_peak_validation_run_bytes")
        }
        report["memory_note"] = "process peak RSS (getrusage) and MLX peak, never summed"
        reports[system] = report
        # Eligibility refers to the 160-request validation run (pre-run record, D10).
        completed = meas.get("validation_completed") is True and len(preds) == len(val)
        per = report["per_example"]
        model_runs[system] = ModelRun(
            name=system,
            completed=completed,
            peak_rss_bytes=meas.get("rss_peak_after_validation_bytes"),
            median_warm_latency_ms=(res["warm_latency_ms_160"] or {}).get("median"),
            exact=[r["exact_match"] for r in per],
            schema_valid=[r["schema_valid"] for r in per],
            unauthorized_dispatches=report["after_guard"]["unauthorized_dispatch_count"],
        )
        fresh = {}
        fresh_meas = {}
        for i in FRESH:
            p = d / f"fresh_{i}.jsonl"
            if p.exists():
                fresh[i] = load_jsonl(p)
                fm = json.loads((d / f"fresh_{i}_measurements.json").read_text(encoding="utf-8"))
                fresh_meas[i] = {
                    k: fm.get(k)
                    for k in (
                        "status",
                        "stop_reason",
                        "cold_load_seconds",
                        "rss_peak_process_bytes",
                    )
                }
        ref = {p["example_id"]: p for p in preds}
        extra[system] = {
            "resources": res,
            "fresh_processes": fresh_meas,
            "repeatability": repeatability(
                by_id, rep_ids, ref, load_jsonl(d / "main_repeat.jsonl"), fresh
            )
            if all(i in ref for i in rep_ids)
            else None,
            "environment": meas["environment"],
            "model": meas["model"],
            "eos_token_ids": meas.get("eos_token_ids"),
            "warmup_ids": meas.get("warmup_ids"),
        }

    def vec(system: str, field: str) -> list[bool]:
        return [r[field] for r in reports[system]["per_example"]]

    comparisons = {}
    for a, b in (("rules", SMALL), ("rules", LARGE), (SMALL, LARGE)):
        for field in ("exact_match", "schema_valid"):
            comparisons[f"{b} vs {a}: {field}"] = paired(
                vec(a, field), vec(b, field), clusters, reps=2000, seed=2026
            )
    decision = select_base(
        model_runs[SMALL],
        model_runs[LARGE],
        clusters,
        reports["rules"]["after_guard"]["unauthorized_dispatch_count"],
    )
    result = {
        "experiment": "EXP-009",
        "git": git_state(),
        "corpus_version": manifest["corpus_version"],
        "corpus_manifest_sha256": observed,
        "corpus_files": manifest["files"],
        "corpus_status": "owner-authorized AI review, 0 human decisions; exploratory",
        "harness_version": HARNESS_VERSION,
        "score_version": SCORE_VERSION,
        "rules_version": RULES_VERSION,
        "selection_version": SELECTION_VERSION,
        "repeat_ids": rep_ids,
        "rules_rerun_consistency": rules_consistency,
        "systems": reports,
        "models": extra,
        "paired_comparisons": comparisons,
        "selection": {
            "selected": decision.selected,
            "basis": decision.basis,
            "by_evidence": decision.by_evidence,
            "trace": decision.trace,
            "eligibility": decision.eligibility,
            "comparisons": decision.comparisons,
        },
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
