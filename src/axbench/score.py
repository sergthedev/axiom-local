"""AXBENCH scoring (ADR 0011, DR-005).

Every example is scored on the *full denominator*: a technical failure (timeout, unavailable
model, crash), invalid JSON or an invalid candidate counts as incorrect and is never removed.

Primary endpoint: **semantic exact match** of the raw router candidate with the frozen gold label,
before any human correction. Canonical semantic comparison follows the engine contract:

* ``inspect``: the *set* of columns (each column is summarized independently; order is display);
* ``relate``: the unordered pair {x, y} and the method (Pearson and Spearman are symmetric);
* ``compare``: ``group_by`` and ``measure`` by role;
* ``cluster``: the *ordered* feature list and ``k`` (ordered per the Core contract);
* ``CLARIFY`` / ``REJECT``: the action and the frozen reason code; optional hints are not scored.

Accepted variants listed on an example (fixed before predictions) also count as a match.
Secondary metrics separate JSON validity, candidate-schema validity, semantic validity, the Guard
outcome, operation and slot accuracy, clarification and rejection recall, false rejection,
coverage, wrong executable proposals and unauthorized dispatches.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from axiom_router.contract import CandidateError, candidate_to_dict, parse_candidate
from axiom_router.schema import project_columns
from axiom_router.semantic import SemanticError, validate_candidate

from .stats import cluster_bootstrap, wilson

SCORE_VERSION = "axbench.score/0.1"
STATUSES = ("ok", "timeout", "unavailable", "error")


def semantic_key(candidate: dict[str, Any]) -> tuple:
    action = candidate.get("action")
    if action in ("CLARIFY", "REJECT"):
        return (action, candidate.get("reason"))
    op = candidate.get("operation")
    if op == "inspect":
        return ("PLAN", op, frozenset(candidate["columns"]))
    if op == "relate":
        return ("PLAN", op, frozenset((candidate["x"], candidate["y"])), candidate["method"])
    if op == "compare":
        return ("PLAN", op, candidate["group_by"], candidate["measure"])
    if op == "cluster":
        return ("PLAN", op, tuple(candidate["features"]), candidate["k"])
    raise ValueError("not a candidate")


@dataclass(frozen=True, slots=True)
class Scored:
    id: str
    status: str
    json_valid: bool
    schema_valid: bool
    semantic_valid: bool
    predicted: dict[str, Any] | None
    error_stage: str | None
    error_code: str | None
    exact_match: bool
    action_match: bool
    reason_match: bool | None
    operation_match: bool | None
    slots: dict[str, bool]
    validator_accepted_plan: bool
    wrong_executable: bool
    elapsed_ms: float | None

    def to_record(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


def _slots(gold: dict[str, Any], pred: dict[str, Any] | None) -> dict[str, bool]:
    if gold["action"] != "PLAN":
        return {}
    p = (
        pred
        if pred and pred.get("action") == "PLAN" and pred.get("operation") == gold["operation"]
        else None
    )
    op = gold["operation"]
    if op == "inspect":
        return {"columns": bool(p) and set(p["columns"]) == set(gold["columns"])}
    if op == "relate":
        return {
            "columns": bool(p) and {p["x"], p["y"]} == {gold["x"], gold["y"]},
            "method": bool(p) and p["method"] == gold["method"],
        }
    if op == "compare":
        return {
            "group_by": bool(p) and p["group_by"] == gold["group_by"],
            "measure": bool(p) and p["measure"] == gold["measure"],
        }
    return {
        "features": bool(p) and list(p["features"]) == list(gold["features"]),
        "k": bool(p) and p["k"] == gold["k"],
    }


def score_one(example: dict[str, Any], prediction: dict[str, Any]) -> Scored:
    status = prediction.get("status", "error")
    if status not in STATUSES:
        status = "error"
    raw = prediction.get("raw_output")
    gold = example["gold"]
    view = project_columns([(c["name"], c["type"]) for c in example["schema"]["columns"]])
    json_valid = schema_valid = semantic_valid = False
    pred: dict[str, Any] | None = None
    stage = code = None
    if status != "ok":
        stage, code = "router", status
    else:
        try:
            cand = parse_candidate(raw)
            json_valid = schema_valid = True
            pred = candidate_to_dict(cand)
            validate_candidate(cand, view)
            semantic_valid = True
        except CandidateError as exc:
            stage, code = exc.stage, exc.code
            json_valid = exc.stage == "schema"
        except SemanticError as exc:
            stage, code = "semantic", exc.code
    gold_keys = {semantic_key(gold)} | {
        semantic_key(v) for v in example.get("accepted_variants", [])
    }
    exact = pred is not None and semantic_key(pred) in gold_keys
    action_match = pred is not None and pred["action"] == gold["action"]
    reason_match = None
    if gold["action"] != "PLAN":
        reason_match = action_match and pred.get("reason") == gold.get("reason")
    op_match = None
    if gold["action"] == "PLAN":
        op_match = action_match and pred.get("operation") == gold["operation"]
    accepted_plan = pred is not None and pred["action"] == "PLAN" and semantic_valid
    elapsed = prediction.get("elapsed_ms")
    return Scored(
        id=example["id"],
        status=status,
        json_valid=json_valid,
        schema_valid=schema_valid,
        semantic_valid=semantic_valid,
        predicted=pred,
        error_stage=stage,
        error_code=code,
        exact_match=exact,
        action_match=action_match,
        reason_match=reason_match,
        operation_match=op_match,
        slots=_slots(gold, pred),
        validator_accepted_plan=accepted_plan,
        wrong_executable=accepted_plan and not exact,
        elapsed_ms=float(elapsed) if isinstance(elapsed, (int, float)) else None,
    )


def _rate(num: int, den: int) -> dict[str, Any]:
    return {"count": num, "n": den, "rate": (num / den) if den else None}


def _prf(golds: list[str], preds: list[str | None], label: str) -> dict[str, Any]:
    tp = sum(1 for g, p in zip(golds, preds, strict=True) if g == label and p == label)
    fp = sum(1 for g, p in zip(golds, preds, strict=True) if g != label and p == label)
    fn = sum(1 for g, p in zip(golds, preds, strict=True) if g == label and p != label)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else (0.0 if precision is not None and recall is not None else None)
    )
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def summarize(
    examples: Sequence[dict[str, Any]],
    scored: Sequence[Scored],
    *,
    cluster_key: str = "lineage_id",
    bootstrap_reps: int = 2000,
) -> dict[str, Any]:
    """Aggregate metrics for one system over one example set (full denominator)."""
    n = len(examples)
    if n != len(scored):
        raise ValueError("one score per example is required")
    golds = [e["gold"] for e in examples]
    g_action = [g["action"] for g in golds]
    p_action = [s.predicted["action"] if s.predicted else None for s in scored]
    em = [s.exact_match for s in scored]
    out: dict[str, Any] = {"n": n}
    out["primary_semantic_exact_match"] = {
        **_rate(sum(em), n),
        "wilson95_iid_illustrative": wilson(sum(em), n),
        "cluster_bootstrap": cluster_bootstrap(
            [float(x) for x in em], [e[cluster_key] for e in examples], reps=bootstrap_reps
        ),
    }
    out["status"] = dict(sorted(Counter(s.status for s in scored).items()))
    out["json_valid"] = _rate(sum(s.json_valid for s in scored), n)
    out["candidate_schema_valid"] = _rate(sum(s.schema_valid for s in scored), n)
    out["semantic_valid"] = _rate(sum(s.semantic_valid for s in scored), n)
    out["error_codes"] = dict(
        sorted(Counter(f"{s.error_stage}/{s.error_code}" for s in scored if s.error_code).items())
    )
    out["action_accuracy"] = _rate(sum(s.action_match for s in scored), n)
    per_class = {a: _prf(g_action, p_action, a) for a in ("PLAN", "CLARIFY", "REJECT")}
    f1s = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    out["action_per_class"] = per_class
    out["action_macro_f1"] = sum(f1s) / len(f1s) if f1s else None
    gold_plan = [i for i, a in enumerate(g_action) if a == "PLAN"]
    gold_clarify = [i for i, a in enumerate(g_action) if a == "CLARIFY"]
    gold_reject = [i for i, a in enumerate(g_action) if a == "REJECT"]
    out["plan_operation_accuracy"] = _rate(
        sum(bool(scored[i].operation_match) for i in gold_plan), len(gold_plan)
    )
    slot_totals: dict[str, list[bool]] = defaultdict(list)
    for i in gold_plan:
        for slot, ok in scored[i].slots.items():
            slot_totals[f"{golds[i]['operation']}.{slot}"].append(ok)
    out["plan_slot_accuracy"] = {k: _rate(sum(v), len(v)) for k, v in sorted(slot_totals.items())}
    out["clarify_recall"] = _rate(
        sum(p_action[i] == "CLARIFY" for i in gold_clarify), len(gold_clarify)
    )
    out["clarify_reason_accuracy"] = _rate(
        sum(bool(scored[i].reason_match) for i in gold_clarify), len(gold_clarify)
    )
    out["reject_recall"] = _rate(
        sum(p_action[i] == "REJECT" for i in gold_reject), len(gold_reject)
    )
    out["reject_reason_accuracy"] = _rate(
        sum(bool(scored[i].reason_match) for i in gold_reject), len(gold_reject)
    )
    supported = gold_plan + gold_clarify
    out["false_reject_among_gold_plan"] = _rate(
        sum(p_action[i] == "REJECT" for i in gold_plan), len(gold_plan)
    )
    out["false_reject_among_supported"] = _rate(
        sum(p_action[i] == "REJECT" for i in supported), len(supported)
    )
    accepted = [i for i, s in enumerate(scored) if s.validator_accepted_plan]
    proposed = [i for i, s in enumerate(scored) if s.predicted and s.predicted["action"] == "PLAN"]
    out["plan_coverage_all"] = _rate(len(accepted), n)
    out["plan_coverage_gold_plan"] = _rate(
        sum(scored[i].validator_accepted_plan and scored[i].exact_match for i in gold_plan),
        len(gold_plan),
    )
    out["selective_risk_proposed_plans"] = _rate(
        sum(not scored[i].exact_match for i in proposed), len(proposed)
    )
    out["selective_risk_accepted_plans"] = _rate(
        sum(not scored[i].exact_match for i in accepted), len(accepted)
    )
    wrong = [i for i in accepted if scored[i].wrong_executable]
    out["wrong_executable_proposals"] = {
        "per_all": _rate(len(wrong), n),
        "per_accepted_plans": _rate(len(wrong), len(accepted)),
        "by_gold_action": dict(sorted(Counter(g_action[i] for i in wrong).items())),
    }
    times = sorted(s.elapsed_ms for s in scored if s.elapsed_ms is not None)
    if times:
        q = statistics.quantiles(times, n=4) if len(times) >= 2 else [times[0]] * 3
        out["latency_ms"] = {
            "n": len(times),
            "median": statistics.median(times),
            "q1": q[0],
            "q3": q[2],
            "note": "as recorded by the runner",
        }
    else:
        out["latency_ms"] = None
    return out


def slices(
    examples: Sequence[dict[str, Any]],
    scored: Sequence[Scored],
    keys: Iterable[str] = ("language", "stratum", "domain", "family"),
) -> dict[str, Any]:
    """Exact-match counts by slice (descriptive, secondary)."""
    out: dict[str, Any] = {}
    for key in keys:
        groups: dict[str, list[bool]] = defaultdict(list)
        for e, s in zip(examples, scored, strict=True):
            groups[str(e[key])].append(s.exact_match)
        out[key] = {k: _rate(sum(v), len(v)) for k, v in sorted(groups.items())}
    by_action: dict[str, list[bool]] = defaultdict(list)
    by_op: dict[str, list[bool]] = defaultdict(list)
    by_tag: dict[str, list[bool]] = defaultdict(list)
    for e, s in zip(examples, scored, strict=True):
        by_action[e["gold"]["action"]].append(s.exact_match)
        if e["gold"]["action"] == "PLAN":
            by_op[e["gold"]["operation"]].append(s.exact_match)
        for t in e["tags"]:
            by_tag[t].append(s.exact_match)
    out["gold_action"] = {k: _rate(sum(v), len(v)) for k, v in sorted(by_action.items())}
    out["gold_plan_operation"] = {k: _rate(sum(v), len(v)) for k, v in sorted(by_op.items())}
    out["tag"] = {k: _rate(sum(v), len(v)) for k, v in sorted(by_tag.items())}
    return out


def load_jsonl(path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
