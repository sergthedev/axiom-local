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

Version 0.2 (Phase 3A review of PR #8; 0.1 is preserved with EXP-008 at its commit):

* a match requires a **semantically valid** candidate (0.1 counted, for example, a ``CLARIFY`` with
  the right reason but a nonexistent column id as exact);
* action, operation, slot and reason metrics credit only semantically valid candidates, so an
  invalid prediction is wrong on every metric; ``raw_action_recognition`` separately reports the
  action of any schema-valid candidate, valid or not;
* per-class precision/recall/F1 use zero for an undefined ratio when the class has gold support, so
  a supported class with no predicted positives contributes F1 = 0 to the macro F1 (0.1 omitted
  it); classes without gold support are listed as excluded, never silently dropped;
* one prediction per example: duplicate or unknown prediction ids are an error
  (:func:`index_predictions`), never a silent overwrite.

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

SCORE_VERSION = "axbench.score/0.2"
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
    raw_action: str | None
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
    valid = pred if semantic_valid else None  # only a valid candidate earns credit (0.2)
    exact = valid is not None and semantic_key(valid) in gold_keys
    action_match = valid is not None and valid["action"] == gold["action"]
    reason_match = None
    if gold["action"] != "PLAN":
        reason_match = action_match and valid.get("reason") == gold.get("reason")
    op_match = None
    if gold["action"] == "PLAN":
        op_match = action_match and valid.get("operation") == gold["operation"]
    accepted_plan = valid is not None and valid["action"] == "PLAN"
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
        raw_action=pred["action"] if pred is not None else None,
        action_match=action_match,
        reason_match=reason_match,
        operation_match=op_match,
        slots=_slots(gold, valid),
        validator_accepted_plan=accepted_plan,
        wrong_executable=accepted_plan and not exact,
        elapsed_ms=float(elapsed) if isinstance(elapsed, (int, float)) else None,
    )


def _rate(num: int, den: int) -> dict[str, Any]:
    return {"count": num, "n": den, "rate": (num / den) if den else None}


def _prf(golds: list[str], preds: list[str | None], label: str) -> dict[str, Any]:
    """Per-class scores; for a class with gold support an undefined ratio counts as zero.

    A class without gold support has no defined recall or F1 (``None``) and is excluded from the
    macro average by :func:`summarize`, which lists it.
    """
    tp = sum(1 for g, p in zip(golds, preds, strict=True) if g == label and p == label)
    fp = sum(1 for g, p in zip(golds, preds, strict=True) if g != label and p == label)
    fn = sum(1 for g, p in zip(golds, preds, strict=True) if g == label and p != label)
    support = tp + fn
    if not support:
        return {
            "precision": (tp / (tp + fp)) if tp + fp else None,
            "recall": None,
            "f1": None,
            "support": 0,
            "predicted": tp + fp,
        }
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / support
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "predicted": tp + fp,
    }


ACTIONS = ("PLAN", "CLARIFY", "REJECT")


def index_predictions(
    examples: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """One prediction per example id, in example order; missing ones become ``error``.

    Raises ``ValueError`` on a duplicate prediction id, an id that is not in ``examples`` or a
    prediction without a string id: a prediction file must never be silently overwritten or
    partially ignored.
    """
    ids = [e["id"] for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate example ids")
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    unknown: list[str] = []
    known = set(ids)
    for p in predictions:
        pid = p.get("example_id") if isinstance(p, dict) else None
        if not isinstance(pid, str):
            raise ValueError("prediction without a string example_id")
        if pid in by_id:
            duplicates.append(pid)
        elif pid not in known:
            unknown.append(pid)
        else:
            by_id[pid] = p
    if duplicates or unknown:
        raise ValueError(
            f"prediction ids must match the examples one to one: {len(duplicates)} duplicate "
            f"({sorted(set(duplicates))[:5]}), {len(unknown)} unknown ({sorted(unknown)[:5]})"
        )
    ordered = [by_id.get(i, {"status": "error", "raw_output": None}) for i in ids]
    return ordered, sum(1 for i in ids if i not in by_id)


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
    # Only semantically valid candidates have a predicted action (0.2); see raw_action_recognition.
    p_action = [s.predicted["action"] if s.semantic_valid else None for s in scored]
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
    out["raw_action_recognition"] = {
        **_rate(sum(s.raw_action == g for s, g in zip(scored, g_action, strict=True)), n),
        "note": "action of any schema-valid candidate, including semantically invalid ones",
    }
    per_class = {a: _prf(g_action, p_action, a) for a in ACTIONS}
    included = [a for a in ACTIONS if per_class[a]["support"]]
    out["action_per_class"] = per_class
    out["action_macro_f1"] = (
        sum(per_class[a]["f1"] for a in included) / len(included) if included else None
    )
    out["action_macro_f1_classes"] = {
        "included": included,
        "excluded_no_gold_support": [a for a in ACTIONS if a not in included],
    }
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
    proposed = [i for i, s in enumerate(scored) if s.raw_action == "PLAN"]
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
