"""Phase 3A review regressions (PR #8 review of 51a6a3a), scoring blocker 2: ``axbench.score/0.2``.

Each case reproduces a defect observed on the reviewed head with scorer 0.1 (recorded in the
development log) and fixes the corrected behaviour.
"""

from __future__ import annotations

import json

import pytest

from axbench.evaluate import HARNESS_VERSION, evaluate, guard_layer
from axbench.score import SCORE_VERSION, index_predictions, score_one, summarize

V = "axiom.candidate/1"
SCHEMA = {
    "version": "axiom.schema-view/1",
    "columns": [
        {"id": "c1", "name": "income", "type": "numeric"},
        {"id": "c2", "name": "age", "type": "numeric"},
        {"id": "c3", "name": "region", "type": "text"},
    ],
}


def _ex(i: str, gold: dict, variants: list | None = None) -> dict:
    return {
        "id": i,
        "lineage_id": f"L-{i}",
        "accepted_variants": variants or [],
        "tags": [],
        "language": "en",
        "stratum": "s",
        "domain": "d",
        "family": "f",
        "request": "a request",
        "gold": {"version": V, **gold},
        "schema": SCHEMA,
    }


def _pred(i: str, cand) -> dict:
    raw = cand if isinstance(cand, str) else json.dumps({"version": V, **cand})
    return {"example_id": i, "status": "ok", "raw_output": raw}


RELATE = {"action": "PLAN", "operation": "relate", "x": "c1", "y": "c2", "method": "pearson"}
CLARIFY = {"action": "CLARIFY", "reason": "ambiguous_column", "columns": ["c1", "c2"]}
REJECT = {"action": "REJECT", "reason": "forecasting"}


def test_versions_bumped() -> None:
    assert SCORE_VERSION == "axbench.score/0.2"
    assert HARNESS_VERSION == "axbench.evaluate/0.2"


def test_invalid_clarify_with_right_reason_is_not_exact() -> None:
    ex = _ex("x", CLARIFY)
    s = score_one(ex, _pred("x", {**CLARIFY, "columns": ["c9"]}))
    assert s.schema_valid and not s.semantic_valid
    assert s.error_code == "unknown_column_id"
    assert not s.exact_match  # 0.1: True
    assert not s.action_match and s.reason_match is False
    assert s.raw_action == "CLARIFY"  # still visible as raw recognition
    assert score_one(ex, _pred("x", CLARIFY)).exact_match


def test_invalid_plan_gets_no_operation_or_slot_credit() -> None:
    ex = _ex("x", RELATE)
    s = score_one(ex, _pred("x", {**RELATE, "y": "c3"}))  # text column: semantically invalid
    assert not s.semantic_valid
    assert s.operation_match is False
    assert s.slots == {"columns": False, "method": False}
    assert not s.validator_accepted_plan and not s.wrong_executable
    summary = summarize([ex], [s], bootstrap_reps=10)
    assert summary["action_accuracy"]["count"] == 0
    assert summary["raw_action_recognition"]["count"] == 1
    assert summary["plan_operation_accuracy"]["count"] == 0
    assert summary["selective_risk_proposed_plans"] == {"count": 1, "n": 1, "rate": 1.0}


def test_macro_f1_keeps_supported_class_without_predictions() -> None:
    exs = [_ex("a", RELATE), _ex("b", CLARIFY), _ex("c", REJECT)]
    scored = [score_one(e, _pred(e["id"], RELATE)) for e in exs]
    s = summarize(exs, scored, bootstrap_reps=10)
    per = s["action_per_class"]
    assert per["REJECT"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 1,
        "predicted": 0,
    }  # 0.1: f1 None, dropped from the macro average
    assert per["PLAN"]["f1"] == pytest.approx(0.5)
    assert s["action_macro_f1"] == pytest.approx(0.5 / 3)
    assert s["action_macro_f1_classes"]["included"] == ["PLAN", "CLARIFY", "REJECT"]


def test_class_without_gold_support_is_listed_as_excluded() -> None:
    exs = [_ex("a", RELATE), _ex("b", RELATE)]
    scored = [score_one(exs[0], _pred("a", RELATE)), score_one(exs[1], _pred("b", REJECT))]
    s = summarize(exs, scored, bootstrap_reps=10)
    assert s["action_per_class"]["REJECT"]["f1"] is None
    assert s["action_per_class"]["REJECT"]["predicted"] == 1
    assert s["action_macro_f1_classes"]["excluded_no_gold_support"] == ["CLARIFY", "REJECT"]
    assert s["action_macro_f1"] == pytest.approx(s["action_per_class"]["PLAN"]["f1"])


def test_duplicate_prediction_ids_are_rejected() -> None:
    ex = _ex("x", RELATE)
    with pytest.raises(ValueError, match="1 duplicate"):
        evaluate([ex], [_pred("x", "garbage"), _pred("x", RELATE)], system="t", split="s")


def test_unknown_prediction_ids_are_rejected() -> None:
    ex = _ex("x", RELATE)
    with pytest.raises(ValueError, match="1 unknown"):
        evaluate([ex], [_pred("x", RELATE), _pred("zzz", RELATE)], system="t", split="s")
    with pytest.raises(ValueError, match="string example_id"):
        index_predictions([ex], [{"status": "ok"}])


def test_missing_predictions_still_count_as_errors() -> None:
    exs = [_ex("a", RELATE), _ex("b", RELATE)]
    ordered, missing = index_predictions(exs, [_pred("b", RELATE)])
    assert missing == 1 and ordered[0]["status"] == "error"
    report = evaluate(exs, [_pred("b", RELATE)], system="t", split="s", bootstrap_reps=10)
    assert report["raw_candidate"]["primary_semantic_exact_match"]["count"] == 1
    assert report["raw_candidate"]["n"] == 2


def test_after_guard_counts_accepted_variant_as_correct() -> None:
    variant = {"version": V, **RELATE, "method": "spearman"}
    ex = _ex("x", RELATE, variants=[variant])
    pred = _pred("x", {**RELATE, "method": "spearman"})
    assert score_one(ex, pred).exact_match
    layer = guard_layer([ex], [pred])
    assert layer["awaiting_confirmation_correct"] == 1  # 0.1 harness: counted as wrong
    assert layer["awaiting_confirmation_wrong"] == 0
    assert layer["unauthorized_dispatch_count"] == 0


def test_guard_layer_uses_request_contexts_and_failure_statuses() -> None:
    exs = [_ex("a", RELATE), _ex("b", RELATE)]
    preds = [
        {"example_id": "a", "status": "timeout", "raw_output": None},
        _pred("b", RELATE),
    ]
    layer = guard_layer(exs, preds)
    assert layer["guard_status"] == {"awaiting_confirmation": 1, "router_timeout": 1}
    assert layer["unauthorized_dispatch_count"] == 0
