"""EXP-TR04 synthetic held-out benchmark: disjointness, label audit, scorer self-check.

The benchmark is AI-authored, synthetic and unreviewed. These tests check its construction
guarantees and that scoring it works; they say nothing about model quality, and no model output
exists anywhere in this repository.
"""

from __future__ import annotations

import copy
import json
from collections import Counter

import pytest

from axbench_tr import tr04, tr04_cues
from axbench_tr import tr04_heldout as ho
from axbench_tr.synth import normalize_request

ROWS = ho.load_rows()
CORPUS = sum(tr04.load(), [])


def test_committed_files_equal_a_fresh_rebuild() -> None:
    assert ho.check() == []


def test_manifest_pins_the_files_and_the_audit_passed() -> None:
    manifest = json.loads((ho.HELDOUT_DIR / "HELDOUT_MANIFEST.json").read_text("utf-8"))
    assert set(manifest["files_sha256"]) == {"HELDOUT.jsonl", "HELDOUT_AUDIT.json"}
    audit = json.loads((ho.HELDOUT_DIR / "HELDOUT_AUDIT.json").read_text("utf-8"))
    assert audit["passed"] is True
    assert "not human-authored" in audit["status"] and "not human-validated" in audit["status"]
    assert audit["checks"]["sealed_test_referenced"] is False
    assert any("same AI tool" in text for text in audit["limitations"])


def test_every_row_says_it_is_ai_generated_and_unreviewed() -> None:
    assert len(ROWS) == 346 and len({r["id"] for r in ROWS}) == 346
    for r in ROWS:
        prov = r["provenance"]
        assert prov["origin"] == "synthetic_heldout"
        assert prov["request_authoring"] == "ai_generated_from_templates"
        assert prov["human_review"] == "none"
        assert prov["schema_origin"] == "fictional_ai_authored"
        assert r["split"] == "heldout" and r["stratum"] == "ai_authored_synthetic_heldout"
        assert r["schema_id"].startswith("heldout/") and r["lineage_id"].startswith("H4-")


def test_every_row_passes_the_same_row_audit_as_the_training_rows() -> None:
    for r in ROWS:
        assert tr04.row_problems(r, ho.HELDOUT_REGISTRY, ho.HELDOUT_SPLITS) == [], r["id"]
    # the audit is the real one: a corrupted held-out label is flagged
    bad = copy.deepcopy(next(r for r in ROWS if r["gold"]["action"] == "PLAN"))
    bad["gold"] = {"action": "REJECT", "reason": "out_of_scope"}
    assert tr04.row_problems(bad, ho.HELDOUT_REGISTRY, ho.HELDOUT_SPLITS)


def test_every_heldout_phrasing_passes_the_template_contract_in_both_languages() -> None:
    assert len(ho.HELDOUT_LINEAGES) == 55
    for lin in ho.HELDOUT_LINEAGES:
        assert tr04_cues.lineage_problems(lin) == [], lin.id


def test_the_benchmark_is_disjoint_from_the_whole_corpus() -> None:
    corpus_requests = {normalize_request(r["request"]) for r in CORPUS}
    assert not {normalize_request(r["request"]) for r in ROWS} & corpus_requests
    assert not {r["family"] for r in ROWS} & {r["family"] for r in CORPUS}
    assert not {r["schema_id"] for r in ROWS} & {r["schema_id"] for r in CORPUS}
    assert not {r["lineage_id"] for r in ROWS} & {r["lineage_id"] for r in CORPUS}
    assert not {r["id"] for r in ROWS} & {r["id"] for r in CORPUS}
    corpus_templates = {ho._slotless(t) for lin in ho.LINEAGES for t in (lin.tr, lin.en)}
    held_templates = {ho._slotless(t) for lin in ho.HELDOUT_LINEAGES for t in (lin.tr, lin.en)}
    assert not held_templates & corpus_templates


def test_disjointness_audit_would_catch_a_leak() -> None:
    leaked = copy.deepcopy(ROWS)
    donor = CORPUS[0]
    leaked[0]["request"] = donor["request"]
    report = ho.audit(leaked)
    assert report["passed"] is False
    assert report["checks"]["shared_normalized_requests_with_corpus"] >= 1
    fam = copy.deepcopy(ROWS)
    fam[1]["family"] = donor["family"]
    assert ho.audit(fam)["passed"] is False
    dup = copy.deepcopy(ROWS)
    dup[2]["id"] = dup[3]["id"]
    assert ho.audit(dup)["passed"] is False


def test_no_heldout_layout_resembles_a_corpus_layout() -> None:
    audit = json.loads((ho.HELDOUT_DIR / "HELDOUT_AUDIT.json").read_text("utf-8"))["checks"]
    assert audit["layouts_at_or_above_jaccard_limit"] == []
    assert audit["max_column_set_jaccard_with_any_corpus_layout"] < ho.COLUMN_JACCARD_MAX
    assert audit["overlap_with_axbench_v0.1_requests"]["exact_normalized_matches"] == 0


def test_the_benchmark_covers_every_action_frame_language_and_both_header_languages() -> None:
    assert {r["gold"]["action"] for r in ROWS} == {"PLAN", "CLARIFY", "REJECT"}
    assert len({r["frame"] for r in ROWS}) == 23  # every frame of the corpus
    assert {r["language"] for r in ROWS} == {"tr", "en"}
    assert {r["provenance"]["header_language"] for r in ROWS} == {"tr", "en"}
    assert len({r["schema_id"] for r in ROWS}) == 28
    counts = Counter(r["language"] for r in ROWS)
    assert counts["tr"] == counts["en"]


def test_the_heldout_rows_are_never_in_the_training_export_or_the_keep_lists() -> None:
    keep = tr04.kept_ids()
    ids = {r["id"] for r in ROWS}
    assert not ids & (set(keep["train"]) | set(keep["validation"]))
    assert not ids & {r["id"] for r in CORPUS}


# ------------------------------------------------------------------------------- scorer self-check


def test_gold_predictions_score_perfectly_and_a_bad_prediction_lowers_the_score() -> None:
    gold = ho.gold_predictions(ROWS)
    result = ho.score_predictions(ROWS, gold)
    summary = result["summary"]
    assert result["missing_predictions"] == 0
    assert summary["primary_semantic_exact_match"]["rate"] == 1.0
    assert summary["json_valid"]["rate"] == 1.0
    assert summary["wrong_executable_proposals"]["per_all"]["count"] == 0
    broken = [dict(p) for p in gold]
    broken[0]["raw_output"] = "not json"
    broken[1]["status"] = "timeout"
    worse = ho.score_predictions(ROWS, broken)["summary"]
    assert worse["primary_semantic_exact_match"]["count"] == 344
    assert worse["primary_semantic_exact_match"]["n"] == 346  # full denominator
    assert worse["status"] == {"ok": 345, "timeout": 1}


def test_a_wrong_but_valid_prediction_is_not_credited() -> None:
    gold = ho.gold_predictions(ROWS)
    plan_idx = next(i for i, r in enumerate(ROWS) if r["gold"]["action"] == "PLAN")
    wrong = [dict(p) for p in gold]
    wrong[plan_idx]["raw_output"] = json.dumps(
        {"action": "REJECT", "reason": "out_of_scope"}, separators=(",", ":")
    )
    summary = ho.score_predictions(ROWS, wrong)["summary"]
    assert summary["primary_semantic_exact_match"]["count"] == 345


def test_duplicate_unknown_and_missing_predictions_are_refused_or_reported() -> None:
    gold = ho.gold_predictions(ROWS)
    with pytest.raises(ValueError, match="duplicate"):
        ho.score_predictions(ROWS, [*gold, gold[0]])
    with pytest.raises(ValueError, match="unknown"):
        ho.score_predictions(ROWS, [*gold, {**gold[0], "example_id": "h4-99999"}])
    assert ho.score_predictions(ROWS, gold[:-2])["missing_predictions"] == 2


def test_slices_report_language_domain_and_frame() -> None:
    result = ho.score_predictions(ROWS, ho.gold_predictions(ROWS))
    assert {"language", "domain", "frame", "family"} <= set(result["slices"])
    assert result["summary"]["primary_semantic_exact_match"]["cluster_bootstrap"]["n_clusters"] > 1
