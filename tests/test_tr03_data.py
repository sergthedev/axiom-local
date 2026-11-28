"""EXP-TR03 data tooling: eligibility, audits, worklists, sizing and the human-gate checker.

Everything runs on the committed corpus or on synthetic ledgers; no model is loaded and no
system is scored. A synthetic ledger tests the gate *logic* only: it is not a review.
"""

from __future__ import annotations

import copy
import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from axbench_tr import tr03

ROOT = Path(__file__).resolve().parents[1]
PREP = tr03.PREP_DIR


@pytest.fixture(scope="module")
def corpus():
    train, valid, per, hashes = tr03.load()
    elig_t, why_t = tr03.split_eligible(train)
    elig_v, why_v = tr03.split_eligible(valid)
    return {
        "train": train,
        "valid": valid,
        "per": per,
        "elig_t": elig_t,
        "elig_v": elig_v,
        "why_t": why_t,
        "why_v": why_v,
    }


@pytest.fixture(scope="module")
def outputs():
    return tr03.build_outputs()


def _rec(origin="synthetic_template", source="SRC-O01", tags=()):
    return {"provenance": {"origin": origin, "schema_source_id": source}, "tags": list(tags)}


# ------------------------------------------------------------------------------- eligibility


def test_eligibility_rule_on_synthetic_records():
    assert tr03.is_eligible(_rec()) == (True, "")
    assert tr03.is_eligible(_rec(source="SRC-O02"))[0]
    assert tr03.is_eligible(_rec(source="SRC-O06"))[0]
    assert tr03.is_eligible(_rec(origin="external_adapted", source=None)) == (
        False,
        "origin:external_adapted",
    )
    for unverified in ("SRC-O03", "SRC-O04", "SRC-O05"):
        assert tr03.is_eligible(_rec(source=unverified)) == (False, f"schema_source:{unverified}")
    assert tr03.is_eligible(_rec(tags=["translation_unverified"])) == (
        False,
        "tag:translation_unverified",
    )


def test_eligible_set_matches_the_tr02_export_rule(corpus):
    """Same constants as the frozen export: nothing outside the documented sources is eligible."""
    from axbench_tr import export

    assert tr03.ELIGIBLE_SOURCES == export.RIGHTS_VERIFIED_SCHEMA_SOURCES
    assert tr03.ELIGIBLE_ORIGIN == export.CLEARED_ORIGIN
    for r in (*corpus["elig_t"], *corpus["elig_v"]):
        assert r["provenance"]["origin"] == "synthetic_template"
        assert r["provenance"]["schema_source_id"] in ("SRC-O01", "SRC-O02", "SRC-O06")
        assert r["provenance"]["request_authoring"] == "ai_generated_from_templates"
        assert "machine_translated" not in r["tags"]


def test_eligible_counts_and_exclusions_account_for_the_whole_corpus(corpus):
    assert (len(corpus["elig_t"]), len(corpus["elig_v"])) == (5091, 1274)
    assert len(corpus["elig_t"]) + sum(corpus["why_t"].values()) == len(corpus["train"])
    assert len(corpus["elig_v"]) + sum(corpus["why_v"].values()) == len(corpus["valid"])
    assert corpus["why_t"]["origin:external_adapted"] == 473
    assert {k for k in corpus["why_t"]} == {
        "origin:external_adapted",
        "schema_source:SRC-O03",
        "schema_source:SRC-O04",
        "schema_source:SRC-O05",
    }


def test_no_record_has_a_human_review_yet(corpus):
    for r in (*corpus["elig_t"], *corpus["elig_v"]):
        assert r["provenance"]["human_review"] == "none"
    inv = tr03.inventory()
    assert inv["human_reviewed_records"] == {"train": 0, "validation": 0}


def test_every_eligible_record_fits_the_sequence_cap(corpus):
    stats = tr03.token_stats(corpus["elig_t"], corpus["per"])
    assert stats["rows_over_2048"] == 0
    assert stats["full_sequence_tokens"]["max"] <= tr03.MAX_SEQ_LENGTH
    assert stats["rows_over_1024"] == 70
    assert stats["boundary_straddle_rows"] == 0


def test_padded_tokens_equals_the_frozen_tr02_rule():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("tr02_for_tr03", ROOT / "training/tr02/tr02.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tr02_for_tr03"] = mod
    spec.loader.exec_module(mod)
    for n in (1, 31, 32, 33, 483, 1024, 1342, 2047, 2048, 3000):
        assert tr03.padded_tokens(n) == mod.padded_tokens(n, 2048)


# --------------------------------------------------------------------------------- the audits


def test_split_independence_on_the_eligible_records(outputs):
    audit = json.loads(outputs["AUDIT.json"])
    split = audit["split_independence"]
    assert split["ids_unique"] is True
    assert set(split["shared_between_train_and_validation"].values()) == {0}
    near = split["lexical_near_duplicate_pairs_validation_vs_train"]
    assert near["char_4gram_jaccard_ge_0.8"] == 0
    v01 = split["overlap_with_axbench_v0.1_train_and_validation_requests"]
    for part in ("train", "validation"):
        assert v01[part]["exact_normalized_matches"] == 0
        assert v01[part]["near_duplicate_pairs"] == 0


def test_gold_labels_are_consistent_with_the_contract(outputs):
    gold = json.loads(outputs["AUDIT.json"])["gold_consistency"]
    assert gold["train"]["problems"] == {} and gold["validation"]["problems"] == {}
    assert gold["train"]["records_checked"] == 5091


def test_shortcut_audit_reports_no_label_conflict_by_column_name(outputs):
    stats = json.loads(outputs["AUDIT.json"])["shortcut_statistics"]
    for part in ("train", "validation"):
        assert stats[part]["same_request_and_layout_with_different_gold_by_column_name"] == 0
        assert 0.0 < stats[part]["largest_action_share"] < 1.0


def test_shortcut_audit_detects_a_planted_label_conflict(corpus):
    rows = copy.deepcopy(corpus["elig_t"][:400])
    target = next(
        r
        for r in rows
        if r["frame"] == "compare"
        and sum(c["type"] == "numeric" for c in r["schema"]["columns"]) >= 2
    )
    twin = copy.deepcopy(target)
    twin["id"] = "planted"
    # same request, same layout, but the label points at a different measure column
    numeric = [c["id"] for c in twin["schema"]["columns"] if c["type"] == "numeric"]
    twin["gold"]["measure"] = next(c for c in numeric if c != target["gold"]["measure"])
    base = tr03.shortcut_audit(rows)
    planted = tr03.shortcut_audit([*rows, twin])
    assert base["same_request_and_layout_with_different_gold_by_column_name"] == 0
    assert planted["same_request_and_layout_with_different_gold_by_column_name"] == 1


def test_heldout_candidates_never_list_a_layout_the_training_rows_show(corpus):
    cands = tr03.heldout_candidates(
        [*corpus["train"], *corpus["valid"]], {r["schema_id"] for r in corpus["elig_t"]}
    )
    work = tr03.heldout_worklist(cands)
    shown = {r["schema_id"] for r in corpus["elig_t"]}
    assert work and all(w["table_id"] not in shown for w in work)
    assert {c["source_id"] for c in cands} <= set(tr03.ELIGIBLE_SOURCES)


def test_table_capabilities():
    cols = [{"kind": "numeric"}, {"kind": "numeric"}, {"kind": "text"}]
    cap = tr03.table_capabilities(cols)
    assert cap["supports_relate"] and cap["supports_compare"] and cap["supports_cluster"]
    only_text = tr03.table_capabilities([{"kind": "text"}])
    assert only_text["supports_inspect"] and not only_text["supports_compare"]


# ----------------------------------------------------------------------------------- sizing


def test_zero_failure_sample_sizes_have_known_answers():
    assert tr03.zero_failure_n(0.05) == 59  # the 'rule of three' neighbourhood: 0.95**59 < 0.05
    assert 0.95**59 <= 0.05 < 0.95**58
    assert tr03.zero_failure_n(0.10) == 29
    assert 0.90**29 <= 0.05 < 0.90**28
    assert tr03.TR_ROWS_PER_STRATUM == tr03.zero_failure_n(0.05)


def test_wilson_half_width_known_answer():
    assert tr03.wilson_half_width(0.5, 100) == pytest.approx(0.0962, abs=1e-4)
    assert tr03.wilson_half_width(0.5, 300) < tr03.wilson_half_width(0.5, 100)


# --------------------------------------------------------------------------------- worklists


def test_committed_prep_files_equal_a_fresh_rebuild():
    assert tr03.check_outputs() == []
    manifest = json.loads((PREP / "MANIFEST.json").read_text("utf-8"))
    assert set(manifest["files_sha256"]) == {
        "INVENTORY.json",
        "AUDIT.json",
        "SIZING.json",
        "TEMPLATE_REVIEW_WORKLIST.csv",
        "ROW_REVIEW_WORKLIST.csv",
        "RIGHTS_WORKLIST.csv",
        "HELDOUT_SCHEMA_CANDIDATES.csv",
        "HUMAN_ROSTER.template.json",
    }


def test_build_is_deterministic(outputs):
    again = tr03.build_outputs()
    assert again == outputs


def _read(outputs, name):
    return list(csv.DictReader(outputs[name].decode("utf-8").splitlines()))


HUMAN_ROW_COLUMNS = (
    "reviewer_id",
    "reviewer_type",
    "request_natural",
    "gold_correct",
    "serious_error",
    "decision",
    "notes",
    "second_reviewer_id",
    "second_reviewer_type",
    "second_decision",
    "adjudicated_decision",
)


def test_worklists_hold_no_decision_of_any_kind(outputs):
    for name in ("ROW_REVIEW_WORKLIST.csv", "TEMPLATE_REVIEW_WORKLIST.csv"):
        for row in _read(outputs, name):
            assert all(
                row[c] == "" for c in row if c in HUMAN_ROW_COLUMNS or c.endswith("_natural")
            )
            assert row["reviewer_type"] == ""
    for row in _read(outputs, "RIGHTS_WORKLIST.csv"):
        assert row["owner_decision"] == "" and row["decided_by"] == ""
    roster = json.loads(outputs["HUMAN_ROSTER.template.json"])
    assert roster["designated_by"] == ""
    assert tr03.roster_problems(roster)  # the template is not a valid roster


def test_row_worklist_strata_and_scope(outputs, corpus):
    rows = _read(outputs, "ROW_REVIEW_WORKLIST.csv")
    assert len(rows) == 7 * tr03.TR_ROWS_PER_STRATUM + 7 * tr03.EN_ROWS_PER_STRATUM
    by_stratum = Counter((r["language"], tr03.frame_class(r["frame"])) for r in rows)
    for cls in tr03.FRAME_CLASSES:
        assert by_stratum[("tr", cls)] == tr03.TR_ROWS_PER_STRATUM
        assert by_stratum[("en", cls)] == tr03.EN_ROWS_PER_STRATUM
    ids = {r["id"]: r for r in corpus["elig_t"]}
    assert len({r["record_id"] for r in rows}) == len(rows)
    assert all(r["record_id"] in ids for r in rows)  # eligible train rows only
    assert all(
        ids[r["record_id"]]["provenance"]["schema_source_id"] in tr03.ELIGIBLE_SOURCES for r in rows
    )
    high = [r for r in rows if r["risk"] == "high"]
    assert high and all(r["frame"] in tr03.HIGH_RISK_FRAMES for r in high)
    planned = sum(r["second_review_planned"] == "yes" for r in rows)
    assert 0.03 < planned / len(rows) < 0.20  # about a tenth, by hash


def test_template_worklist_covers_every_eligible_template(outputs, corpus):
    rows = _read(outputs, "TEMPLATE_REVIEW_WORKLIST.csv")
    expected = {r["lineage_id"] for r in (*corpus["elig_t"], *corpus["elig_v"])}
    assert {r["lineage_id"] for r in rows} == expected
    assert len(rows) == 122
    assert all(r["authored_by"] == "ai_tool" for r in rows)


def test_rights_worklist_names_the_unverified_sources_as_excluded(outputs):
    rows = {r["source_id"]: r for r in _read(outputs, "RIGHTS_WORKLIST.csv")}
    for sid in ("SRC-O03", "SRC-O04", "SRC-O05"):
        assert "excluded" in rows[sid]["status_in_repository"]
    for sid in tr03.ELIGIBLE_SOURCES:
        assert "not legal advice" in rows[sid]["status_in_repository"]
        assert rows[sid]["open_caveat"]


def test_inventory_templates_are_ai_authored_and_counted(outputs):
    inv = json.loads(outputs["INVENTORY.json"])
    assert len(inv["templates"]) == 122
    assert sum(t["records"] for t in inv["templates"]) == 5091 + 1274
    assert all("no human" in t["authored_by"] for t in inv["templates"])
    assert "legal clearance" in inv["eligibility_rule"]["meaning"]
    assert {a["source_id"] for a in inv["source_attribution"]} == set(tr03.ELIGIBLE_SOURCES)


# ----------------------------------------------------------------------------------- the gates


ROSTER = {
    "schema": "axbench-tr.tr03-roster/1",
    "designated_by": "owner",
    "designated_on": "2026-12-01",
    "people": [
        {"id": "R1", "type": "human", "native_turkish": True, "roles": ["reviewer"]},
        {"id": "R2", "type": "human", "native_turkish": False, "roles": ["reviewer"]},
    ],
}


def _filled_row_ledger(outputs, *, second=True):
    rows = _read(outputs, "ROW_REVIEW_WORKLIST.csv")
    for r in rows:
        r.update(
            reviewer_id="R1",
            reviewer_type="human",
            request_natural="yes",
            gold_correct="yes",
            serious_error="no",
            decision="keep",
        )
        if second and r["second_review_planned"] == "yes":
            r.update(second_reviewer_id="R2", second_reviewer_type="human", second_decision="keep")
    return rows


def test_roster_problems():
    assert tr03.roster_problems(None) == ["roster_missing"]
    assert tr03.roster_problems(ROSTER) == []
    solo = copy.deepcopy(ROSTER)
    solo["people"] = solo["people"][:1]
    assert "fewer_than_two_human_reviewers" in tr03.roster_problems(solo)
    no_tr = copy.deepcopy(ROSTER)
    no_tr["people"][0]["native_turkish"] = False
    assert "no_native_turkish_reviewer" in tr03.roster_problems(no_tr)
    ai = copy.deepcopy(ROSTER)
    ai["people"].append({"id": "A1", "type": "ai", "roles": ["reviewer"]})
    assert "non_human_person_listed" in tr03.roster_problems(ai)
    undesignated = {**ROSTER, "designated_by": ""}
    assert "designation_missing" in tr03.roster_problems(undesignated)


def test_a_complete_synthetic_ledger_passes_the_row_gates(outputs):
    report = tr03.row_gate_report(_filled_row_ledger(outputs), ROSTER)
    assert report["all_row_gates_passed"], report["gates"]
    assert report["rows_refused_not_a_human_on_the_roster"] == 0
    assert "not a research-quality claim" in report["claim"]


def test_ai_reviewed_rows_are_refused_as_decisions(outputs):
    rows = _filled_row_ledger(outputs)
    for r in rows:
        r["reviewer_type"] = "ai"
    report = tr03.row_gate_report(rows, ROSTER)
    assert report["rows_counted_as_human_decisions"] == 0
    assert report["rows_refused_not_a_human_on_the_roster"] == len(rows)
    assert not report["all_row_gates_passed"]
    stranger = _filled_row_ledger(outputs)
    for r in stranger:
        r["reviewer_id"] = "R9"  # not on the roster
    assert tr03.row_gate_report(stranger, ROSTER)["rows_counted_as_human_decisions"] == 0


def test_row_gates_fail_on_each_predeclared_condition(outputs):
    def first_tr(rows, cls):
        return [r for r in rows if r["language"] == "tr" and tr03.frame_class(r["frame"]) == cls]

    # a stratum below 95% gold correct (4 of 59 wrong is 93.2%)
    rows = _filled_row_ledger(outputs)
    for r in first_tr(rows, "compare")[:4]:
        r.update(gold_correct="no", decision="drop")
        r.update(second_reviewer_id="R2", second_reviewer_type="human", second_decision="drop")
    report = tr03.row_gate_report(rows, ROSTER)
    assert not report["gates"]["G3c_every_turkish_stratum_has_min_n_and_95pct"]

    # a serious error that is kept
    rows = _filled_row_ledger(outputs)
    first_tr(rows, "reject")[0].update(serious_error="yes")
    assert not tr03.row_gate_report(rows, ROSTER)["gates"]["G3d_zero_serious_errors_kept"]

    # a 'no' without a second review
    rows = _filled_row_ledger(outputs)
    target = first_tr(rows, "relate")[0]
    target.update(
        gold_correct="no", decision="drop", second_reviewer_id="", second_reviewer_type=""
    )
    assert not tr03.row_gate_report(rows, ROSTER)["gates"]["G3e_second_review_complete"]

    # a disagreement without adjudication
    rows = _filled_row_ledger(outputs)
    planned = next(r for r in rows if r["second_review_planned"] == "yes")
    planned.update(second_decision="drop", adjudicated_decision="")
    assert not tr03.row_gate_report(rows, ROSTER)["gates"]["G3e_second_review_complete"]
    planned["adjudicated_decision"] = "keep"
    assert tr03.row_gate_report(rows, ROSTER)["gates"]["G3e_second_review_complete"]

    # too few rows in a stratum
    rows = _filled_row_ledger(outputs)
    drop = first_tr(rows, "cluster")[0]["sample_id"]
    rows = [r for r in rows if r["sample_id"] != drop]
    assert not tr03.row_gate_report(rows, ROSTER)["gates"][
        "G3c_every_turkish_stratum_has_min_n_and_95pct"
    ]

    # an empty ledger passes nothing
    assert not tr03.row_gate_report([], ROSTER)["all_row_gates_passed"]


def test_template_gate_and_research_row_rule(outputs, corpus):
    templates = _read(outputs, "TEMPLATE_REVIEW_WORKLIST.csv")
    expected = [t["lineage_id"] for t in templates]
    for t in templates:
        t.update(reviewer_id="R1", reviewer_type="human", decision="keep", serious_error="no")
    ok = tr03.template_gate_report(templates, ROSTER, expected)
    assert ok["gate_G2_every_template_has_a_human_decision"]
    assert ok["templates_kept"] == len(expected)

    templates[0].update(decision="fix")  # fix removes the template: nothing is relabelled
    templates[1].update(serious_error="yes")
    templates[2].update(reviewer_type="ai")
    rep = tr03.template_gate_report(templates, ROSTER, expected)
    assert not rep["gate_G2_every_template_has_a_human_decision"]
    assert rep["templates_missing_a_human_decision"] == 1
    assert templates[0]["lineage_id"] not in rep["kept_lineage_ids"]
    assert templates[1]["lineage_id"] not in rep["kept_lineage_ids"]

    kept = {t["lineage_id"] for t in templates[3:] if t["reviewer_type"] == "human"}
    row_ledger = _filled_row_ledger(outputs)
    victim = next(r for r in row_ledger if r["lineage_id"] in kept)
    victim.update(decision="fix")
    ai_drop = next(r for r in row_ledger if r["lineage_id"] in kept and r is not victim)
    ai_drop.update(decision="drop", reviewer_type="ai")  # an AI 'drop' is not a human decision
    ids = tr03.research_training_ids(corpus["elig_t"], kept, row_ledger, ROSTER)
    assert victim["record_id"] not in ids
    assert ai_drop["record_id"] in ids
    assert all(r["lineage_id"] in kept for r in corpus["elig_t"] if r["id"] in set(ids))
    assert ids == sorted(ids)


def test_gates_report_is_closed_today_and_opens_only_with_every_file(tmp_path, outputs):
    today = tr03.gates_report()  # the repository has no ledgers directory
    assert today["research_training_open"] is False
    assert today["G1_roster_designated_by_owner"] is False
    assert "roster_missing" in today["G1_problems"]
    assert not (PREP / "ledgers").exists()

    led = tmp_path / "ledgers"
    led.mkdir()
    (led / "HUMAN_ROSTER.json").write_text(json.dumps(ROSTER), "utf-8")
    templates = _read(outputs, "TEMPLATE_REVIEW_WORKLIST.csv")
    for t in templates:
        t.update(reviewer_id="R1", reviewer_type="human", decision="keep", serious_error="no")
    for name, rows, cols in (
        ("TEMPLATE_REVIEW_RESULTS.csv", templates, tr03.TEMPLATE_COLUMNS),
        ("ROW_REVIEW_RESULTS.csv", _filled_row_ledger(outputs), tr03.ROW_COLUMNS),
    ):
        with (led / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cols), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    partial = tr03.gates_report(tmp_path)
    assert partial["G2_templates"]["gate_G2_every_template_has_a_human_decision"]
    assert partial["G3_rows"]["all_row_gates_passed"]
    assert (
        partial["research_training_open"] is False
    )  # rights, held-out, preregistration, clearance
    for name in (
        "RIGHTS_DECISIONS.json",
        "HELDOUT_FROZEN.json",
        "PREREGISTRATION.json",
        "CLEARANCE.json",
    ):
        (led / name).write_text("{}", "utf-8")
    assert tr03.gates_report(tmp_path)["research_training_open"] is True


def test_module_reads_no_test_split_and_loads_no_model():
    source = (ROOT / "src/axbench_tr/tr03.py").read_text("utf-8")
    for forbidden in ("test.jsonl", "mlx", "transformers", "torch", "load_model", "generate("):
        assert forbidden not in source.replace("mlx-lm", "")
