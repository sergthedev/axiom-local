"""EXP-TR03 data tooling: eligibility, audits, worklists, sizing and the human-gate checker.

Everything runs on the committed corpus or on synthetic ledgers; no model is loaded and no
system is scored. A synthetic ledger tests the gate *logic* only: it is not a review.
"""

from __future__ import annotations

import copy
import csv
import json
import re
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


def test_worklists_hold_no_decision_of_any_kind(outputs):
    for name, cols in (
        ("ROW_REVIEW_WORKLIST.csv", tr03.ROW_COLUMNS),
        ("TEMPLATE_REVIEW_WORKLIST.csv", tr03.TEMPLATE_COLUMNS),
    ):
        human = cols[cols.index("reviewer_id") :]
        for row in _read(outputs, name):
            assert all(row[c] == "" for c in human), name
    for row in _read(outputs, "RIGHTS_WORKLIST.csv"):
        assert row["owner_decision"] == "" and row["decided_by"] == ""
    roster = json.loads(outputs["HUMAN_ROSTER.template.json"])
    assert roster["designated_by"] == "" and roster["people"] == []  # the template stays empty
    assert roster["attestation"]["statement_accepted"] is False


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


def test_worklists_carry_frozen_workload_batches_and_a_planned_second_tenth(outputs):
    rows = _read(outputs, "ROW_REVIEW_WORKLIST.csv")
    assert {r["batch"][:6] for r in rows} == {"RW-tr-", "RW-en-"}
    assert (
        max(Counter(r["batch"] for r in rows).values())
        - min(Counter(r["batch"] for r in rows).values())
        <= 3
    )
    assert all(r["batch"].startswith("RW-" + r["language"]) for r in rows)
    templates = _read(outputs, "TEMPLATE_REVIEW_WORKLIST.csv")
    assert {t["batch"][:3] for t in templates} == {"TW-"}
    assert 0 < sum(t["second_review_planned"] == "yes" for t in templates) < 30


def test_batches_are_a_pure_function_of_the_keys():
    keys = [f"k{i}" for i in range(40)]
    a = tr03.assign_batches(keys, 4, "B")
    assert a == tr03.assign_batches(list(reversed(keys)), 4, "B")
    assert sorted(Counter(a.values()).values()) == [10, 10, 10, 10]


def test_module_reads_no_test_split_and_loads_no_model():
    for name in ("tr03.py", "tr03_gates.py"):
        source = (ROOT / "src/axbench_tr" / name).read_text("utf-8")
        for forbidden in ("test.jsonl", "mlx", "transformers", "torch", "load_model", "generate("):
            assert forbidden not in source.replace("mlx-lm", ""), (name, forbidden)


def test_review_guide_examples_are_real_and_outside_the_review_sample(corpus, outputs):
    guide = (
        ROOT / "docs/evidence/experiments/EXP-TR03-research-protocol/REVIEW_GUIDE.md"
    ).read_text("utf-8")
    sample = {r["record_id"] for r in _read(outputs, "ROW_REVIEW_WORKLIST.csv")}
    by_id = {r["id"]: r for r in corpus["elig_t"]}
    flat = " ".join(guide.split())  # the guide wraps lines
    cited = sorted(set(re.findall(r"`(t2-train-\d{5})`", guide)))
    assert len(cited) >= 9
    for rid in cited:
        assert rid in by_id, rid  # a real eligible train record
        assert rid not in sample, rid  # never a row a reviewer will judge
        quoted = " ".join(by_id[rid]["request"].split())
        assert quoted in flat, rid  # the quoted request is the record's own
    assert "An AI system may not judge" in guide


def test_rights_options_cover_every_rights_row_without_deciding(outputs):
    options = (
        ROOT / "docs/evidence/experiments/EXP-TR03-research-protocol/RIGHTS_DECISION_OPTIONS.md"
    ).read_text("utf-8")
    for row in _read(outputs, "RIGHTS_WORKLIST.csv"):
        key = row["source_id"].split(" ")[0]
        assert key in options or key == "external_adapted", key
    assert "not legal advice" in options and "no decision is recorded" in options
