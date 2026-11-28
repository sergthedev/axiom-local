"""EXP-TR04 data side: rights screen, template contract, row audit, mutations, kept-id invariants.

The audit is automated and restates the generator's rules. These tests check that it does what its
reports say, that its committed outputs equal a fresh rebuild, and that deliberately wrong inputs
are flagged; they do not make any row human-reviewed.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import hashlib
import json
import re

import pytest

from axbench_tr import tr04, tr04_cues, tr04_rights
from axbench_tr.templates import LINEAGES

KEEP = tr04.kept_ids()
TRAIN, VALID = tr04.load()
BY_ID = {r["id"]: r for r in TRAIN + VALID}


# ----------------------------------------------------------------------------- committed outputs


def test_committed_audit_outputs_equal_a_fresh_rebuild() -> None:
    assert tr04.check() == []


def test_the_committed_manifest_pins_every_generated_file_and_input() -> None:
    manifest = json.loads((tr04.TR04_DIR / "MANIFEST.json").read_text("utf-8"))
    assert set(manifest["files_sha256"]) >= {
        "AUDIT.json",
        "EXCLUSIONS.csv",
        "KEEP_TRAIN.txt",
        "KEEP_VALIDATION.txt",
        "MUTATION_REPORT.json",
        "TEMPLATE_CONTRACT.json",
    }
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((tr04.TR04_DIR / name).read_bytes()).hexdigest() == digest


def test_audit_reports_say_automated_not_human_reviewed() -> None:
    audit = json.loads((tr04.TR04_DIR / "AUDIT.json").read_text("utf-8"))
    assert "not human-reviewed" in audit["status"] and "not human-validated" in audit["status"]
    assert any("is human-reviewed" in text for text in audit["limitations"])
    assert any("restates the generator" in text for text in audit["limitations"])
    mutation = json.loads((tr04.TR04_DIR / "MUTATION_REPORT.json").read_text("utf-8"))
    assert "not accuracy on real label errors" in mutation["meaning"]


# ----------------------------------------------------------------------------------- rights


def test_the_text_screen_flags_restriction_permission_and_licence_wording() -> None:
    assert tr04_rights.scan_text("Veri seti İzmir ili sınırları içerisindedir.") == []
    assert tr04_rights.scan_text("Open data about bus stops, updated monthly.") == []
    assert "commercial_wording" in tr04_rights.scan_text("Ticari amaçla kullanılamaz")
    assert "restriction_wording" in tr04_rights.scan_text("Ticari amaçla kullanılamaz")
    assert "permission_or_consent" in tr04_rights.scan_text("Kullanım için izin gerekir")
    assert "copyright_reserved" in tr04_rights.scan_text("© Example. All rights reserved")
    assert "personal_data" in tr04_rights.scan_text("Contains personal data (KVKK)")
    assert "other_licence" in tr04_rights.scan_text("Licensed CC BY-NC 4.0")
    assert "third_party_ownership" in tr04_rights.scan_text("Veri sahibi başka kurumdur")


def test_committed_package_evidence_is_consistent_and_has_no_contact_data() -> None:
    doc = tr04_rights.load_izmir()
    assert tr04_rights.check_evidence(doc) == []
    text = json.dumps(doc, ensure_ascii=False)
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)  # no e-mail address
    assert not re.search(r"/Users/|/home/|[A-Za-z]:\\\\", text)  # no local path
    for rec in doc["packages"].values():
        assert tr04_rights.scan_text(tr04_rights.package_text(rec)) == rec["flags"]


def test_every_izmir_table_of_the_corpus_has_a_package_decision() -> None:
    doc = tr04_rights.load_izmir()
    status = tr04_rights.izmir_status(doc)
    families = {
        r["family"] for r in TRAIN + VALID if r["provenance"].get("schema_source_id") == "SRC-O06"
    }
    assert families <= set(status)
    assert sum(not s["usable"] for s in status.values()) >= 1  # the screen does exclude something


def test_a_tampered_package_is_unusable_and_stale_flags_are_caught() -> None:
    doc = copy.deepcopy(tr04_rights.load_izmir())
    name = next(n for n, r in doc["packages"].items() if not r["flags"])
    table = doc["packages"][name]["tables"][0]
    group = tr04_rights.table_group(table)
    assert tr04_rights.izmir_status(doc)[group]["usable"] is True
    doc["packages"][name]["notes"] = (
        doc["packages"][name]["notes"] or ""
    ) + " Ticari kullanım yasaktır."
    status = tr04_rights.izmir_status(doc)[group]
    assert status["usable"] is False
    assert "stored_flags_differ_from_rescan" in status["problems"]
    assert any(p.startswith("flag:") for p in status["problems"])
    doc = copy.deepcopy(tr04_rights.load_izmir())
    doc["packages"][name]["license_id"] = "other-licence"
    assert "licence:other-licence" in tr04_rights.izmir_status(doc)[group]["problems"]


def _rec(**prov) -> dict:
    base = {"origin": "synthetic_template", "schema_source_id": "SRC-O06"}
    return {"provenance": base | prov, "tags": [], "family": "izmir/example"}


def test_rights_decision_excludes_every_source_except_a_clean_izmir_package() -> None:
    ok = {"izmir/example": {"usable": True, "problems": []}}
    assert tr04.rights_decision(_rec(), ok) == ""
    assert tr04.rights_decision(_rec(), {}) == "izmir_group_not_screened"
    bad = {"izmir/example": {"usable": False, "problems": ["flag:commercial_wording"]}}
    assert tr04.rights_decision(_rec(), bad).startswith("izmir_package_not_usable")
    for source in ("SRC-O01", "SRC-O02", "SRC-O03", "SRC-O04", "SRC-O05"):
        assert tr04.rights_decision(_rec(schema_source_id=source), ok).startswith(source)
    assert tr04.rights_decision(_rec(schema_source_id="SRC-X"), ok).startswith(
        "schema_source_unlisted"
    )
    assert tr04.rights_decision(_rec(origin="external_adapted"), ok).startswith("origin:")
    rec = _rec()
    rec["tags"] = ["translation_unverified"]
    assert tr04.rights_decision(rec, ok) == "tag:translation_unverified"


def test_unlisted_and_unverified_sources_default_to_exclusion() -> None:
    assert tr04.SOURCE_DECISIONS["SRC-O06"] == ""
    assert all(v for k, v in tr04.SOURCE_DECISIONS.items() if k != "SRC-O06")


# ---------------------------------------------------------------------- kept rows and exclusions


def test_every_kept_row_is_a_clean_izmir_template_row_that_passes_the_row_audit() -> None:
    izmir = tr04_rights.izmir_status(tr04_rights.load_izmir())
    for name, ids in KEEP.items():
        assert len(ids) == len(set(ids))
        for rid in ids:
            rec = BY_ID[rid]
            assert rec["provenance"]["origin"] == "synthetic_template"
            assert rec["provenance"]["schema_source_id"] == "SRC-O06"
            assert rec["provenance"]["human_review"] == "none"
            assert tr04.rights_decision(rec, izmir) == ""
            assert tr04.row_problems(rec) == [], rid
        assert all(BY_ID[i]["split"] == name for i in ids)


def test_kept_splits_are_independent_and_the_train_count_is_a_multiple_of_four() -> None:
    train = [BY_ID[i] for i in KEEP["train"]]
    valid = [BY_ID[i] for i in KEEP["validation"]]
    assert len(train) % tr04.BATCH == 0
    assert not set(KEEP["train"]) & set(KEEP["validation"])
    for key in ("lineage_id", "family", "schema_id"):
        assert not {r[key] for r in train} & {r[key] for r in valid}
    norm = lambda r: re.sub(r"\W+", " ", r["request"].lower()).strip()  # noqa: E731
    assert not {norm(r) for r in train} & {norm(r) for r in valid}
    audit = json.loads((tr04.TR04_DIR / "AUDIT.json").read_text("utf-8"))
    assert audit["independence_of_kept_splits"] == dict.fromkeys(
        ("lineages", "normalized_requests", "schema_fingerprints", "table_families"), 0
    )


def test_exclusions_list_every_dropped_row_once_with_a_reason_and_stage() -> None:
    with (tr04.TR04_DIR / "EXCLUSIONS.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    assert header[:4] == ["split", "id", "stage", "reason"]
    ids = [r[1] for r in body]
    assert len(ids) == len(set(ids))
    kept = set(KEEP["train"]) | set(KEEP["validation"])
    assert not set(ids) & kept
    assert set(ids) | kept == set(BY_ID)
    assert {r[2] for r in body} <= {"rights", "label", "trim"}
    assert all(r[3] for r in body)


def test_no_uncertain_source_row_survives() -> None:
    sources = {BY_ID[i]["provenance"]["schema_source_id"] for ids in KEEP.values() for i in ids}
    assert sources == {"SRC-O06"}
    assert all(
        BY_ID[i]["provenance"]["origin"] == "synthetic_template"
        for ids in KEEP.values()
        for i in ids
    )


# ---------------------------------------------------------------------------- template contract


def test_every_lineage_passes_the_contract_in_both_languages() -> None:
    assert len(LINEAGES) == 122
    assert {
        lin.id: tr04_cues.lineage_problems(lin)
        for lin in LINEAGES
        if tr04_cues.lineage_problems(lin)
    } == {}
    committed = json.loads((tr04.TR04_DIR / "TEMPLATE_CONTRACT.json").read_text("utf-8"))
    assert set(committed["verdicts"]) == {lin.id for lin in LINEAGES}
    assert all(v == [] for v in committed["verdicts"].values())


@pytest.mark.parametrize(
    ("frame_swap", "expected"),
    [
        ("compare", "own_family_cue_missing:compare"),
        ("cluster", "own_family_cue_missing:cluster"),
        ("reject:forecasting", "own_reject_cue_missing:forecasting"),
        ("clarify:missing_cluster_count", "missing_count_without_cluster_cue_or_feats"),
    ],
)
def test_the_contract_flags_a_template_whose_label_contradicts_its_wording(
    frame_swap: str, expected: str
) -> None:
    lin = next(lin for lin in LINEAGES if lin.frame == "relate" and not lin.fixed)
    wrong = dataclasses.replace(lin, frame=frame_swap, hint=None)
    problems = " ".join(tr04_cues.lineage_problems(wrong))
    assert expected in problems


def test_the_contract_flags_slot_and_method_mismatches() -> None:
    lin = next(
        lin for lin in LINEAGES if lin.frame == "relate" and lin.fixed.get("method") == "pearson"
    )
    spearman = dataclasses.replace(lin, fixed={"method": "spearman"})
    text = " ".join(tr04_cues.lineage_problems(spearman))
    assert (
        "fixed_method_not_in_text:spearman" in text and "contradicting_method_cue:pearson" in text
    )
    en_only = dataclasses.replace(lin, en=lin.en + " {extra}")
    assert "slot_sets_differ_between_languages" in tr04_cues.lineage_problems(en_only)
    inspect_all = next(lin for lin in LINEAGES if lin.frame == "inspect_all")
    with_slot = dataclasses.replace(
        inspect_all, tr=inspect_all.tr + " {cols}", en=inspect_all.en + " {cols}"
    )
    assert "tr:inspect_all_with_a_column_slot" in tr04_cues.lineage_problems(with_slot)


def test_cue_matching_is_turkish_aware_and_ignores_placeholders() -> None:
    assert "relate" in tr04_cues.families("{x} ile {y} arasındaki İLİŞKİYİ hesapla")
    assert tr04_cues.families("{cluster} {correlation}") == set()  # slot names are not words
    assert "causal_inference" in tr04_cues.reject_reasons("Nedensel etkiyi göster")


# ------------------------------------------------------------------------- row audit and faults


def _plan(frame: str, lang: str = "en") -> dict:
    return next(
        BY_ID[i]
        for i in KEEP["train"]
        if BY_ID[i]["frame"] == frame and BY_ID[i]["language"] == lang
    )


def test_a_clean_row_passes_and_each_deliberate_fault_is_named() -> None:
    row = _plan("relate")
    assert tr04.row_problems(row) == []
    swapped = copy.deepcopy(row)
    swapped["gold"]["method"] = "spearman" if row["gold"]["method"] == "pearson" else "pearson"
    assert tr04.row_problems(swapped)
    rejected = copy.deepcopy(row)
    rejected["gold"] = {"action": "REJECT", "reason": "out_of_scope"}
    assert tr04.row_problems(rejected)
    other_split = copy.deepcopy(row)
    other_split["split"] = "validation" if row["split"] == "train" else "train"
    assert "split_differs_from_the_preregistered_lineage_split" in tr04.row_problems(other_split)
    reviewed = copy.deepcopy(row)
    reviewed["provenance"]["human_review"] = "approved"
    assert any(p.startswith("human_review") for p in tr04.row_problems(reviewed))
    unknown = copy.deepcopy(row)
    unknown["lineage_id"] = "nope"
    assert tr04.row_problems(unknown) == ["unknown_lineage"]


def test_a_record_the_audit_cannot_read_is_a_problem_never_a_pass() -> None:
    assert tr04.row_problems({}) != []
    assert tr04.row_problems({"lineage_id": "x"}) != []
    row = copy.deepcopy(_plan("compare"))
    del row["mentions"]
    assert tr04.row_problems(row)


def test_a_cluster_row_with_a_changed_k_is_flagged() -> None:
    row = copy.deepcopy(_plan("cluster"))
    row["gold"]["k"] = int(row["gold"]["k"]) + 1
    assert tr04.row_problems(row)


def test_every_mutation_family_is_applied_and_flagged_on_a_fresh_sample() -> None:
    report = tr04.mutation_report([BY_ID[i] for i in KEEP["train"]], sample=60)
    mutations = (
        report["row_mutations_on_kept_train"] if "row_mutations_on_kept_train" in report else report
    )
    assert mutations
    for name, entry in mutations.items():
        assert entry["applied"] > 0, name
        assert entry["flagged"] == entry["applied"], (name, entry["not_flagged_examples"][:1])


def test_the_committed_mutation_report_applied_each_fault_and_flagged_it() -> None:
    committed = json.loads((tr04.TR04_DIR / "MUTATION_REPORT.json").read_text("utf-8"))
    rows = committed["row_mutations_on_kept_train"]
    assert len(rows) >= 10
    for name, entry in rows.items():
        assert entry["applied"] > 0 and entry["flagged"] == entry["applied"], name
    assert "not accuracy" in committed["meaning"]


def test_rows_in_the_corpus_that_the_audit_dropped_for_labels_really_have_a_problem() -> None:
    with (tr04.TR04_DIR / "EXCLUSIONS.csv").open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.reader(fh)][1:]
    labelled = [r for r in rows if r[2] == "label"]
    assert labelled
    reasons = {r[3] for r in labelled}
    assert reasons <= {
        "duplicate_request_on_the_same_layout",
        "ambiguous_mention_is_also_an_exact_column",
    }
