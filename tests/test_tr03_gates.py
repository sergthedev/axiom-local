"""EXP-TR03 gate checker: negative and positive tests on synthetic ledgers.

A synthetic ledger tests the checker's *logic* only. It is not a review, nobody judged anything,
and no ledger like it is committed. Every "good" fixture here is built in memory or in ``tmp_path``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date

import pytest

from axbench_tr import tr03
from axbench_tr import tr03_gates as g

TODAY = date(2026, 12, 10)
D1, D2, D3 = "2026-12-02", "2026-12-03", "2026-12-04"


def person(pid, roles, tr=True, en=True, independent=True):
    return {
        "id": pid,
        "type": "human",
        "roles": roles,
        "native_turkish": tr,
        "fluent_english": en,
        "independent_of_corpus_authoring": independent,
    }


def make_roster():
    people = [
        person("OWN1", ["owner"], independent=False),
        person("RV1", ["reviewer"]),
        person("RV2", ["reviewer"]),
        person("AD1", ["adjudicator"]),
        *(person(f"AU{i}", ["author"]) for i in range(1, 6)),
        person("CK1", ["checker"]),
        person("CK2", ["checker"]),
    ]
    return {
        "schema": g.ROSTER_SCHEMA,
        "designated_by": "OWN1",
        "designated_on": "2026-12-01",
        "attestation": {
            "statement": tr03.ROSTER_STATEMENT,
            "statement_accepted": True,
            "attested_by": "OWN1",
            "attested_on": "2026-12-01",
        },
        "people": people,
    }


ROSTER = make_roster()


@pytest.fixture(scope="module")
def frozen():
    return g.frozen_worklists()


@pytest.fixture(scope="module")
def corpus():
    train, valid, _, _ = tr03.load()
    elig_t, _ = tr03.split_eligible(train)
    elig_v, _ = tr03.split_eligible(valid)
    return elig_t, elig_v


@pytest.fixture(scope="module")
def index():
    idx = g.RosterIndex.build(ROSTER)
    assert idx is not None
    return idx


# ------------------------------------------------------------------------------ ledger builders

SECOND_ROW_COLS = tr03.review_blocks(tr03.ROW_JUDGMENTS)[1]
CLEAN_ROW = dict(request_natural="yes", gold_correct="yes", serious_error="no", decision="keep")
CLEAN_TEMPLATE = dict(
    tr_natural="yes", en_natural="yes", gold_rule_correct="yes", serious_error="no", decision="keep"
)


def good_ledger(frozen_rows, clean, kind):
    rows = []
    for fr in frozen_rows:
        r = dict(fr)
        r.update(reviewer_id="RV1", reviewer_type="human", reviewed_on=D1, **clean)
        if r["second_review_planned"] == "yes":
            r.update(second_reviewer_id="RV2", second_reviewer_type="human", second_reviewed_on=D2)
            r.update({f"second_{k}": v for k, v in clean.items()})
        rows.append(r)
    return rows


def as_bytes(spec, rows):
    return tr03._csv_bytes(spec.columns, rows)


def second(row, **kv):
    row.update(second_reviewer_id="RV2", second_reviewer_type="human", second_reviewed_on=D2)
    row.update({f"second_{k}": v for k, v in {**CLEAN_ROW, **kv}.items()})


def adjudicate(row, adjudicator="AD1", **kv):
    row.update(adjudicator_id=adjudicator, adjudicator_type="human", adjudicated_on=D3)
    row.update({f"adjudicated_{k}": v for k, v in {**CLEAN_ROW, **kv}.items()})


def run_rows(frozen, index, mutate=None, today=TODAY):
    rows = good_ledger(frozen["row"], CLEAN_ROW, "row")
    if mutate:
        out = mutate(rows)
        rows = out if isinstance(out, list) else rows
    return g.evaluate_ledger(g.ROW_SPEC, as_bytes(g.ROW_SPEC, rows), frozen["row"], index, today)


def run_templates(frozen, index, mutate=None):
    rows = good_ledger(frozen["template"], CLEAN_TEMPLATE, "template")
    if mutate:
        out = mutate(rows)
        rows = out if isinstance(out, list) else rows
    data = as_bytes(g.TEMPLATE_SPEC, rows)
    return g.evaluate_ledger(g.TEMPLATE_SPEC, data, frozen["template"], index, TODAY)


def has(result, fragment):
    return any(fragment in p for p in result.problems)


# --------------------------------------------------------------------------------------- roster


def test_valid_roster_is_consistent_but_never_authenticated():
    assert g.roster_problems(ROSTER, "training") == []
    assert g.roster_problems(ROSTER, "heldout") == []


def test_the_shipped_template_is_empty_and_invalid():
    template = json.loads(tr03.build_outputs.__globals__["_json_bytes"](tr03.ROSTER_TEMPLATE))
    assert template["people"] == []
    problems = g.roster_problems(template, "training")
    assert "no_people" in problems and "attestation_not_accepted" in problems
    assert g.roster_problems(None) == ["roster_missing"]
    assert g.roster_problems([]) == ["roster_not_an_object"]


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda r: r["people"].append(person("RV1", ["reviewer"])), "duplicate_person_id:RV1"),
        (lambda r: r["people"].append(person("rv1", ["reviewer"])), "duplicate_person_id:rv1"),
        (lambda r: r["people"].append(person("", ["reviewer"])), "person_id_invalid_or_reserved"),
        (lambda r: r["people"].append(person("x", ["reviewer"])), "person_id_invalid_or_reserved"),
        (lambda r: r["people"].append(person("claude", ["reviewer"])), "reserved"),
        (lambda r: r["people"].append(person("gpt-9", ["reviewer"])), "reserved"),
        (lambda r: r["people"].append({**person("AI7", ["reviewer"]), "type": "ai"}), "not_human"),
        (
            lambda r: r["people"].append({**person("H8", ["reviewer"]), "type": "Human"}),
            "not_human",
        ),
        (lambda r: r["people"].append(person("H9", [])), "roles_invalid:H9"),
        (lambda r: r["people"].append(person("H9", ["janitor"])), "roles_invalid:H9"),
        (lambda r: r["people"].append(person("H9", ["reviewer", "reviewer"])), "roles_invalid"),
        (lambda r: r["people"].append(person("H9", ["reviewer"], tr=None)), "not_boolean:H9"),
        (lambda r: r["people"].append(person("H9", ["reviewer"], tr="yes")), "not_boolean:H9"),
        (
            lambda r: r["people"].append(person("H9", ["reviewer"], independent=False)),
            "not_independent_of_corpus_authoring:H9",
        ),
        (
            lambda r: r["people"].append(person("H9", ["author", "reviewer"])),
            "author_also_reviews_training_data:H9",
        ),
        (lambda r: r["people"].append({**person("H9", ["reviewer"]), "extra": 1}), "fields_wrong"),
        (lambda r: r.update(designated_by=""), "designated_by_missing"),
        (lambda r: r.update(designated_on="2026-02-30"), "designated_on_invalid"),
        (lambda r: r.update(designated_on="01/12/2026"), "designated_on_invalid"),
        (lambda r: r.update(schema="axbench-tr.tr03-roster/1"), "roster_schema_wrong"),
        (lambda r: r["attestation"].update(statement_accepted=False), "attestation_not_accepted"),
        (lambda r: r["attestation"].update(statement="I agree"), "statement_altered"),
        (lambda r: r["attestation"].update(attested_by="someone"), "not_by_designator"),
        (lambda r: r.update(attestation=None), "attestation_missing"),
        (lambda r: r.update(people=[]), "no_people"),
    ],
)
def test_roster_rejects_bad_designation(mutation, expected):
    roster = copy.deepcopy(ROSTER)
    mutation(roster)
    assert any(expected in p for p in g.roster_problems(roster, "training")), expected


def test_roster_needs_the_owner_and_enough_competent_independent_judges():
    def without(*ids):
        r = copy.deepcopy(ROSTER)
        r["people"] = [p for p in r["people"] if p["id"] not in ids]
        return r

    assert "no_owner" in g.roster_problems(without("OWN1"))
    assert "fewer_than_two_tr_competent_reviewers" in g.roster_problems(without("RV2"))
    assert "no_tr_competent_adjudicator" in g.roster_problems(without("AD1"))
    assert "fewer_than_three_en_competent_judges" in g.roster_problems(without("RV2", "AD1"))
    no_turkish = copy.deepcopy(ROSTER)
    for p in no_turkish["people"]:
        if p["id"] == "RV1":
            p["native_turkish"] = False
    assert "fewer_than_two_tr_competent_reviewers" in g.roster_problems(no_turkish)
    # held-out purpose: authors must be native Turkish, and checkers exist
    assert "too_few_native_turkish_authors" in g.roster_problems(without("AU1", "AU2"), "heldout")
    assert "fewer_than_two_checkers" in g.roster_problems(without("CK2"), "heldout")
    foreign_authors = copy.deepcopy(ROSTER)
    for p in foreign_authors["people"]:
        if "author" in p["roles"]:
            p["native_turkish"] = False
    assert "too_few_native_turkish_authors" in g.roster_problems(foreign_authors, "heldout")


# -------------------------------------------------------------------------------- csv structure


def test_a_clean_row_ledger_is_consistent_and_meets_every_gate(frozen, index):
    res = run_rows(frozen, index)
    assert res.valid and res.complete, res.problems[:5]
    rep = g.row_gate_report(res, frozen["row"])
    assert rep["all_row_gates_met"], rep["gates"]
    assert rep["frozen_items"] == 518 and rep["item_states"]["resolved"] == 518
    assert rep["by_language"]["tr"]["gold_correct"]["of_frozen_n"] == 413
    assert (
        "not cluster adjusted" in rep["limits"].replace("-", " ")
        or "not a population" in (rep["limits"])
    )


def test_header_and_row_shape_must_be_exact(frozen, index):
    rows = good_ledger(frozen["row"], CLEAN_ROW, "row")
    good = as_bytes(g.ROW_SPEC, rows)
    lines = good.decode().split("\n")
    reordered = "\n".join(
        [lines[0].replace("sample_id,record_id", "record_id,sample_id"), *lines[1:]]
    )
    assert has(
        g.evaluate_ledger(g.ROW_SPEC, reordered.encode(), frozen["row"], index, TODAY), "header"
    )
    extra = "\n".join([lines[0] + ",bonus", *[line + ",x" for line in lines[1:] if line]])
    assert has(g.evaluate_ledger(g.ROW_SPEC, extra.encode(), frozen["row"], index, TODAY), "header")
    short = "\n".join([*lines[:5], lines[5][: len(lines[5]) // 2], *lines[6:]])
    r = g.evaluate_ledger(g.ROW_SPEC, short.encode(), frozen["row"], index, TODAY)
    assert not r.valid and not r.complete
    assert not g.evaluate_ledger(g.ROW_SPEC, b"", frozen["row"], index, TODAY).valid
    assert not g.evaluate_ledger(g.ROW_SPEC, b"\xff\xfe\x00", frozen["row"], index, TODAY).valid
    with_bom = b"\xef\xbb\xbf" + good  # a spreadsheet BOM is tolerated, content unchanged
    assert g.evaluate_ledger(g.ROW_SPEC, with_bom, frozen["row"], index, TODAY).complete


def test_roster_missing_or_invalid_blocks_every_ledger(frozen):
    rows = good_ledger(frozen["row"], CLEAN_ROW, "row")
    res = g.evaluate_ledger(g.ROW_SPEC, as_bytes(g.ROW_SPEC, rows), frozen["row"], None, TODAY)
    assert not res.valid and "roster_missing_or_invalid" in res.problems


# ----------------------------------------------------------------- identity, keys, completeness


def test_blank_rows_are_failures_not_a_smaller_denominator(frozen, index):
    def blank(rows):
        for r in rows[:30]:
            for c in tr03.ROW_COLUMNS[tr03.ROW_COLUMNS.index("reviewer_id") :]:
                r[c] = ""

    res = run_rows(frozen, index, blank)
    assert res.valid and not res.complete and res.counts()["not_reviewed"] == 30
    rep = g.row_gate_report(res, frozen["row"])
    assert not rep["gates"]["G3_ledger_valid_and_complete"]
    assert not any(rep["gates"].values())
    assert sum(b["frozen_n"] for b in rep["by_language"].values()) == 518  # not 488


def test_missing_duplicate_and_foreign_rows_invalidate_the_ledger(frozen, index):
    res = run_rows(frozen, index, lambda rows: rows.pop(7))
    assert has(res, "missing_key:") and not res.valid
    res = run_rows(frozen, index, lambda rows: rows.append(dict(rows[3])))
    assert has(res, "duplicate_key:") and not res.valid  # a repeated row never counts twice
    assert len(res.items) == 518

    def foreign(rows):
        rows.append({**rows[0], "sample_id": "tr03-9999"})

    assert has(run_rows(frozen, index, foreign), "foreign_key:tr03-9999")
    # a row repeated while another is deleted keeps n at 518 only by invalidating, never by passing
    res = run_rows(frozen, index, lambda rows: (rows.pop(1), rows.append(dict(rows[0]))))
    assert not res.valid


@pytest.mark.parametrize(
    "column, value",
    [
        ("language", "en"),
        ("frame", "inspect"),
        ("request", "totally different request"),
        ("gold_json", "{}"),
        ("gold_readable", "REJECT (none)"),
        ("lineage_id", "TPL-FAKE"),
        ("schema_source_id", "SRC-O03"),
        ("schema_columns", "c1:x(numeric)"),
        ("risk", "high"),
        ("record_id", "forged-record"),
        ("batch", "RW-tr-99"),
        ("second_review_planned", "yes"),
    ],
)
def test_spoofed_identity_cells_are_caught(frozen, index, column, value):
    def spoof(rows):
        target = next(r for r in rows if r[column] != value)
        target[column] = value

    res = run_rows(frozen, index, spoof)
    assert has(res, "identity_differs") and not res.valid


# ---------------------------------------------------------------------------- field validity


@pytest.mark.parametrize(
    "column, value",
    [
        ("decision", ""),
        ("decision", "Keep"),
        ("decision", " keep"),
        ("decision", "approve"),
        ("gold_correct", "maybe"),
        ("gold_correct", "YES"),
        ("request_natural", "ok"),
        ("serious_error", "none"),
        ("reviewer_type", "ai"),
        ("reviewer_type", "Human"),
        ("reviewer_type", ""),
        ("reviewed_on", "2026-13-45"),
        ("reviewed_on", "12/02/2026"),
        ("reviewed_on", "2026-12-2"),
        ("reviewed_on", "yesterday"),
        ("reviewed_on", "2026-11-30"),  # before the owner designated anyone
        ("reviewed_on", "2027-01-01"),  # in the future
    ],
)
def test_invalid_enums_dates_and_types_invalidate_or_block(frozen, index, column, value):
    def bad(rows):
        rows[11][column] = value

    res = run_rows(frozen, index, bad)
    assert not res.complete
    assert res.items[frozen["row"][11]["sample_id"]].state in {"invalid", "not_reviewed"} or (
        not res.valid
    )
    assert not g.row_gate_report(res, frozen["row"])["all_row_gates_met"]


def test_partially_filled_reviews_are_invalid_not_counted(frozen, index):
    def partial(rows):
        rows[0]["decision"] = ""
        rows[0]["notes"] = "kept only a note"

    res = run_rows(frozen, index, partial)
    assert has(res, "partially_filled") and not res.valid

    def only_notes(rows):
        rows[2]["notes"] = "x"
        for c in ("reviewer_id", "reviewer_type", "reviewed_on", *tr03.ROW_JUDGMENTS):
            rows[2][c] = ""

    assert has(run_rows(frozen, index, only_notes), "partially_filled")


# ---------------------------------------------------------------------- roles and independence


def test_reviewer_must_be_a_human_on_the_roster_with_the_role(frozen, index):
    for rid, fragment in (
        ("NOBODY", "not_on_roster"),
        ("AD1", "lacks_role_reviewer"),  # an adjudicator is not a reviewer
        ("OWN1", "lacks_role_reviewer"),
        ("AU1", "lacks_role_reviewer"),  # an author never reviews training data
        ("CK1", "lacks_role_reviewer"),
    ):
        res = run_rows(frozen, index, lambda rows, rid=rid: rows[0].update(reviewer_id=rid))
        assert has(res, fragment) and not res.valid, rid


def test_language_competence_is_enforced_per_item(frozen):
    roster = copy.deepcopy(ROSTER)
    roster["people"].append(person("RVEN", ["reviewer"], tr=False, en=True))
    roster["people"].append(person("RVTR", ["reviewer"], tr=True, en=False))
    idx = g.RosterIndex.build(roster)
    tr_row = next(i for i, r in enumerate(frozen["row"]) if r["language"] == "tr")
    en_row = next(i for i, r in enumerate(frozen["row"]) if r["language"] == "en")
    res = run_rows(frozen, idx, lambda rows: rows[tr_row].update(reviewer_id="RVEN"))
    assert has(res, "not_native_turkish:RVEN")
    res = run_rows(frozen, idx, lambda rows: rows[en_row].update(reviewer_id="RVTR"))
    assert has(res, "not_fluent_english:RVTR")
    res = run_rows(frozen, idx, lambda rows: rows[en_row].update(reviewer_id="RVEN"))
    assert res.valid


def test_second_reviewer_must_be_another_person_with_the_reviewer_role(frozen, index):
    planned = next(i for i, r in enumerate(frozen["row"]) if r["second_review_planned"] == "yes")

    def same(rows):
        rows[planned]["second_reviewer_id"] = rows[planned]["reviewer_id"]

    assert has(run_rows(frozen, index, same), "second_reviewer_is_the_first_reviewer")

    def wrong_role(rows):
        rows[planned]["second_reviewer_id"] = "AD1"

    assert has(run_rows(frozen, index, wrong_role), "second:lacks_role_reviewer")

    def case_trick(rows):
        rows[planned]["second_reviewer_id"] = "rv1"

    assert has(run_rows(frozen, index, case_trick), "not_on_roster:rv1")  # ids are exact

    def ai_second(rows):
        rows[planned]["second_reviewer_type"] = "ai"

    assert has(run_rows(frozen, index, ai_second), "second_type_not_human")


def test_adjudicator_must_be_a_third_person_with_the_role_and_a_later_date(frozen, index):
    i = next(i for i, r in enumerate(frozen["row"]) if r["second_review_planned"] == "yes")

    def disputed(rows, **adj):
        second(rows[i], decision="drop", gold_correct="no")
        adjudicate(rows[i], **adj)

    ok = run_rows(frozen, index, lambda rows: disputed(rows))
    assert ok.valid and ok.items[frozen["row"][i]["sample_id"]].state == "resolved"
    for who, fragment in (
        ("RV1", "lacks_role_adjudicator"),
        ("RV2", "lacks_role_adjudicator"),
        ("OWN1", "lacks_role_adjudicator"),
        ("GHOST", "not_on_roster"),
    ):
        res = run_rows(frozen, index, lambda rows, who=who: disputed(rows, adjudicator=who))
        assert has(res, fragment), who

    dual = copy.deepcopy(ROSTER)
    dual["people"].append(person("RA1", ["reviewer", "adjudicator"]))
    idx2 = g.RosterIndex.build(dual)
    for field in ("reviewer_id", "second_reviewer_id"):

        def own_adjudicator(rows, field=field):
            disputed(rows, adjudicator="RA1")
            rows[i][field] = "RA1"
            if field == "reviewer_id":
                rows[i]["second_reviewer_id"] = "RV2"

        res = run_rows(frozen, idx2, own_adjudicator)
        assert has(res, "adjudicator_is_a_reviewer_of_the_item"), field

    def too_early(rows):
        disputed(rows)
        rows[i]["adjudicated_on"] = "2026-12-01"

    assert has(run_rows(frozen, index, too_early), "adjudicated_before_the_reviews")

    def no_dispute(rows):
        second(rows[i])
        adjudicate(rows[i])

    assert has(run_rows(frozen, index, no_dispute), "adjudication_without_dispute")

    def no_second(rows):
        rows[i].update(dict.fromkeys(SECOND_ROW_COLS, ""))
        adjudicate(rows[i])

    assert has(run_rows(frozen, index, no_second), "adjudication_without_second_review")

    def later_without_first(rows):
        for c in ("reviewer_id", "reviewer_type", "reviewed_on", *tr03.ROW_JUDGMENTS):
            rows[i][c] = ""

    assert has(run_rows(frozen, index, later_without_first), "later_review_without_first")


# ------------------------------------------------------------------ adjudication rule, effects


def test_conflicting_reviews_stay_unresolved_until_adjudicated(frozen, index):
    i = next(i for i, r in enumerate(frozen["row"]) if r["second_review_planned"] == "yes")
    key = frozen["row"][i]["sample_id"]

    def conflict(rows):
        second(rows[i], decision="drop")

    res = run_rows(frozen, index, conflict)
    assert res.valid and not res.complete and res.items[key].state == "disputed"
    rep = g.row_gate_report(res, frozen["row"])
    assert rep["disputed"] == 1 and not rep["all_row_gates_met"]
    # any single differing field is a dispute, not only the decision
    res = run_rows(
        frozen, index, lambda rows: second(rows[i], request_natural="fixable", decision="fix")
    )
    assert res.items[key].state == "disputed"
    # adjudication decides; the adjudicator's judgement is the effective one
    res = run_rows(
        frozen,
        index,
        lambda rows: (second(rows[i], decision="drop"), adjudicate(rows[i], decision="keep")),
    )
    assert res.items[key].effective["decision"] == "keep" and res.complete
    res = run_rows(
        frozen,
        index,
        lambda rows: (
            second(rows[i], decision="drop", gold_correct="no"),
            adjudicate(rows[i], decision="drop", gold_correct="no"),
        ),
    )
    assert res.items[key].effective["gold_correct"] == "no"


def test_every_not_clean_keep_needs_a_second_review(frozen, index):
    i = next(i for i, r in enumerate(frozen["row"]) if r["second_review_planned"] != "yes")
    key = frozen["row"][i]["sample_id"]
    for kv in (
        dict(gold_correct="no", decision="drop"),
        dict(serious_error="yes", decision="drop"),
        dict(request_natural="fixable", decision="fix"),
        dict(decision="drop"),
    ):
        res = run_rows(frozen, index, lambda rows, kv=kv: rows[i].update(kv))
        assert res.items[key].state == "awaiting_second", kv
        assert not res.complete
        res = run_rows(
            frozen, index, lambda rows, kv=kv: (rows[i].update(kv), second(rows[i], **kv))
        )
        assert res.items[key].state == "resolved", kv
    # planned second reviews cannot be skipped either
    j = next(i for i, r in enumerate(frozen["row"]) if r["second_review_planned"] == "yes")

    def skip(rows):
        rows[j].update(dict.fromkeys(SECOND_ROW_COLS, ""))

    assert run_rows(frozen, index, skip).items[frozen["row"][j]["sample_id"]].state == (
        "awaiting_second"
    )


def test_inconsistent_judgements_are_invalid(frozen, index):
    cases = {
        "serious error kept": dict(serious_error="yes", decision="keep"),
        "wrong gold kept": dict(gold_correct="no", decision="keep"),
        "fixable kept as is": dict(request_natural="fixable", decision="keep"),
        "unnatural kept": dict(request_natural="no", decision="keep"),
        "irreparable fixed": dict(request_natural="no", decision="fix"),
        "serious fixed": dict(serious_error="yes", decision="fix"),
    }
    for name, kv in cases.items():
        res = run_rows(frozen, index, lambda rows, kv=kv: rows[4].update(kv))
        assert not res.valid, name
        assert not g.row_gate_report(res, frozen["row"])["gates"]["G3d_zero_serious_errors_kept"]


# ------------------------------------------------------------------------------ numeric gates


def tr_rows(frozen, cls):
    return [
        i
        for i, r in enumerate(frozen["row"])
        if r["language"] == "tr" and tr03.frame_class(r["frame"]) == cls
    ]


def mark_wrong(rows, idxs):
    for i in idxs:
        rows[i].update(gold_correct="no", decision="drop")
        second(rows[i], gold_correct="no", decision="drop")


def test_a_turkish_stratum_below_95_percent_fails_the_stratum_gate(frozen, index):
    idxs = tr_rows(frozen, "compare")
    assert len(idxs) == 59
    res = run_rows(frozen, index, lambda rows: mark_wrong(rows, idxs[:3]))  # 56/59 = 94.9%
    rep = g.row_gate_report(res, frozen["row"])
    assert res.complete
    assert not rep["gates"]["G3f_every_turkish_frame_stratum_has_59_frozen_rows_and_gold_ge_95pct"]
    assert rep["gates"]["G3t_turkish_gold_ge_95pct"]  # the language-wide gate alone would pass
    res = run_rows(frozen, index, lambda rows: mark_wrong(rows, idxs[:2]))  # 57/59 = 96.6%
    assert g.row_gate_report(res, frozen["row"])["all_row_gates_met"]
    assert g.row_gate_report(res, frozen["row"])["serious_errors_kept"] == 0


def test_english_and_per_origin_gates_are_kept_and_reported(frozen, index):
    en = [i for i, r in enumerate(frozen["row"]) if r["language"] == "en"]
    assert len(en) == 105
    res = run_rows(frozen, index, lambda rows: mark_wrong(rows, en[:6]))  # 99/105 = 94.3%
    rep = g.row_gate_report(res, frozen["row"])
    assert not rep["gates"]["G3l_english_gold_ge_95pct"]
    assert (
        rep["gates"]["G3t_turkish_gold_ge_95pct"]
        and rep["gates"]["G3o_origin_gold_ge_95pct_all_languages"]
    )
    assert set(rep["by_stratum_reported"]) >= {f"en:{c}" for c in tr03.FRAME_CLASSES}
    assert set(rep["by_schema_source_reported"]) == {"SRC-O01", "SRC-O02", "SRC-O06"}
    assert set(rep["by_language"]) == {"tr", "en"}
    # natural-or-fixable counts "fixable" as acceptable and only "no" as a failure
    fixable = run_rows(
        frozen, index, lambda rows: mark(rows, range(60), request_natural="fixable", decision="fix")
    )
    assert g.row_gate_report(fixable, frozen["row"])["gates"][
        "G3o_origin_natural_or_fixable_ge_90pct_all_languages"
    ]
    for n, expected in ((40, True), (60, False)):  # 478/518 = 92.3%, 458/518 = 88.4%
        res = run_rows(
            frozen,
            index,
            lambda rows, n=n: mark(rows, range(n), request_natural="no", decision="drop"),
        )
        rep = g.row_gate_report(res, frozen["row"])
        key = "G3o_origin_natural_or_fixable_ge_90pct_all_languages"
        assert rep["gates"][key] is expected, n


def mark(rows, idxs, **kv):
    for i in idxs:
        rows[i].update(kv)
        second(rows[i], **kv)


# --------------------------------------------------------------------------- template ledger


def test_template_ledger_expects_all_122_lineages_train_and_validation(frozen, index):
    assert len(frozen["template"]) == 122
    res = run_templates(frozen, index)
    rep = g.template_gate_report(res, frozen["template"])
    assert res.complete and all(rep["gates"].values())
    assert rep["templates_frozen"] == 122 and len(rep["kept_lineage_ids"]) == 122
    validation = next(i for i, t in enumerate(frozen["template"]) if t["split"] == "validation")
    res = run_templates(frozen, index, lambda rows: rows.pop(validation))
    assert has(res, "missing_key:") and not res.valid
    assert g.template_gate_report(res, frozen["template"])["kept_lineage_ids"] == []


def test_template_blank_decision_duplicates_and_spoofs_do_not_count(frozen, index):
    res = run_templates(frozen, index, lambda rows: rows[3].update(decision=""))
    assert not res.valid  # a stage with a blank decision is partial, never a decision
    res = run_templates(frozen, index, lambda rows: rows.append(dict(rows[5])))
    assert has(res, "duplicate_key:")
    res = run_templates(frozen, index, lambda rows: rows[0].update(template_tr="rewritten"))
    assert has(res, "identity_differs")
    res = run_templates(frozen, index, lambda rows: rows[0].update(authored_by="R1 (human)"))
    assert has(res, "identity_differs")  # the AI authorship label cannot be edited away

    def blank_all(rows):
        for c in tr03.TEMPLATE_COLUMNS[tr03.TEMPLATE_COLUMNS.index("reviewer_id") :]:
            rows[9][c] = ""

    res = run_templates(frozen, index, blank_all)
    rep = g.template_gate_report(res, frozen["template"])
    assert res.valid and not res.complete and len(rep["kept_lineage_ids"]) == 0
    assert not rep["gates"]["G2_every_frozen_template_has_a_resolved_human_decision"]


def test_template_fix_drop_and_serious_error_never_train(frozen, index):
    def mutate(rows):
        rows[0].update(decision="fix", tr_natural="fixable")
        second_t(rows[0], decision="fix", tr_natural="fixable")
        rows[1].update(decision="drop", serious_error="yes")
        second_t(rows[1], decision="drop", serious_error="yes")

    res = run_templates(frozen, index, mutate)
    assert res.complete
    rep = g.template_gate_report(res, frozen["template"])
    kept = set(rep["kept_lineage_ids"])
    assert frozen["template"][0]["lineage_id"] not in kept
    assert frozen["template"][1]["lineage_id"] not in kept
    assert len(kept) == 120
    assert rep["decisions"] == {"drop": 1, "fix": 1, "keep": 120}


def second_t(row, **kv):
    row.update(second_reviewer_id="RV2", second_reviewer_type="human", second_reviewed_on=D2)
    row.update({f"second_{k}": v for k, v in {**CLEAN_TEMPLATE, **kv}.items()})


def test_dropping_most_templates_of_a_frame_class_fails_coverage(frozen, index):
    victims = [
        i
        for i, t in enumerate(frozen["template"])
        if t["split"] == "train" and tr03.frame_class(t["frame"]) == "inspect_all"
    ]
    assert len(victims) == 3

    def mutate(rows):
        for i in victims[:2]:
            rows[i].update(decision="drop")
            second_t(rows[i], decision="drop")

    res = run_templates(frozen, index, mutate)
    rep = g.template_gate_report(res, frozen["template"])
    assert res.complete and not rep["gates"]["G2c_every_frame_class_keeps_enough_templates"]
    assert rep["kept_templates_by_frame_class"]["train"]["inspect_all"] == 1


# ------------------------------------------------------------------------------------- rights


def rights_rows(frozen, **include):
    rows = []
    for r in frozen["rights"]:
        row = dict(r)
        keep = r["source_id"] in tr03.ELIGIBLE_SOURCES
        row.update(
            owner_decision=include.get(r["source_id"], "include" if keep else "exclude"),
            decided_by="OWN1",
            decided_on=D1,
            notes="caveat read and addressed in the recorded way" if keep else "excluded by rule",
        )
        rows.append(row)
    return rows


def rights_report(frozen, index, rows):
    data = tr03._csv_bytes(tr03.RIGHTS_COLUMNS, rows)
    return g.rights_gate_report(data, frozen["rights"], index, TODAY)


def test_rights_ledger_valid_and_negative_cases(frozen, index):
    ok = rights_report(frozen, index, rights_rows(frozen))
    assert ok["complete"] and ok["included_sources"] == sorted(tr03.ELIGIBLE_SOURCES)
    assert "no legal opinion" in ok["note"]

    def problem(mutate):
        rows = rights_rows(frozen)
        mutate(rows)
        rep = rights_report(frozen, index, rows)
        assert not rep["complete"] and rep["included_sources"] == []
        return " ".join(rep["problems"]["first"])

    sid = {r["source_id"]: i for i, r in enumerate(frozen["rights"])}
    assert "excluded_by_rule" in problem(
        lambda rows: rows[sid["SRC-O03"]].update(owner_decision="include")
    )
    assert "excluded_by_rule" in problem(
        lambda rows: rows[-1].update(owner_decision="include")
    )  # external_adapted
    assert "needs_notes" in problem(lambda rows: rows[sid["SRC-O01"]].update(notes=" "))
    assert "not_in_enum" in problem(lambda rows: rows[0].update(owner_decision="maybe"))
    assert "owner_on_the_roster" in problem(lambda rows: rows[0].update(decided_by="RV1"))
    assert "owner_on_the_roster" in problem(lambda rows: rows[0].update(decided_by="ZZ"))
    assert "decided_on_invalid" in problem(lambda rows: rows[0].update(decided_on="2026-12-99"))
    assert "decided_on_invalid" in problem(lambda rows: rows[0].update(decided_on="2026-11-01"))
    assert "missing_key" in problem(lambda rows: rows.pop(2))
    assert "duplicate_key" in problem(lambda rows: rows.append(dict(rows[0])))
    assert "identity_differs" in problem(lambda rows: rows[0].update(licence_id="CC0"))
    assert not rights_report(
        frozen,
        index,
        [
            {**r, "owner_decision": "", "decided_by": "", "decided_on": ""}
            for r in rights_rows(frozen)
        ],
    )["complete"]
    none = g.rights_gate_report(b"", frozen["rights"], index, TODAY)
    assert not none["complete"]
    assert not g.rights_gate_report(b"x", frozen["rights"], None, TODAY)["complete"]


# ------------------------------------------------------------------- the derived training set


def derive(frozen, index, corpus, mutate_rows=None, mutate_templates=None, rights=None):
    t = run_templates(frozen, index, mutate_templates)
    r = run_rows(frozen, index, mutate_rows)
    rr = rights or rights_report(frozen, index, rights_rows(frozen))
    return g.research_manifest(t, r, rr, corpus[0], corpus[1], frozen["row"]), (t, r, rr)


def test_all_kept_yields_every_eligible_row(frozen, index, corpus):
    m, _ = derive(frozen, index, corpus)
    assert m["ok"] and m["counts"]["train"] == 5091 and m["counts"]["validation"] == 1274
    assert m["train_ids"] == sorted(r["id"] for r in corpus[0])
    assert m["derived_from_unverified_human_ledgers"] is True


def test_derivation_honours_every_kind_of_decision(frozen, index, corpus):
    pick: list[int] = []
    seen: set[str] = set()
    for i, r in enumerate(frozen["row"]):
        if r["lineage_id"] not in seen:
            seen.add(r["lineage_id"])
            pick.append(i)
        if len(pick) == 4:
            break
    a, b, c, d = pick
    row = frozen["row"]

    def rows_mut(rows):
        # a: reviewers disagree, adjudicator drops -> row removed, template kept
        second(rows[a], decision="drop")
        adjudicate(rows[a], decision="drop")
        # b: gold wrong by both reviewers -> the whole template is removed
        kv = dict(gold_correct="no", decision="drop")
        rows[b].update(kv)
        second(rows[b], **kv)
        # c: reviewers disagree, adjudicator keeps -> row stays
        second(rows[c], decision="drop")
        adjudicate(rows[c], decision="keep")
        # d: fix -> row removed (no verified correction mechanism exists)
        kv = dict(decision="fix", request_natural="fixable")
        rows[d].update(kv)
        second(rows[d], **kv)

    m, _ = derive(frozen, index, corpus, rows_mut)
    assert m["ok"]
    ids = set(m["train_ids"])
    by_id = {r["id"]: r for r in corpus[0]}
    assert row[a]["record_id"] not in ids and row[d]["record_id"] not in ids
    assert row[c]["record_id"] in ids
    assert not any(by_id[i]["lineage_id"] == row[b]["lineage_id"] for i in ids)
    assert any(by_id[i]["lineage_id"] == row[a]["lineage_id"] for i in ids)  # template kept
    why = m["counts"]["train_excluded_by_reason"]
    assert why["template_removed_by_a_reviewed_row_finding"] > 0
    assert why["row_not_kept_by_review"] >= 2


def test_derivation_never_trains_on_a_wrong_gold_marked_keep(frozen, index, corpus):
    def wrong_but_keep(rows):
        rows[5].update(gold_correct="no", decision="keep")

    t = run_templates(frozen, index)
    r = run_rows(frozen, index, wrong_but_keep)
    assert not r.valid  # the ledger is refused outright
    m = g.research_manifest(
        t, r, rights_report(frozen, index, rights_rows(frozen)), corpus[0], corpus[1], frozen["row"]
    )
    assert not m["ok"] and "row_ledger_not_complete_and_valid" in m["reasons"]
    assert "train_ids" not in m


def test_derivation_applies_template_and_rights_decisions(frozen, index, corpus):
    tv = next(i for i, t in enumerate(frozen["template"]) if t["split"] == "validation")
    tt = next(i for i, t in enumerate(frozen["template"]) if t["split"] == "train")

    def tmpl(rows):
        for i in (tv, tt):
            rows[i].update(decision="drop", gold_rule_correct="no")
            second_t(rows[i], decision="drop", gold_rule_correct="no")

    rights = rights_report(frozen, index, rights_rows(frozen, **{"SRC-O02": "exclude"}))
    assert rights["complete"] and rights["included_sources"] == ["SRC-O01", "SRC-O06"]
    m, _ = derive(frozen, index, corpus, mutate_templates=tmpl, rights=rights)
    assert m["ok"]
    by_id = {r["id"]: r for r in (*corpus[0], *corpus[1])}
    lins = {frozen["template"][i]["lineage_id"] for i in (tv, tt)}
    kept_ids = set(m["train_ids"]) | set(m["validation_ids"])
    assert all(by_id[i]["lineage_id"] not in lins for i in kept_ids)
    assert all(by_id[i]["provenance"]["schema_source_id"] != "SRC-O02" for i in kept_ids)
    assert m["counts"]["train_excluded_by_reason"]["rights_source_not_included"] > 0
    assert m["counts"]["validation_excluded_by_reason"]["template_not_kept"] > 0


def test_derivation_refuses_unresolved_or_incomplete_inputs(frozen, index, corpus):
    i = next(i for i, r in enumerate(frozen["row"]) if r["second_review_planned"] == "yes")
    m, _ = derive(frozen, index, corpus, lambda rows: second(rows[i], decision="drop"))
    assert not m["ok"] and "row_ledger_not_complete_and_valid" in m["reasons"]
    m, _ = derive(frozen, index, corpus, mutate_templates=lambda rows: rows[0].update(decision=""))
    assert not m["ok"] and "template_ledger_not_complete_and_valid" in m["reasons"]
    m, _ = derive(frozen, index, corpus, rights={"complete": False})
    assert not m["ok"] and "rights_ledger_not_complete_and_valid" in m["reasons"]


def test_filtered_exposure_plan_is_exact(frozen, index, corpus):
    import importlib.util
    import sys

    path = tr03.PREP_DIR.parents[2] / "training/tr03/tr03.py"
    spec = importlib.util.spec_from_file_location("tr03_plan_for_gates_test", path)
    plan = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plan
    spec.loader.exec_module(plan)

    def tmpl(rows):
        rows[4].update(decision="drop")
        second_t(rows[4], decision="drop")

    m, _ = derive(frozen, index, corpus, mutate_templates=tmpl)
    p = plan.build_plan(None, m["train_ids"], m["validation_ids"])
    assert p["rows"] == {
        "train_eligible": len(m["train_ids"]),
        "validation_eligible": len(m["validation_ids"]),
    }
    assert p["row_selection"]["train_ids_sha256"] == m["train_ids_sha256"]
    assert p["schedule"]["optimizer_updates"] == len(m["train_ids"]) // 4
    assert "not a clearance" in p["status"]
    assert plan.plan_bytes(p) == plan.plan_bytes(
        plan.build_plan(None, m["train_ids"], m["validation_ids"])
    )
    with pytest.raises(ValueError):
        plan.build_plan(None, m["train_ids"], None)
    with pytest.raises(ValueError):
        plan.build_plan(None, [*m["train_ids"], m["train_ids"][0]], m["validation_ids"])
    with pytest.raises(ValueError):
        plan.build_plan(None, ["not-an-eligible-id"], m["validation_ids"])


# ------------------------------------------------------------------------ G5 to G7 and report


def heldout_object(bindings):
    tasks = []
    langs = ["tr"] * 300 + ["en"] * 100
    for n, lang in enumerate(langs):
        author = f"AU{n % 5 + 1}"
        tasks.append(
            {
                "task_id": f"H{n:04d}",
                "language": lang,
                "action": ("PLAN", "CLARIFY", "REJECT")[n % 3],
                "author_id": author,
                "checker_id": "CK1" if n % 2 else "CK2",
                "table_family_id": f"FAM{n % 25:02d}",
                "table_source": "independent_fictional",
                "layout_held_out": True,
                "intent_held_out": n % 2 == 0,
                "near_miss": n % 7 == 0,
            }
        )
    return {
        "schema": "axbench-tr.tr03-heldout-frozen/1",
        "bindings": {k: bindings[k] for k in ("protocol_sha256", "prep_manifest_sha256")},
        "tasks_file_sha256": "a" * 64,
        "attestation": {**dict.fromkeys(g.HELDOUT_ATTESTATIONS, True), "attested_by": "OWN1"},
        "frozen_on": D3,
        "tasks": tasks,
    }


def prereg_object(bindings):
    cfg = json.loads(g.TRAINING_CONFIG_PATH.read_text("utf-8"))
    return {
        "schema": "axbench-tr.tr03-preregistration/1",
        "bindings": dict(bindings),
        "comparators": {
            "C0": {"revision": cfg["model"]["revision"]},
            "C1": {"rules": "axiom.rules/0.1"},
            "C2": None,
        },
        "analysis": {
            "bootstrap_replicates": 2000,
            "seed": 2026,
            "family_alpha": 0.05,
            "coprimary_contrasts": 2,
            "per_contrast_one_sided_alpha": 0.025,
        },
        "approved_by": "OWN1",
        "approved_on": D3,
    }


def clearance_object(bindings, hashes):
    url = "https://github.com/sergthedev/axiom-local/pull/15#issuecomment-1"
    return {
        "schema": "axbench-tr.tr03-clearance/1",
        "commit": "b" * 40,
        "bindings": dict(bindings),
        "ledger_sha256": dict(hashes),
        "cleared_by": "technical lead",
        "owner_authorization": "owner",
        "clearance_comment_url": url,
        "owner_authorization_url": url,
        "macos_ci_run_url": "https://example.invalid/ci/1",
        "windows_ci_run_url": "https://example.invalid/ci/2",
        "m5_no_weight_preflight_sha256": "c" * 64,
        "cleared_on": D3,
    }


def test_g5_g6_g7_checkers_reject_empty_or_arbitrary_content():
    b = g.current_bindings()
    for obj in ({}, {"approved": True}, {"schema": "x"}, {"cleared": True, "commit": "abc"}):
        assert g.heldout_problems(obj, ROSTER, b, TODAY)
        assert g.preregistration_problems(obj, b, TODAY)
        assert g.clearance_problems(obj, b, {}, TODAY)


def test_g5_heldout_accepts_a_well_formed_freeze_and_rejects_each_flaw():
    b = g.current_bindings()
    assert g.heldout_problems(heldout_object(b), ROSTER, b, TODAY) == []

    def bad(mutate):
        obj = heldout_object(b)
        mutate(obj)
        return " ".join(g.heldout_problems(obj, ROSTER, b, TODAY))

    assert "stale_binding:protocol_sha256" in bad(
        lambda o: o["bindings"].update(protocol_sha256="0" * 64)
    )
    assert "tasks_file_sha256" in bad(lambda o: o.update(tasks_file_sha256="short"))
    assert "attestation_not_true:no_ai_authored" in bad(
        lambda o: o["attestation"].update(no_ai_authored=False)
    )
    assert "attested_by_is_not_the_owner" in bad(
        lambda o: o["attestation"].update(attested_by="RV1")
    )
    assert "checker_missing_wrong_role_or_the_author" in bad(
        lambda o: o["tasks"][0].update(checker_id=o["tasks"][0]["author_id"])
    )
    assert "author_not_an_author_on_the_roster" in bad(
        lambda o: o["tasks"][0].update(author_id="RV1")
    )
    assert "author_not_an_author_on_the_roster" in bad(
        lambda o: o["tasks"][0].update(author_id="ghost")
    )
    assert "task_id_missing_or_duplicate" in bad(
        lambda o: o["tasks"][1].update(task_id=o["tasks"][0]["task_id"])
    )
    assert "too_few_tasks" in bad(lambda o: o.update(tasks=o["tasks"][:350]))
    assert "table_source_invalid" in bad(
        lambda o: o["tasks"][0].update(table_source="training_layout")
    )
    assert "language_or_action_invalid" in bad(lambda o: o["tasks"][0].update(action="OTHER"))
    assert "not_boolean" in bad(lambda o: o["tasks"][0].update(near_miss="yes"))
    assert "frozen_on_invalid" in bad(lambda o: o.update(frozen_on="2099-01-01"))

    def one_author(o):
        for t in o["tasks"]:
            t["author_id"] = "AU1"

    assert "author_clusters_too_few_or_too_concentrated" in bad(one_author)

    def one_family(o):
        for t in o["tasks"]:
            t["table_family_id"] = "FAM00"

    assert "table_family_clusters_too_few_or_too_concentrated" in bad(one_family)


def test_g6_preregistration_binds_the_exact_protocol_config_plan_and_tooling():
    b = g.current_bindings()
    assert g.preregistration_problems(prereg_object(b), b, TODAY) == []
    for key in b:
        stale = {**b, key: "0" * 64}
        assert f"stale_binding:{key}" in g.preregistration_problems(prereg_object(stale), b, TODAY)
    obj = prereg_object(b)
    obj["comparators"]["C0"]["revision"] = "f" * 40
    assert "C0_revision_not_the_pinned_base" in g.preregistration_problems(obj, b, TODAY)
    obj = prereg_object(b)
    obj["analysis"]["per_contrast_one_sided_alpha"] = 0.05  # unadjusted
    assert "analysis_not_the_protocol_values" in g.preregistration_problems(obj, b, TODAY)
    obj = prereg_object(b)
    obj["comparators"]["C2"] = {"adapter_sha256": "d" * 64}
    assert (
        "C2_present_but_not_a_verified_frozen_secondary_comparator"
        in g.preregistration_problems(obj, b, TODAY)
    )
    obj["comparators"]["C2"] = {
        "adapter_sha256": "d" * 64,
        "inference_commit": "e" * 40,
        "retrained": False,
        "used_for_selection": False,
    }
    assert g.preregistration_problems(obj, b, TODAY) == []


def test_g7_clearance_is_bound_to_every_ledger_hash_and_the_current_code():
    b = g.current_bindings()
    hashes = {
        k: hashlib.sha256(k.encode()).hexdigest()
        for k in ("roster", "template", "row", "rights", "heldout", "preregistration")
    }
    assert g.clearance_problems(clearance_object(b, hashes), b, hashes, TODAY) == []
    stale = {**hashes, "row": "0" * 64}
    assert "stale_or_missing_ledger_hash:row" in g.clearance_problems(
        clearance_object(b, hashes), b, stale, TODAY
    )
    for key in ("tooling_sha256", "protocol_sha256", "plan_sha256"):
        old = {**b, key: "1" * 64}
        assert f"stale_binding:{key}" in g.clearance_problems(
            clearance_object(old, hashes), b, hashes, TODAY
        )
    obj = clearance_object(b, hashes)
    obj["commit"] = "main"
    obj["clearance_comment_url"] = "https://evil.example/x"
    obj["cleared_on"] = "2099-01-01"
    problems = g.clearance_problems(obj, b, hashes, TODAY)
    assert {
        "commit_not_a_40_hex_sha",
        "clearance_comment_url_not_a_github_comment_url",
        "cleared_on_invalid",
    } <= set(problems)


def write_ledgers(tmp_path, frozen):
    led = tmp_path / "ledgers"
    led.mkdir()
    (led / "HUMAN_ROSTER.json").write_text(json.dumps(ROSTER), "utf-8")
    t = good_ledger(frozen["template"], CLEAN_TEMPLATE, "template")
    r = good_ledger(frozen["row"], CLEAN_ROW, "row")
    (led / "TEMPLATE_REVIEW_RESULTS.csv").write_bytes(as_bytes(g.TEMPLATE_SPEC, t))
    (led / "ROW_REVIEW_RESULTS.csv").write_bytes(as_bytes(g.ROW_SPEC, r))
    (led / "RIGHTS_DECISIONS.csv").write_bytes(
        tr03._csv_bytes(tr03.RIGHTS_COLUMNS, rights_rows(frozen))
    )
    b = g.current_bindings()
    (led / "HELDOUT_FROZEN.json").write_text(json.dumps(heldout_object(b)), "utf-8")
    (led / "PREREGISTRATION.json").write_text(json.dumps(prereg_object(b)), "utf-8")
    return led, b


def clearance_for(led, b):
    names = {
        "roster": "HUMAN_ROSTER.json",
        "template": "TEMPLATE_REVIEW_RESULTS.csv",
        "row": "ROW_REVIEW_RESULTS.csv",
        "rights": "RIGHTS_DECISIONS.csv",
        "heldout": "HELDOUT_FROZEN.json",
        "preregistration": "PREREGISTRATION.json",
    }
    hashes = {k: hashlib.sha256((led / v).read_bytes()).hexdigest() for k, v in names.items()}
    return clearance_object(b, hashes)


def test_the_repository_has_no_ledgers_and_every_gate_is_absent():
    rep = g.gates_report(today=TODAY)
    assert not (tr03.PREP_DIR / "ledgers").exists()
    assert {k: v["status"] for k, v in rep["gates"].items()} == {
        "G1_roster": "absent",
        "G2_templates": "absent",
        "G3_rows": "absent",
        "G4_rights": "absent",
        "G5_heldout": "absent",
        "G6_preregistration": "absent",
        "G7_clearance": "absent",
    }
    assert rep["research_training_open"] is False and g.exit_code(rep) == 1
    assert rep["identity_authenticated"] is False and rep["frozen_worklist_problems"] == []


def test_presence_never_opens_anything_and_even_a_full_consistent_set_stays_closed(
    tmp_path, frozen
):
    led, b = write_ledgers(tmp_path, frozen)
    (led / "CLEARANCE.json").write_text(json.dumps(clearance_for(led, b)), "utf-8")
    rep = g.gates_report(ledgers=led, today=TODAY)
    states = {k: v["status"] for k, v in rep["gates"].items()}
    assert set(states.values()) == {"consistent_unverified"}, states
    assert rep["all_gates_consistent_unverified"] is True
    assert rep["research_training_open"] is False and rep["execution_gate"]["open"] is False
    assert g.exit_code(rep) == 2  # never 0
    assert rep["derived_training_set"]["ok"] is True
    assert "train_ids" not in rep["derived_training_set"]
    full = g.gates_report(ledgers=led, today=TODAY, include_ids=True)
    assert len(full["derived_training_set"]["train_ids"]) == 5091


def test_empty_arbitrary_or_malformed_files_are_invalid_not_present(tmp_path, frozen):
    led, b = write_ledgers(tmp_path, frozen)
    for name in ("HELDOUT_FROZEN.json", "PREREGISTRATION.json", "CLEARANCE.json"):
        for content in (
            "",
            "{}",
            "[]",
            '{"approved": true}',
            "not json",
            '{"a":1,"a":2}',
            '{"x": NaN}',
        ):
            (led / name).write_text(content, "utf-8")
            rep = g.gates_report(ledgers=led, today=TODAY)
            gate = {
                "HELDOUT_FROZEN.json": "G5_heldout",
                "PREREGISTRATION.json": "G6_preregistration",
                "CLEARANCE.json": "G7_clearance",
            }[name]
            assert rep["gates"][gate]["status"] == "invalid", (name, content)
            assert rep["all_gates_consistent_unverified"] is False and g.exit_code(rep) == 1
            assert rep["research_training_open"] is False


def test_stale_hashes_after_a_ledger_changes_invalidate_the_clearance(tmp_path, frozen):
    led, b = write_ledgers(tmp_path, frozen)
    (led / "CLEARANCE.json").write_text(json.dumps(clearance_for(led, b)), "utf-8")
    rows = good_ledger(frozen["row"], CLEAN_ROW, "row")
    rows[0]["notes"] = "edited after clearance"
    (led / "ROW_REVIEW_RESULTS.csv").write_bytes(as_bytes(g.ROW_SPEC, rows))
    rep = g.gates_report(ledgers=led, today=TODAY)
    assert rep["gates"]["G7_clearance"]["status"] == "invalid"
    assert "stale_or_missing_ledger_hash:row" in rep["gates"]["G7_clearance"]["problems"]
    assert rep["gates"]["G3_rows"]["status"] == "consistent_unverified"  # only the binding broke
    # a clearance that cites nothing is invalid too
    (led / "CLEARANCE.json").write_text(
        json.dumps({**clearance_for(led, b), "ledger_sha256": {}}), "utf-8"
    )
    assert g.gates_report(ledgers=led, today=TODAY)["gates"]["G7_clearance"]["status"] == "invalid"


def test_clearance_without_the_ledgers_it_binds_is_invalid(tmp_path, frozen):
    led, b = write_ledgers(tmp_path, frozen)
    (led / "CLEARANCE.json").write_text(json.dumps(clearance_for(led, b)), "utf-8")
    (led / "RIGHTS_DECISIONS.csv").unlink()
    rep = g.gates_report(ledgers=led, today=TODAY)
    assert rep["gates"]["G4_rights"]["status"] == "absent"
    assert rep["gates"]["G7_clearance"]["status"] == "invalid"
    assert rep["derived_training_set"]["ok"] is False
    assert rep["research_training_open"] is False


def test_bad_roster_file_blocks_the_dependent_ledgers(tmp_path, frozen):
    led, b = write_ledgers(tmp_path, frozen)
    bad = copy.deepcopy(ROSTER)
    bad["people"].append({**person("ROBOT1", ["reviewer"]), "type": "ai"})
    (led / "HUMAN_ROSTER.json").write_text(json.dumps(bad), "utf-8")
    rep = g.gates_report(ledgers=led, today=TODAY)
    assert rep["gates"]["G1_roster"]["status"] == "invalid"
    assert rep["gates"]["G2_templates"]["status"] == "invalid"
    assert rep["gates"]["G3_rows"]["status"] == "invalid"
    assert rep["gates"]["G4_rights"]["status"] != "consistent_unverified"
    assert rep["research_training_open"] is False


def test_frozen_worklists_are_committed_and_a_drifted_worklist_blocks_the_ledgers(tmp_path, frozen):
    led, b = write_ledgers(tmp_path, frozen)
    prep = tmp_path / "prep"
    prep.mkdir()
    for p in tr03.PREP_DIR.iterdir():
        if p.is_file():
            (prep / p.name).write_bytes(p.read_bytes())
    ok = g.gates_report(prep_dir=prep, ledgers=led, today=TODAY)
    assert ok["frozen_worklist_problems"] == []
    (prep / "ROW_REVIEW_WORKLIST.csv").write_bytes(b"sample_id\ntr03-0001\n")
    rep = g.gates_report(prep_dir=prep, ledgers=led, today=TODAY)
    assert any("ROW_REVIEW_WORKLIST" in p for p in rep["frozen_worklist_problems"])
    assert rep["gates"]["G3_rows"]["status"] == "invalid"
    # any drift in a frozen worklist blocks every review and rights ledger, not only its own
    assert rep["gates"]["G2_templates"]["status"] == "invalid"
    assert rep["gates"]["G4_rights"]["status"] == "invalid"
