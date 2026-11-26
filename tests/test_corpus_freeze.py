"""Phase 3A review regressions (PR #8 review of 51a6a3a), dataset-freezing blocker 3.

On the reviewed head every consumer regenerated the development corpus from templates, so a
reviewer's correction in the review CSV could not reach the dataset, the audit, the MLX export, the
evaluator or the seal; the seal also wrote its files when the audit failed and labelled examples
from owner lineages like repository ones. These tests use synthetic decision ledgers written by
the tests themselves; no owner decision exists yet and none is simulated in the repository.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

import pytest

import axbench.seal as seal_module
from axbench.__main__ import main as cli
from axbench.corpus import CorpusError, freeze, load_corpus
from axbench.seal import SealError, load_owner_lineages, seal

REPO = Path(__file__).resolve().parents[1]
DRAFT = REPO / "data" / "axbench" / "v0.1-draft"


def _ledger(rows: list[dict[str, str]]) -> bytes:
    buf = io.StringIO(newline="")
    fields = ["id", "decision", "corrected_gold", "corrected_request", "corrected_tags", "note"]
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({f: row.get(f, "") for f in fields})
    return buf.getvalue().encode("utf-8")


@pytest.fixture(scope="module")
def draft():
    return load_corpus(DRAFT)[1]


def _first(records, pred):
    return next(r for r in records if pred(r))


@pytest.fixture(scope="module")
def frozen_plain(tmp_path_factory):
    out = tmp_path_factory.mktemp("frozen") / "v0.1-frozen-test"
    freeze(DRAFT, out, version="axbench/0.1-frozen-test")
    return out


# --- verified loading ------------------------------------------------------------------------


def test_committed_draft_loads_with_verified_hashes(draft):
    assert len(draft["train"]) == 400 and len(draft["validation"]) == 160


def test_tampered_or_incomplete_corpus_is_refused(tmp_path):
    copy = tmp_path / "c"
    shutil.copytree(DRAFT, copy, ignore=shutil.ignore_patterns("review", "AUDIT.json"))
    lines = (copy / "validation.jsonl").read_bytes().splitlines(keepends=True)
    (copy / "validation.jsonl").write_bytes(b"".join(lines[1:]) + lines[0])  # reordered
    with pytest.raises(CorpusError, match="manifest hash"):
        load_corpus(copy)
    with pytest.raises(SystemExit, match="corpus not usable"):
        cli(["rules", "--corpus", str(copy), "--split", "validation", "--out", str(tmp_path / "p")])
    shutil.copy(DRAFT / "validation.jsonl", copy / "validation.jsonl")
    load_corpus(copy)
    (copy / "test.jsonl").write_bytes(b"")
    with pytest.raises(CorpusError, match="test.jsonl"):
        load_corpus(copy)


# --- freezing --------------------------------------------------------------------------------


def test_freeze_without_decisions_records_zero_human_review(frozen_plain):
    manifest, splits = load_corpus(frozen_plain)
    assert manifest["status"] == "frozen"
    assert manifest["parent"]["corpus_version"] == "axbench/0.1-draft"
    review = manifest["review"]
    assert review["examples_with_a_human_decision"] == 0 and not review["attested_human"]
    assert review["by_split"] == {"train": {"unreviewed": 400}, "validation": {"unreviewed": 160}}
    assert all(r["provenance"]["human_review"] == "none" for s in splits.values() for r in s)
    assert not (frozen_plain / "DECISIONS.jsonl").exists()


def test_ai_review_cannot_be_recorded_as_human(tmp_path):
    ledger = _ledger([{"id": "train-0001", "decision": "approve"}])
    with pytest.raises(CorpusError, match="attested as a human review"):
        freeze(DRAFT, tmp_path / "o", version="v-x", decisions_csv=ledger, reviewer="owner")
    with pytest.raises(CorpusError, match="name the reviewer"):
        freeze(
            DRAFT, tmp_path / "o", version="v-x", decisions_csv=ledger, human_review_attested=True
        )
    assert not (tmp_path / "o").exists()


def test_decisions_flow_into_the_frozen_version(tmp_path, draft):
    relate = _first(draft["validation"], lambda r: r["gold"].get("operation") == "relate")
    other_method = "spearman" if relate["gold"]["method"] == "pearson" else "pearson"
    corrected = {**relate["gold"], "method": other_method}
    approve = draft["train"][0]["id"]
    reject = draft["train"][1]["id"]
    retext = draft["train"][2]
    ledger = _ledger(
        [
            {"id": approve, "decision": "approve", "note": "fine"},
            {"id": relate["id"], "decision": "correct", "corrected_gold": json.dumps(corrected)},
            {
                "id": retext["id"],
                "decision": "correct",
                "corrected_request": retext["request"] + " .",
                "corrected_tags": ";".join(retext["tags"]) or "none",
            },
            {"id": reject, "decision": "Reject"},
            {"id": draft["train"][3]["id"], "decision": ""},  # left blank: not a decision
        ]
    )
    out = tmp_path / "v"
    manifest = freeze(
        DRAFT,
        out,
        version="axbench/0.1-r1",
        decisions_csv=ledger,
        reviewer="owner",
        human_review_attested=True,
    )
    review = manifest["review"]
    assert review["examples_with_a_human_decision"] == 4
    assert review["by_split"]["train"] == {
        "approve": 1,
        "correct": 1,
        "reject": 1,
        "unreviewed": 397,
    }
    assert review["by_split"]["validation"] == {"correct": 1, "unreviewed": 159}
    assert review["removed_ids"] == [reject]
    _, splits = load_corpus(out)
    by_id = {r["id"]: r for s in splits.values() for r in s}
    assert reject not in by_id and len(splits["train"]) == 399
    fixed = by_id[relate["id"]]
    assert fixed["gold"] == corrected
    assert fixed["provenance"]["original_gold"] == relate["gold"]
    assert fixed["provenance"]["label_source"] == "human_corrected"
    assert by_id[retext["id"]]["request"].endswith(" .")
    reviewed = {i for i, r in by_id.items() if r["provenance"]["human_review"] != "none"}
    assert reviewed == {approve, relate["id"], retext["id"]}
    assert all(
        r["provenance"]["authoring"] == "ai_generated_from_templates" for r in by_id.values()
    )
    ledger_lines = (out / "DECISIONS.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger_lines) == 4 and "fine" not in "".join(ledger_lines)  # notes stay private

    # the correction reaches the evaluator, the rule baseline and the MLX export
    preds = tmp_path / "p.jsonl"
    assert cli(["rules", "--corpus", str(out), "--split", "validation", "--out", str(preds)]) == 0
    report = tmp_path / "r.json"
    args = ["score", "--corpus", str(out), "--split", "validation", "--predictions", str(preds)]
    assert cli([*args, "--system", "rules", "--out", str(report)]) == 0
    for gold, expected in ((corrected, True), (relate["gold"], False)):
        one = tmp_path / "one.jsonl"
        pred = {"example_id": relate["id"], "status": "ok", "raw_output": json.dumps(gold)}
        one.write_text(json.dumps(pred) + "\n", encoding="utf-8")
        args = ["score", "--corpus", str(out), "--split", "validation", "--predictions", str(one)]
        assert cli([*args, "--system", "t", "--out", str(report)]) == 0
        per = json.loads(report.read_text(encoding="utf-8"))["per_example"]
        assert next(r for r in per if r["id"] == relate["id"])["exact_match"] is expected
    export = tmp_path / "mlx"
    assert cli(["export-mlx", "--corpus", str(out), "--out", str(export)]) == 0
    targets = [
        json.loads(line)["messages"][-1]["content"]
        for line in (export / "valid.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert json.dumps(corrected, ensure_ascii=False, separators=(",", ":")) in targets
    assert len((export / "train.jsonl").read_text(encoding="utf-8").splitlines()) == 399


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"id": "train-0001", "decision": "maybe"}], "unknown decision"),
        ([{"id": "nope-1", "decision": "approve"}], "unknown example ids"),
        (
            [
                {"id": "train-0001", "decision": "approve"},
                {"id": "train-0001", "decision": "reject"},
            ],
            "more than one decision",
        ),
        ([{"id": "train-0001", "decision": "correct"}], "needs a corrected"),
        ([{"id": "train-0001", "decision": "correct", "corrected_gold": "{"}], "not JSON"),
        (
            [{"id": "train-0001", "decision": "approve", "corrected_gold": "{}"}],
            "decision is 'approve'",
        ),
    ],
)
def test_bad_ledgers_are_refused_and_nothing_is_written(tmp_path, rows, message):
    with pytest.raises(CorpusError, match=message):
        freeze(
            DRAFT,
            tmp_path / "o",
            version="v-x",
            decisions_csv=_ledger(rows),
            reviewer="owner",
            human_review_attested=True,
        )
    assert not (tmp_path / "o").exists()


def test_invalid_correction_or_failed_audit_writes_nothing(tmp_path, draft):
    target = draft["train"][0]
    bad_gold = {"version": "axiom.candidate/1", "action": "CLARIFY", "reason": "ambiguous_column"}
    bad_gold["columns"] = ["c99"]
    kwargs = {"version": "v-x", "reviewer": "owner", "human_review_attested": True}
    with pytest.raises(CorpusError, match="corrected label is invalid"):
        freeze(
            DRAFT,
            tmp_path / "o",
            decisions_csv=_ledger(
                [
                    {
                        "id": target["id"],
                        "decision": "correct",
                        "corrected_gold": json.dumps(bad_gold),
                    }
                ]
            ),
            **kwargs,
        )
    # a corrected request that duplicates a validation request breaks split separation
    dup = draft["validation"][0]["request"]
    with pytest.raises(CorpusError, match="audit failed"):
        freeze(
            DRAFT,
            tmp_path / "o",
            decisions_csv=_ledger(
                [
                    {
                        "id": target["id"],
                        "decision": "correct",
                        "corrected_request": dup,
                        "corrected_tags": "none",
                    }
                ]
            ),
            **kwargs,
        )
    assert not (tmp_path / "o").exists()
    with pytest.raises(CorpusError, match="new version"):
        freeze(DRAFT, tmp_path / "o", version="axbench/0.1-draft")


# --- sealing ---------------------------------------------------------------------------------


def test_seal_refuses_a_draft_corpus(tmp_path):
    with pytest.raises(SealError, match="frozen"):
        seal(tmp_path / "s", "owner-secret-seed-for-unit-test", corpus_dir=DRAFT, size=30)
    assert not (tmp_path / "s").exists()


def test_seal_records_the_frozen_corpus_it_was_made_against(tmp_path, frozen_plain):
    manifest = seal(
        tmp_path / "s", "owner-secret-seed-for-unit-test", corpus_dir=frozen_plain, size=30
    )
    frozen_manifest = json.loads((frozen_plain / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["development_corpus"]["files"] == frozen_manifest["files"]
    assert manifest["corpus_version"] == "axbench/0.1-frozen-test"
    assert manifest["owner_lineages"]["supplied"] == 0
    assert manifest["examples_from_repository_lineages"] == 30


def test_failed_seal_audit_writes_nothing(tmp_path, frozen_plain, monkeypatch):
    def failing(corpus, **_):
        return {"checks": {"passed": False, "failures": ["synthetic failure"]}, "splits": {}}

    monkeypatch.setattr(seal_module, "audit", failing)
    with pytest.raises(SealError, match="audit failed"):
        seal(tmp_path / "s", "owner-secret-seed-for-unit-test", corpus_dir=frozen_plain, size=30)
    assert not (tmp_path / "s").exists()
    seed_file = tmp_path / "seed.txt"
    seed_file.write_text("owner-secret-seed-for-unit-test", encoding="utf-8")
    with pytest.raises(SystemExit, match="nothing written"):
        cli(
            [
                "seal",
                "--corpus",
                str(frozen_plain),
                "--out",
                str(tmp_path / "s2"),
                "--seed-file",
                str(seed_file),
            ]
        )
    assert not (tmp_path / "s2").exists()


def test_owner_lineage_examples_are_labelled_and_counted(tmp_path, frozen_plain):
    lineage_file = tmp_path / "owner.json"
    lineage_file.write_text(
        json.dumps(
            [
                {
                    "id": "t1",
                    "frame": "relate",
                    "tr": "{x} ve {y} için {m} korelasyonunu hesaplar mısın lütfen?",
                    "en": "Please compute the {m} correlation for {x} and {y}.",
                },
                {
                    "id": "t2",
                    "frame": "cluster",
                    "tr": "{feats} değişkenleriyle {k} küme oluştur lütfen.",
                    "en": "Please form {k} clusters using {feats}.",
                },
            ]
        ),
        encoding="utf-8",
    )
    owner = load_owner_lineages(lineage_file)
    out = tmp_path / "s"
    manifest = seal(
        out, "owner-secret-seed-for-unit-test", corpus_dir=frozen_plain, owner_lineages=owner
    )
    info = manifest["owner_lineages"]
    records = [json.loads(x) for x in (out / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    from_owner = [r for r in records if r["lineage_id"].startswith("O-")]
    assert info["supplied"] == 2 and from_owner  # the owner lineages were actually drawn
    assert info["examples"] == len(from_owner) == sum(info["examples_per_lineage"].values())
    assert info["used"] == sum(1 for n in info["examples_per_lineage"].values() if n)
    assert manifest["examples_from_repository_lineages"] == len(records) - len(from_owner)
    assert all(r["provenance"]["authoring"] == "owner_template_ai_rendered" for r in from_owner)
    assert all(r["provenance"]["human_review"] == "none" for r in from_owner)
    others = [r for r in records if not r["lineage_id"].startswith("O-")]
    assert all(r["provenance"]["authoring"] == "ai_generated_from_templates" for r in others)


# --- owner-authorized AI review (revision 3) -------------------------------------------------

AUTH = "https://github.com/sergthedev/axiom-local/issues/4#issuecomment-0000000000"


def test_ai_review_keeps_every_example_unreviewed_by_humans(tmp_path, draft):
    target = draft["validation"][0]
    approve = draft["train"][0]["id"]
    ledger = _ledger(
        [
            {"id": approve, "decision": "approve"},
            {
                "id": target["id"],
                "decision": "correct",
                "corrected_request": target["request"] + " lütfen.",
                "corrected_tags": ";".join(target["tags"]) or "none",
            },
        ]
    )
    out = tmp_path / "ai"
    manifest = freeze(
        DRAFT,
        out,
        version="axbench/0.1-ai-test",
        decisions_csv=ledger,
        reviewer="technical lead AI",
        ai_review_authorization=AUTH,
    )
    review = manifest["review"]
    assert review["reviewer_type"] == "ai" and not review["attested_human"]
    assert review["authorization"] == AUTH
    assert review["examples_with_a_decision"] == 2
    assert review["examples_with_a_human_decision"] == 0
    assert review["corrections"] == [
        {"id": target["id"], "split": "validation", "changed": ["request"]}
    ]
    assert "not human annotation" in review["note"]
    _, splits = load_corpus(out)
    records = [r for s in splits.values() for r in s]
    assert len(records) == 560
    assert all(r["provenance"]["human_review"] == "none" for r in records)
    by_id = {r["id"]: r for r in records}
    fixed = by_id[target["id"]]["provenance"]
    assert fixed["request_source"] == "ai_review_corrected"
    assert fixed["original_request"] == target["request"]
    assert fixed["review"] == {
        "decision": "correct",
        "reviewer": "technical lead AI",
        "reviewer_type": "ai",
        "authorization": AUTH,
    }
    assert by_id[approve]["provenance"]["review"]["reviewer_type"] == "ai"


def test_review_types_are_exclusive_and_need_a_ledger(tmp_path):
    ledger = _ledger([{"id": "train-0001", "decision": "approve"}])
    with pytest.raises(CorpusError, match="not both"):
        freeze(
            DRAFT,
            tmp_path / "o",
            version="v-x",
            decisions_csv=ledger,
            reviewer="r",
            human_review_attested=True,
            ai_review_authorization=AUTH,
        )
    with pytest.raises(CorpusError, match="without a decision ledger"):
        freeze(DRAFT, tmp_path / "o", version="v-x", ai_review_authorization=AUTH)
    with pytest.raises(CorpusError, match="attested as a human review"):
        freeze(
            DRAFT,
            tmp_path / "o",
            version="v-x",
            decisions_csv=ledger,
            reviewer="r",
            ai_review_authorization="  ",
        )
    assert not (tmp_path / "o").exists()


def test_corrected_tags_are_required_with_a_request_and_must_be_known(tmp_path, draft):
    rid = draft["train"][0]["id"]
    kwargs = {"version": "v-x", "reviewer": "r", "ai_review_authorization": AUTH}
    cases = [
        ({"corrected_request": "x y z"}, "reviewed tags"),
        ({"corrected_request": "x y z", "corrected_tags": "not_a_tag"}, "unknown tags"),
        ({"corrected_gold": "{}", "corrected_tags": "typo"}, "without a corrected request"),
    ]
    for extra, message in cases:
        with pytest.raises(CorpusError, match=message):
            freeze(
                DRAFT,
                tmp_path / "o",
                decisions_csv=_ledger([{"id": rid, "decision": "correct", **extra}]),
                **kwargs,
            )
    assert not (tmp_path / "o").exists()


def test_cli_records_ai_review_and_never_human(tmp_path, draft):
    csv_path = tmp_path / "ledger.csv"
    csv_path.write_bytes(_ledger([{"id": draft["train"][0]["id"], "decision": "approve"}]))
    out = tmp_path / "cli"
    args = ["freeze", "--corpus", str(DRAFT), "--out", str(out), "--version", "v-cli"]
    args += ["--decisions", str(csv_path), "--reviewer", "technical lead AI"]
    assert cli([*args, "--ai-review-authorization", AUTH]) == 0
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["review"]["reviewer_type"] == "ai"
    assert manifest["review"]["examples_with_a_human_decision"] == 0
    with pytest.raises(SystemExit, match="not both"):
        cli(
            [
                *args[:4],
                str(tmp_path / "x"),
                *args[5:],
                "--human-review",
                "--ai-review-authorization",
                AUTH,
            ]
        )


# --- the committed AI-reviewed frozen corpus -------------------------------------------------

FROZEN = REPO / "data" / "axbench" / "v0.1-ai-reviewed"
LEDGER = DRAFT / "review" / "AI_REVIEW_DECISIONS.csv"
DECISION_URL = "https://github.com/sergthedev/axiom-local/issues/4#issuecomment-7185793594"


def test_committed_frozen_corpus_is_ai_reviewed_and_never_human():
    manifest, splits = load_corpus(FROZEN)
    assert (
        manifest["status"] == "frozen" and manifest["corpus_version"] == "axbench/0.1-ai-reviewed"
    )
    assert {s: len(r) for s, r in splits.items()} == {"train": 400, "validation": 160}
    review = manifest["review"]
    assert review["reviewer_type"] == "ai" and review["authorization"] == DECISION_URL
    assert review["examples_with_a_decision"] == 54
    assert review["examples_with_a_human_decision"] == 0 and not review["attested_human"]
    assert review["removed_ids"] == []
    assert [c["id"] for c in review["corrections"]] == [
        "train-0235",
        "validation-0025",
        "validation-0037",
    ]
    draft_manifest = json.loads((DRAFT / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["parent"]["files"] == draft_manifest["files"]
    records = [r for s in splits.values() for r in s]
    assert all(r["provenance"]["human_review"] == "none" for r in records)
    _, draft = load_corpus(DRAFT)
    gold = {r["id"]: r["gold"] for s in draft.values() for r in s}
    assert all(r["gold"] == gold[r["id"]] for r in records)  # gold unchanged everywhere


def test_committed_frozen_corpus_is_reproducible_from_the_committed_ledger(tmp_path):
    out = tmp_path / "again"
    freeze(
        DRAFT,
        out,
        version="axbench/0.1-ai-reviewed",
        decisions_csv=LEDGER.read_bytes(),
        reviewer="technical lead AI (owner-delegated review)",
        ai_review_authorization=DECISION_URL,
    )
    for name in ("train.jsonl", "validation.jsonl", "DECISIONS.jsonl", "MANIFEST.json"):
        assert (out / name).read_bytes() == (FROZEN / name).read_bytes(), name
