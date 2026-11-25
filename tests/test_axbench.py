"""AXBENCH tooling tests (ADR 0011): generation, splits, leakage audit, scoring, stats, sealing.

These tests check the tooling, not any model. The test split is generated in memory with the
public development seed only; no test file is written into the repository.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from axbench.__main__ import main as cli
from axbench.__main__ import review_selection
from axbench.audit import audit, gold_problems
from axbench.contamination import contamination_findings
from axbench.evaluate import evaluate, guard_layer, rule_router, run_router, training_lexicon
from axbench.families import FAMILIES, INSTANCES
from axbench.generate import (
    LANGUAGE_SHARES,
    PILOT_SIZES,
    TARGET_SIZES,
    generate_all_for_audit,
    generate_development_corpus,
    generate_split,
)
from axbench.lineages import LINEAGES
from axbench.score import score_one, semantic_key
from axbench.seal import SealError, seal, verify_sealed
from axbench.stats import cluster_bootstrap, mcnemar_exact, paired, wilson
from axiom_router import parse_candidate

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus() -> dict[str, list[dict]]:
    return {s: [e.to_record() for e in exs] for s, exs in generate_all_for_audit().items()}


# --- generation ------------------------------------------------------------------------------


def test_split_sizes_and_language_marginals(corpus):
    for split, n in TARGET_SIZES.items():
        records = corpus[split]
        assert len(records) == n
        lang = Counter(r["language"] for r in records)
        assert lang == {k: round(v * n) for k, v in LANGUAGE_SHARES.items()}
    assert Counter(r["language"] for r in corpus["test"]) == {"tr": 144, "en": 144, "mixed": 72}


def test_all_operations_actions_and_strata_present(corpus):
    for split, records in corpus.items():
        ops = {r["gold"].get("operation") for r in records if r["gold"]["action"] == "PLAN"}
        assert ops == {"inspect", "relate", "compare", "cluster"}, split
        assert {r["gold"]["action"] for r in records} == {"PLAN", "CLARIFY", "REJECT"}
    for split in ("validation", "test"):
        strata = Counter(r["stratum"] for r in corpus[split])
        assert max(strata.values()) - min(strata.values()) <= 1, (split, strata)


def test_generation_is_deterministic():
    a = [e.to_record() for e in generate_development_corpus()["validation"]]
    b = [e.to_record() for e in generate_development_corpus()["validation"]]
    assert a == b


def test_pilot_sizes_are_supported():
    seen: set[str] = set()
    for split in ("train", "validation"):
        assert len(generate_split(split, size=PILOT_SIZES[split], seen=seen)) == PILOT_SIZES[split]


def test_gold_labels_are_valid_candidates_against_their_schema(corpus):
    for records in corpus.values():
        for r in records:
            assert gold_problems(r) == [], r["id"]
            parse_candidate(json.dumps(r["gold"]))  # strict candidate contract


def test_provenance_is_never_labelled_human(corpus):
    for records in corpus.values():
        for r in records:
            assert r["provenance"]["authoring"] == "ai_generated_from_templates"
            assert r["provenance"]["human_review"] == "none"


def test_both_economic_and_non_economic_families_in_every_split(corpus):
    for records in corpus.values():
        domains = {r["domain"] for r in records}
        assert domains == {"economic", "non_economic"}


def test_every_family_has_a_partition_and_instances():
    families = {f.key for f in FAMILIES}
    assert {i.family for i in INSTANCES} == families


# --- splits and leakage ----------------------------------------------------------------------


def test_lineages_and_texts_are_disjoint_across_splits(corpus):
    lineages = {s: {r["lineage_id"] for r in rs} for s, rs in corpus.items()}
    texts = {s: {r["request"].casefold() for r in rs} for s, rs in corpus.items()}
    for a in corpus:
        for b in corpus:
            if a < b:
                assert not lineages[a] & lineages[b]
                assert not texts[a] & texts[b]


def test_lineages_are_split_before_paraphrase():
    by_id = Counter(lin.id for lin in LINEAGES)
    assert max(by_id.values()) == 1
    for lin in LINEAGES:
        assert lin.split in {"train", "validation", "test"}


def test_held_out_families_never_appear_in_training(corpus):
    train_families = {r["family"] for r in corpus["train"]}
    for split in ("validation", "test"):
        held = {r["family"] for r in corpus[split] if r["stratum"] == "A_heldout_family"}
        assert held and not held & train_families
        novel = {r["schema_id"] for r in corpus[split] if r["stratum"] == "B_novel_instance"}
        assert not novel & {r["schema_id"] for r in corpus["train"]}


def test_audit_passes_and_omits_test_texts(corpus):
    report = audit(corpus)
    assert report["checks"]["passed"], report["checks"]["failures"]
    assert report["checks"]["components_spanning_splits"] == 0
    blob = json.dumps(report, ensure_ascii=False)
    for r in corpus["test"][:50]:
        assert r["request"] not in blob


def test_audit_detects_planted_leakage(corpus):
    leaked = {k: list(v) for k, v in corpus.items()}
    planted = dict(leaked["validation"][0], split="train", id="train-9999")
    leaked["train"] = [*leaked["train"], planted]
    report = audit(leaked)
    assert not report["checks"]["passed"]
    assert report["checks"]["components_spanning_splits"] >= 1


def test_audit_detects_invalid_gold(corpus):
    bad = dict(corpus["train"][0])
    bad["gold"] = {"action": "PLAN", "operation": "inspect", "columns": ["c999"]}
    report = audit({"train": [bad], "validation": [], "test": []})
    assert not report["checks"]["passed"]


def test_rule_baseline_has_no_held_out_only_vocabulary():
    assert contamination_findings() == []


def test_rule_lexicon_uses_shared_families_only():
    lexicon = training_lexicon()
    shared = [c for f in FAMILIES if f.partition == "shared" for c in f.concepts]
    assert len(lexicon.groups) == len(shared)
    forms = {form.casefold() for group in lexicon.groups for form in group}
    for held_out_only in ("ball possession", "average delay", "building age"):
        assert held_out_only not in forms


# --- scoring ---------------------------------------------------------------------------------


def _example(gold, columns=None):
    columns = columns or [
        {"id": "c1", "name": "income", "type": "numeric"},
        {"id": "c2", "name": "spending", "type": "numeric"},
        {"id": "c3", "name": "region", "type": "text"},
    ]
    return {
        "id": "x-1",
        "lineage_id": "L-x",
        "gold": gold,
        "accepted_variants": [],
        "schema": {"version": "axiom.schema-view/1", "columns": columns},
        "tags": [],
    }


def _pred(obj, status="ok"):
    return {"status": status, "raw_output": obj if isinstance(obj, str) else json.dumps(obj)}


V = "axiom.candidate/1"


def test_semantic_equivalence_rules():
    rel = {"version": V, "action": "PLAN", "operation": "relate", "x": "c1", "y": "c2"}
    assert semantic_key({**rel, "method": "pearson"}) == semantic_key(
        {**rel, "x": "c2", "y": "c1", "method": "pearson"}
    )
    assert semantic_key({**rel, "method": "pearson"}) != semantic_key({**rel, "method": "spearman"})
    ins = {"version": V, "action": "PLAN", "operation": "inspect"}
    assert semantic_key({**ins, "columns": ["c1", "c2"]}) == semantic_key(
        {**ins, "columns": ["c2", "c1"]}
    )
    clu = {"version": V, "action": "PLAN", "operation": "cluster", "k": 3}
    assert semantic_key({**clu, "features": ["c1", "c2"]}) != semantic_key(
        {**clu, "features": ["c2", "c1"]}
    )
    cmp_ = {"version": V, "action": "PLAN", "operation": "compare"}
    assert semantic_key({**cmp_, "group_by": "c3", "measure": "c1"}) != semantic_key(
        {**cmp_, "group_by": "c1", "measure": "c3"}
    )
    cla = {"version": V, "action": "CLARIFY", "reason": "missing_cluster_count"}
    assert semantic_key(cla) == semantic_key({**cla, "operation": "cluster"})


def test_full_denominator_and_error_stages():
    gold = {"version": V, "action": "REJECT", "reason": "forecasting"}
    ex = _example(gold)
    assert score_one(ex, _pred(gold)).exact_match
    for pred, stage in (
        (_pred(None, status="timeout"), "router"),
        (_pred("not json"), "json"),
        (_pred({"version": V, "action": "REJECT", "reason": "forecasting", "x": 1}), "schema"),
    ):
        s = score_one(ex, pred)
        assert not s.exact_match and s.error_stage == stage


def test_wrong_executable_proposal_counted():
    gold = {"version": V, "action": "CLARIFY", "reason": "missing_correlation_method"}
    wrong = {
        "version": V,
        "action": "PLAN",
        "operation": "relate",
        "x": "c1",
        "y": "c2",
        "method": "pearson",
    }
    s = score_one(_example(gold), _pred(wrong))
    assert s.validator_accepted_plan and s.wrong_executable and not s.exact_match


def test_semantically_invalid_plan_is_not_accepted():
    gold = {"version": V, "action": "PLAN", "operation": "compare"}
    gold |= {"group_by": "c3", "measure": "c1"}
    swapped = {**gold, "group_by": "c1", "measure": "c3"}
    s = score_one(_example(gold), _pred(swapped))
    assert s.json_valid and s.schema_valid and not s.semantic_valid
    assert s.error_stage == "semantic" and not s.validator_accepted_plan


def test_harness_guard_layer_never_dispatches(corpus):
    records = corpus["validation"][:40]
    gold_outputs = [
        {"example_id": r["id"], "status": "ok", "raw_output": json.dumps(r["gold"])}
        for r in records
    ]
    layer = guard_layer(records, gold_outputs)
    assert layer["unauthorized_dispatch_count"] == 0
    assert layer["confirmations_given"] == 0
    n_plans = sum(r["gold"]["action"] == "PLAN" for r in records)
    assert layer["guard_status"].get("awaiting_confirmation", 0) == n_plans
    assert layer["awaiting_confirmation_correct"] == n_plans


def test_rule_baseline_runs_on_training_split(corpus):
    records = corpus["train"][:60]
    preds = run_router(rule_router(), records, system="rules")
    report = evaluate(records, preds, system="rules", split="train", bootstrap_reps=50)
    raw = report["raw_candidate"]
    assert raw["n"] == 60
    assert raw["json_valid"]["count"] == 60
    assert report["after_guard"]["unauthorized_dispatch_count"] == 0


def test_missing_prediction_counts_as_error(corpus):
    records = corpus["train"][:5]
    report = evaluate(records, [], system="none", split="train", bootstrap_reps=20)
    assert report["missing_predictions_counted_as_error"] == 5
    assert report["raw_candidate"]["primary_semantic_exact_match"]["count"] == 0


# --- statistics: known answers ---------------------------------------------------------------


def test_wilson_known_answer():
    # Independent reference: scipy.stats.binomtest(83, 100).proportion_ci(method="wilson")
    from scipy.stats import binomtest

    ref = binomtest(83, 100).proportion_ci(method="wilson")
    lo, hi = wilson(83, 100)
    assert lo == pytest.approx(ref.low, abs=1e-6) and hi == pytest.approx(ref.high, abs=1e-6)
    assert lo == pytest.approx(0.74452, abs=1e-5) and hi == pytest.approx(0.89106, abs=1e-5)
    assert wilson(0, 0) is None
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == pytest.approx(1.0)


def test_mcnemar_exact_known_answer():
    # b=1, c=10: 2 * P[Binom(11, 1/2) <= 1] = 2 * 12 / 2048
    assert mcnemar_exact(1, 10) == pytest.approx(24 / 2048, rel=1e-12)
    from scipy.stats import binomtest

    assert mcnemar_exact(3, 9) == pytest.approx(binomtest(3, 12).pvalue, rel=1e-12)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0


def test_lead_recalculated_hypothetical_values():
    # DR-005 review: hypothetical 300/360 and 30 vs 12 discordant pairs (arithmetic, not AXIOM data)
    lo, hi = wilson(300, 360)
    assert round(100 * lo, 2) == 79.14 and round(100 * hi, 2) == 86.83
    assert mcnemar_exact(12, 30) == pytest.approx(0.0079159, abs=5e-8)


def test_cluster_bootstrap_properties():
    out = cluster_bootstrap([1, 1, 0, 0, 1, 0], ["a", "a", "b", "b", "c", "c"], reps=500, seed=1)
    assert out["n_clusters"] == 3 and out["estimate"] == pytest.approx(0.5)
    lo, hi = out["interval"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    again = cluster_bootstrap([1, 1, 0, 0, 1, 0], ["a", "a", "b", "b", "c", "c"], reps=500, seed=1)
    assert again == out
    p = paired([True, False, True], [True, True, True], ["a", "b", "c"], reps=50)
    assert p["only_b_correct"] == 1 and p["delta_pp"] == pytest.approx(100 / 3)


# --- review batch, CLI guards, MLX export, sealing -------------------------------------------


def test_review_batch_is_turkish_dev_only(corpus):
    dev = {s: corpus[s] for s in ("train", "validation")}
    batch = review_selection(dev)
    assert {r["split"] for r in batch} <= {"train", "validation"}
    frames = {lin.frame for lin in LINEAGES}
    tr = [r for r in batch if r["language"] == "tr"]
    assert {r["frame"] for r in tr} == frames
    assert all(r["language"] in {"tr", "mixed"} for r in batch)


def test_cli_refuses_test_split(tmp_path):
    with pytest.raises(SystemExit, match="sealed"):
        cli(["rules", "--split", "test", "--out", str(tmp_path / "p.jsonl")])


def test_export_mlx_writes_no_test_file(tmp_path):
    assert cli(["export-mlx", "--out", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["train.jsonl", "valid.jsonl"]
    first = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    roles = [m["role"] for m in first["messages"]]
    assert roles == ["system", "user", "assistant"]
    parse_candidate(first["messages"][-1]["content"])


@pytest.fixture(scope="module")
def frozen_corpus(tmp_path_factory):
    from axbench.corpus import freeze

    out = tmp_path_factory.mktemp("frozen") / "corpus"
    freeze(REPO / "data" / "axbench" / "v0.1-draft", out, version="axbench/0.1-unit-frozen")
    return out


def test_seal_refuses_repository_path_and_short_seed(tmp_path, frozen_corpus):
    with pytest.raises(SealError, match="outside the repository"):
        seal(REPO / "data" / "sealed-test", "x" * 32, corpus_dir=frozen_corpus, size=30)
    with pytest.raises(SealError, match="16 characters"):
        seal(tmp_path / "s", "short", corpus_dir=frozen_corpus, size=30)


def test_seal_writes_verifiable_test_outside_repo(tmp_path, frozen_corpus):
    out = tmp_path / "sealed"
    manifest = seal(out, "owner-secret-seed-for-unit-test", corpus_dir=frozen_corpus, size=30)
    assert manifest["n_test"] == 30 and manifest["audit_passed"]
    data = (out / "test.jsonl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest["test_sha256"]
    published = (out / "SEAL.json").read_text(encoding="utf-8")
    first = json.loads(data.decode("utf-8").splitlines()[0])
    assert first["request"] not in published
    assert first["provenance"]["seed_label"] == "owner_secret"
    assert len(verify_sealed(out / "test.jsonl", out / "SEAL.json")) == 30
    (out / "test.jsonl").write_bytes(data + b"\n{}")
    with pytest.raises(SealError, match="does not match"):
        verify_sealed(out / "test.jsonl", out / "SEAL.json")
    with pytest.raises(SealError, match="empty"):
        seal(out, "owner-secret-seed-for-unit-test", corpus_dir=frozen_corpus, size=30)


def test_committed_draft_corpus_matches_generator():
    manifest_path = REPO / "data" / "axbench" / "v0.1-draft" / "MANIFEST.json"
    if not manifest_path.exists():
        pytest.skip("draft corpus not generated")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, meta in manifest["files"].items():
        data = (manifest_path.parent / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == meta["sha256"], name
    assert not (manifest_path.parent / "test.jsonl").exists()
    # the committed bytes are what the generator produces on this platform and in this process
    for split, examples in generate_development_corpus().items():
        text = "".join(
            json.dumps(e.to_record(), ensure_ascii=False, sort_keys=True) + "\n" for e in examples
        )
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert digest == manifest["files"][f"{split}.jsonl"]["sha256"], split
