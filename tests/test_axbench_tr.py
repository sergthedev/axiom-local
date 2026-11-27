"""AXBENCH-TR: registry, acquisition manifest, schemas, generation, validation and committed
corpus."""

from __future__ import annotations

import copy
import io
import json
import re
from pathlib import Path

import pytest

from axbench_tr import acquire as acq
from axbench_tr import adapters, contamination, fetch, registry, schemas, synth, templates, validate
from axbench_tr.build import OUT_ROOT

ROOT = Path(__file__).resolve().parents[1]
CORPUS = OUT_ROOT / "v0.2-draft"


# ----------------------------------------------------------------------------------- registry


def test_registry_loads_and_is_consistent() -> None:
    reg = registry.load_registry()
    ids = [s["id"] for s in reg["sources"]]
    assert len(ids) == len(set(ids)) and len(ids) >= 30
    for src in reg["sources"]:
        # every accepted source has verified derivative rights, or only its column headers are used
        if src["status"].startswith("accepted") and src["status"] != "accepted_schema_only":
            assert src["license"]["derivatives_allowed"] is True, src["id"]
        if src["status"].startswith("rejected_license"):
            assert src["license"]["derivatives_allowed"] is not True, src["id"]


def test_registry_validation_rejects_bad_entries() -> None:
    reg = registry.load_registry()
    broken = copy.deepcopy(reg)
    broken["sources"][0]["status"] = "accepted_adapted"
    broken["sources"][0]["license"]["derivatives_allowed"] = None
    with pytest.raises(registry.RegistryError):
        registry.validate_registry(broken)
    broken = copy.deepcopy(reg)
    del broken["sources"][1]["justification"]
    with pytest.raises(registry.RegistryError):
        registry.validate_registry(broken)
    broken = copy.deepcopy(reg)
    broken["sources"][2]["url"] = "http://example.invalid/"
    with pytest.raises(registry.RegistryError):
        registry.validate_registry(broken)


def test_acquisition_manifest_matches_registry() -> None:
    reg = registry.load_registry()
    manifest = json.loads(registry.ACQUISITION_PATH.read_text("utf-8"))["sources"]
    for src in reg["sources"]:
        if src["acquire"]["method"] == "none":
            assert src["id"] not in manifest
            continue
        entry = manifest[src["id"]]
        assert entry["errors"] == [], src["id"]
        assert entry["files"], src["id"]
        for rec in entry["files"]:
            assert re.fullmatch(r"[0-9a-f]{64}", rec["sha256"]) and rec["bytes"] > 0
            assert rec["url"].startswith("https://") and not rec["path"].startswith("/")
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", rec["retrieved"])


# ---------------------------------------------------------------------------------- fetching


class _FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_download_records_sha256_and_enforces_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"x" * 1000
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload))
    rec = fetch.download("https://example.invalid/a.csv", "sub/a.csv", root=tmp_path)
    assert rec.bytes == 1000 and rec.sha256 == fetch.sha256_bytes(payload)
    assert (tmp_path / "sub" / "a.csv").read_bytes() == payload
    with pytest.raises(fetch.FetchError):
        fetch.download("https://example.invalid/a.csv", "b.csv", root=tmp_path, max_bytes=10)
    assert not (tmp_path / "b.csv").exists()
    assert not list(tmp_path.glob(".part-*")) and not list((tmp_path / "sub").glob(".part-*"))
    for url, path in (
        ("http://example.invalid/", "c"),
        ("https://example.invalid/", "../c"),
        ("https://example.invalid/", "/c"),
    ):
        with pytest.raises(fetch.FetchError):
            fetch.download(url, path, root=tmp_path)


def test_http_files_tolerates_missing_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(url: str, path: str, **_: object) -> fetch.FileRecord:
        if "missing" in url:
            raise fetch.FetchError("404")
        return fetch.FileRecord(path, url, 1, "0" * 64, "2026-11-27")

    monkeypatch.setattr(acq, "download", fake_download)
    files = [
        {"url": "https://e.invalid/ok", "path": "ok"},
        {"url": "https://e.invalid/missing", "path": "m"},
    ]
    assert len(acq._http_files({"files": files, "tolerate_missing": True})) == 1
    with pytest.raises(fetch.FetchError):
        acq._http_files({"files": files})


def test_raw_downloads_are_git_ignored() -> None:
    assert "data/axbench-tr/_raw/" in (ROOT / ".gitignore").read_text("utf-8")


# ------------------------------------------------------------------------------------ schemas


def test_inventory_has_real_tables_and_partitions() -> None:
    tables = schemas.all_tables()
    assert len(tables) > 200
    assert {t.source_id for t in tables} >= {"SRC-O01", "SRC-O02", "SRC-O05", "SRC-O06"}
    parts = {schemas.table_partition(t) for t in tables if schemas.usable(t)}
    assert parts == {"train", "validation"}
    # a table group never spans both partitions
    seen: dict[str, str] = {}
    for t in tables:
        assert seen.setdefault(t.group, schemas.table_partition(t)) == schemas.table_partition(t)
    # the inventory holds names and kinds only: no observation values
    payload = json.loads(schemas.INVENTORY_PATH.read_text("utf-8"))
    assert all(
        set(c) == {"name", "kind", "slot_ok"} for t in payload["tables"] for c in t["columns"]
    )
    # municipal headers are published as-is; only World Bank and TCMB Turkish ones are localized
    assert {t.source_id for t in tables if t.localized} == {"SRC-O01", "SRC-O05"}


def test_slot_rules() -> None:
    assert not schemas.slot_ok("ILCEID", "numeric")
    assert not schemas.slot_ok("ENLEM", "numeric")
    assert schemas.slot_ok("ADET", "numeric")
    assert (
        schemas.time_like("YIL")
        and schemas.time_like("Year")
        and not schemas.time_like("Yıllık ihracat")
    )


# ----------------------------------------------------------------------------- templates


def test_every_frame_has_train_and_validation_lineages() -> None:
    frames: dict[str, set[str]] = {}
    for lin in templates.LINEAGES:
        frames.setdefault(lin.frame, set()).add(templates.split_of(lin.id))
    assert all(s == {"train", "validation"} for s in frames.values()), frames
    assert len({lin.id for lin in templates.LINEAGES}) == len(templates.LINEAGES)


def test_templates_are_not_copies_of_axbench_v01() -> None:
    report = contamination.template_report()
    assert report["pairs_at_or_above_0.8"] == 0, report["closest"]


# --------------------------------------------------------------------------------- generation


@pytest.fixture(scope="module")
def sample() -> dict[str, list[dict]]:
    tables = schemas.all_tables()
    return {
        s: synth.generate(tables, s, n, "test-seed")
        for s, n in (("train", 400), ("validation", 150))
    }


def test_generation_is_deterministic_and_valid(sample: dict[str, list[dict]]) -> None:
    again = synth.generate(schemas.all_tables(), "validation", 150, "test-seed")
    assert again == sample["validation"]
    report = validate.corpus_report(sample)
    assert report["problem_count"] == 0, report["problems"][:5]


def test_language_and_frame_allocation(sample: dict[str, list[dict]]) -> None:
    recs = sample["train"]
    assert sum(r["language"] == "tr" for r in recs) == 320
    actions = {
        a: sum(r["gold"]["action"] == a for r in recs) for a in ("PLAN", "CLARIFY", "REJECT")
    }
    assert 0.48 <= actions["PLAN"] / 400 <= 0.58 and 0.18 <= actions["REJECT"] / 400 <= 0.25


def test_split_isolation(sample: dict[str, list[dict]]) -> None:
    train_tables = {r["family"] for r in sample["train"]}
    val_tables = {r["family"] for r in sample["validation"]}
    assert not train_tables & val_tables
    assert not {r["lineage_id"] for r in sample["train"]} & {
        r["lineage_id"] for r in sample["validation"]
    }


# ---------------------------------------------------------------------- validation (mutations)


def test_validator_catches_corrupted_records(sample: dict[str, list[dict]]) -> None:
    plan = next(r for r in sample["train"] if r["gold"].get("operation") == "relate")
    assert validate.record_problems(plan) == []
    bad = copy.deepcopy(plan)
    bad["gold"]["x"] = "c999"
    assert any("gold" in p for p in validate.record_problems(bad))
    bad = copy.deepcopy(plan)
    bad["request"] = "completely unrelated request"
    assert any(p.startswith("mention_not_in_request") for p in validate.record_problems(bad))
    bad = copy.deepcopy(plan)
    bad["gold"]["y"] = bad["gold"]["x"]
    assert validate.record_problems(bad)
    bad = copy.deepcopy(plan)
    bad["provenance"]["human_review"] = "owner_reviewed"
    assert "human_review_claim_without_evidence" in validate.record_problems(bad)
    del bad["gold"]
    assert any(p.startswith("missing_key") for p in validate.record_problems(bad))


def test_corpus_report_flags_cross_split_leakage(sample: dict[str, list[dict]]) -> None:
    leaked = {
        "train": copy.deepcopy(sample["train"]),
        "validation": copy.deepcopy(sample["validation"]),
    }
    clone = copy.deepcopy(leaked["train"][0])
    clone["id"], clone["split"] = "leak-1", "validation"
    leaked["validation"].append(clone)
    report = validate.corpus_report(leaked)
    kinds = {p.split(":", 1)[0] for p in report["problems"] if p.startswith("cross_split")}
    assert {"cross_split_lineages", "cross_split_request_texts"} <= kinds


# -------------------------------------------------------------------------------------- adapters


def test_parse_create_and_unsafe_definitions() -> None:
    ok = adapters.parse_create(
        "CREATE TABLE IF NOT EXISTS sales.t "
        "(id INT PRIMARY KEY, region VARCHAR(20), amt DECIMAL(10,2));"
    )
    assert ok == ("t", [("id", "numeric"), ("region", "text"), ("amt", "numeric")])
    assert adapters.parse_create("CREATE TABLE t (a INT, b VARCHAR(5), 12000), (3);") is None
    assert adapters.parse_create("no table here") is None


def _row(sql: str, context: str = "CREATE TABLE t (region VARCHAR(10), amt INT);") -> dict:
    return {
        "id": 7,
        "domain": "retail",
        "sql": sql,
        "sql_context": context,
        "tr_prompt": "Bölgeye göre toplam tutar nedir?",
        "sql_prompt": "What is the total amount by region?",
    }


def test_cognipeer_record_derivation_and_validity() -> None:
    cols = adapters.parse_create(_row("")["sql_context"])[1]
    rec = adapters._cognipeer_record(
        _row(""),
        {"op": "compare", "group_by": "region", "measure": "amt", "agg": "SUM", "cols": cols},
        "tr",
        "Bölgeye göre toplam tutar nedir?",
        "compare",
    )
    assert rec["gold"] == {
        "action": "PLAN",
        "group_by": "c1",
        "measure": "c2",
        "operation": "compare",
        "version": "axiom.candidate/1",
    }
    assert validate.record_problems(rec) == []
    assert "SELECT" not in json.dumps(rec)  # SQL is never stored
    mod = adapters._cognipeer_record(
        _row(""),
        {"op": "data_modification", "cols": cols},
        "en",
        "Delete all rows",
        "data_modification",
    )
    assert mod["gold"]["reason"] == "data_modification" and validate.record_problems(mod) == []


def test_sql_patterns_are_strict() -> None:
    assert adapters._GROUP.match("SELECT region, SUM(amt) FROM t GROUP BY region;")
    assert not adapters._GROUP.match(
        "SELECT region, SUM(amt) FROM t WHERE amt > 1 GROUP BY region;"
    )
    assert not adapters._GROUP.match(
        "SELECT region, SUM(amt) FROM t JOIN u ON 1=1 GROUP BY region;"
    )
    assert adapters._AGG.match("SELECT AVG(amt) AS a FROM t;")
    assert not adapters._AGG.match("SELECT AVG(amt) FROM t WHERE x = 1;")
    assert adapters._MOD.match("UPDATE t SET a = 1") and not adapters._MOD.match("SELECT 1")


def test_adapters_read_only_registered_training_files() -> None:
    reg = {s["id"]: s for s in registry.load_registry()["sources"]}
    assert adapters.COGNIPEER_FILE.endswith(reg["SRC-N01"]["acquire"]["files"][1])
    assert adapters.FINANCE_FILE.endswith(reg["SRC-N03"]["acquire"]["files"][1])
    for sid in ("SRC-N13", "SRC-N14"):  # external benchmarks: no test split is ever acquired
        assert not any("test" in f or "dev" in f for f in reg[sid]["acquire"]["files"]), sid


# ------------------------------------------------------------------------- committed corpus


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")
def test_committed_corpus_matches_manifest_and_validates() -> None:
    manifest = json.loads((CORPUS / "MANIFEST.json").read_text("utf-8"))
    corpus = {}
    for split in ("train", "validation"):
        data = (CORPUS / f"{split}.jsonl").read_bytes()
        meta = manifest["files"][f"{split}.jsonl"]
        assert fetch.sha256_bytes(data) == meta["sha256"]
        corpus[split] = [json.loads(line) for line in data.decode("utf-8").splitlines()]
        assert len(corpus[split]) == meta["records"]
    report = validate.corpus_report(corpus)
    assert report["problem_count"] == 0, report["problems"][:5]
    total = sum(map(len, corpus.values()))
    assert 5000 <= total <= 10000
    tr = sum(r["language"] == "tr" for v in corpus.values() for r in v) / total
    assert 0.75 <= tr <= 0.85
    audit = json.loads((CORPUS / "AUDIT.json").read_text("utf-8"))
    v01 = audit["contamination_against_axbench_v0.1_train_and_validation"]
    assert v01["exact_normalized_matches"] == 0 and v01["near_duplicate_pairs"] == 0
    assert manifest["registry_sha256"] == fetch.sha256_file(registry.REGISTRY_PATH)
    assert manifest["acquisition_manifest_sha256"] == fetch.sha256_file(registry.ACQUISITION_PATH)


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")
def test_committed_files_hold_no_absolute_local_paths_or_secrets() -> None:
    pattern = re.compile(
        r"/(?:home|Users|root|tmp)/[\w.-]+|[A-Za-z]:\\\\Users|hf_[A-Za-z0-9]{20,}|api[_-]?key", re.I
    )
    for path in [
        *CORPUS.glob("*"),
        *(OUT_ROOT / "sources").glob("*"),
        *(OUT_ROOT / "economic").glob("*.json"),
    ]:
        text = path.read_text("utf-8")
        assert not pattern.search(text), f"{path.name}: {pattern.search(text).group(0)}"


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")
def test_committed_corpus_does_not_overlap_v01_public_splits() -> None:
    reqs = [
        json.loads(line)["request"]
        for s in ("train", "validation")
        for line in (CORPUS / f"{s}.jsonl").read_text("utf-8").splitlines()
    ]
    report = contamination.request_report(reqs, contamination.load_v01_requests())
    assert report["exact_normalized_matches"] == 0 and report["near_duplicate_pairs"] == 0


def test_near_duplicate_detector_finds_known_pairs() -> None:
    hits = contamination.near_duplicates(
        ["Show me the mean of income by region please"],
        ["show me the mean of income by region, please!", "something else entirely"],
    )
    assert [(i, j) for i, j, _ in hits] == [(0, 0)]
    assert (
        contamination.near_duplicates(["alpha beta gamma"], ["completely different words here"])
        == []
    )


def test_column_names_drop_leading_byte_order_marks() -> None:
    from axbench_tr.schemas import _col

    assert _col("﻿BARAJ_METRE_KUP", "text").name == "BARAJ_METRE_KUP"


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")
def test_committed_corpus_has_no_byte_order_marks() -> None:
    for split in ("train", "validation"):
        assert "﻿" not in (CORPUS / f"{split}.jsonl").read_text("utf-8")


def test_data_sources_md_is_generated_from_registry() -> None:
    from axbench_tr import docs

    assert docs.DATA_SOURCES_PATH.read_text("utf-8") == docs.render()
