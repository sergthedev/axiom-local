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
from axbench_tr import (
    adapters,
    contamination,
    fetch,
    intent,
    registry,
    schemas,
    synth,
    templates,
    validate,
)
from axbench_tr.build import CURRENT_VERSION, OUT_ROOT

ROOT = Path(__file__).resolve().parents[1]
CORPUS = OUT_ROOT / CURRENT_VERSION
DRAFT = OUT_ROOT / "v0.2-draft"  # the first, superseded corpus, kept unchanged as a record


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
    by_id = {s["id"]: s for s in reg["sources"]}
    assert by_id["SRC-N03"]["status"] == "quarantined_licence_unresolved"  # MIT vs Apache conflict


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
        {"op": "compare", "group_by": "region", "measure": "amt", "agg": "AVG", "cols": cols},
        "tr",
        "Bölgeye göre ortalama tutar nedir?",
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
    assert adapters._GROUP.match("SELECT region, AVG(amt) FROM t GROUP BY region;")
    # SUM and COUNT are not Core descriptive kernels, so they are never accepted
    assert not adapters._GROUP.match("SELECT region, SUM(amt) FROM t GROUP BY region;")
    assert not adapters._GROUP.match("SELECT region, COUNT(amt) FROM t GROUP BY region;")
    assert not adapters._AGG.match("SELECT SUM(amt) AS a FROM t;")
    assert not adapters._GROUP.match(
        "SELECT region, AVG(amt) FROM t WHERE amt > 1 GROUP BY region;"
    )
    assert not adapters._GROUP.match(
        "SELECT region, AVG(amt) FROM t JOIN u ON 1=1 GROUP BY region;"
    )
    assert adapters._AGG.match("SELECT AVG(amt) AS a FROM t;")
    assert not adapters._AGG.match("SELECT AVG(amt) FROM t WHERE x = 1;")
    assert adapters._MOD.match("UPDATE t SET a = 1") and not adapters._MOD.match("SELECT 1")


def test_adapters_read_only_registered_training_files() -> None:
    reg = {s["id"]: s for s in registry.load_registry()["sources"]}
    assert adapters.COGNIPEER_FILE.endswith(reg["SRC-N01"]["acquire"]["files"][1])
    assert adapters.FINANCE_FILE.endswith(reg["SRC-N03"]["acquire"]["files"][1])
    build_source = (Path(adapters.__file__).with_name("build.py")).read_text("utf-8")
    assert "finance_negatives(" not in build_source  # SRC-N03 is quarantined: never called
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
        *(p for p in CORPUS.rglob("*") if p.is_file()),
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


REVIEW = CORPUS / "review"


@pytest.mark.skipif(not REVIEW.exists(), reason="review sample not built")
def test_review_sample_is_deterministic_stratified_and_turkish() -> None:
    from axbench_tr import review

    rows, hashes = review.load_corpus()
    first = review.sample_rows(review.select(rows))
    again = review.sample_rows(review.select(list(reversed(rows))))  # input order must not matter
    assert [r["record_id"] for r in first] == [r["record_id"] for r in again]
    assert len(first) == review.SAMPLE_SIZE
    assert {r["record_id"] for r in first} <= {r["id"] for r in rows}
    assert {r["language"] for r in first} == {"tr", "en"}
    assert {(r["language"], r["origin"]) for r in first} == set(review.ALLOCATION)
    assert {r["frame"] for r in first} >= {"inspect", "relate", "compare", "cluster"}
    assert all(r["reviewer_decision"] == "" for r in first)  # nothing pre-filled
    committed = [
        json.loads(line)
        for line in (REVIEW / "REVIEW_SAMPLE.jsonl").read_text("utf-8").splitlines()
    ]
    assert [r["record_id"] for r in committed] == [r["record_id"] for r in first]
    manifest = json.loads((REVIEW / "REVIEW_SAMPLE_MANIFEST.json").read_text("utf-8"))
    assert manifest["corpus_files_sha256"] == hashes
    for name, digest in manifest["files"].items():
        assert fetch.sha256_file(REVIEW / name) == digest


def test_review_allocation_respects_floor_and_total() -> None:
    from axbench_tr.review import _allocate

    alloc = _allocate({"a": 1000, "b": 10, "c": 2}, 100)
    assert sum(alloc.values()) == 100
    assert alloc["c"] == 2 and alloc["b"] >= 3


def test_gold_is_readable_for_every_action() -> None:
    from axbench_tr.review import readable_gold

    schema = {"columns": [{"id": "c1", "name": "Ülke", "type": "text"}]}
    plan = {"schema": schema, "gold": {"action": "PLAN", "operation": "inspect", "columns": ["c1"]}}
    assert readable_gold(plan) == "PLAN inspect: Ülke (c1)"
    reject = {"schema": schema, "gold": {"action": "REJECT", "reason": "forecasting"}}
    assert readable_gold(reject) == "REJECT: forecasting"


def test_licence_evidence_covers_every_used_source() -> None:
    path = OUT_ROOT / "sources" / "LICENCE_EVIDENCE.json"
    evidence = json.loads(path.read_text("utf-8"))
    assert all(e["status"] == "excerpt_found" for e in evidence["entries"])
    covered = {s for e in evidence["entries"] for s in e["sources"]}
    used = {"SRC-N01", "SRC-O01", "SRC-O02", "SRC-O06"}
    assert used <= covered


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")
def test_audit_reports_near_duplicates_and_no_cross_split_close_pairs() -> None:
    audit = json.loads((CORPUS / "AUDIT.json").read_text("utf-8"))
    nd = audit["near_duplicates_in_corpus"]
    # the build removes validation lineages at cosine >= 0.9 to a train record
    assert nd["cross_split_pairs"][">=0.9"] == 0 and nd["cross_split_pairs"][">=0.95"] == 0
    assert audit["validation"]["cross_split_shared"]["request_texts"] == 0


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")
def test_every_sequence_fits_the_declared_limit_without_truncation() -> None:
    rows = [
        json.loads(line)
        for line in (CORPUS / "token_lengths_per_example.jsonl").read_text("utf-8").splitlines()
    ]
    manifest = json.loads((CORPUS / "MANIFEST.json").read_text("utf-8"))
    assert len(rows) == sum(f["records"] for f in manifest["files"].values())
    assert max(r["sequence_tokens"] for r in rows) <= 2048
    assert all(r["prompt_tokens"] + r["completion_tokens"] == r["separate_tokens"] for r in rows)
    assert all(r["sequence_tokens"] >= r["separate_tokens"] for r in rows)
    assert not any(r["boundary_straddle"] for r in rows)


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")
def test_mlx_export_is_deterministic_chat_format_and_refuses_repo_paths(tmp_path: Path) -> None:
    from axbench_tr import export

    with pytest.raises(SystemExit):
        export.export(OUT_ROOT / "export-here")
    first = export.export(tmp_path / "a")
    second = export.export(tmp_path / "b")
    assert first["files"] == second["files"]
    assert first["files"]["full/train.jsonl"]["examples"] == len(
        (CORPUS / "train.jsonl").read_text("utf-8").splitlines()
    )
    assert first["files"]["smoke/train.jsonl"]["examples"] == export.SMOKE_TRAIN
    assert first["files"]["peak_probe/train.jsonl"]["examples"] == export.PROBE
    assert max(first["peak_probe_lengths"]) <= 2048
    for line in (tmp_path / "a" / "smoke" / "train.jsonl").read_text("utf-8").splitlines():
        roles = [m["role"] for m in json.loads(line)["messages"]]
        assert roles == ["system", "user", "assistant"]
    assert not list((tmp_path / "a").rglob("test*"))


# ------------------------------------------------------------ intent: what a plan may say


def _cmp(agg: str) -> dict[str, object]:
    return {"op": "compare", "group_by": "region", "measure": "amount", "agg": agg}


def _ins(column: str, agg: str) -> dict[str, object]:
    return {"op": "inspect", "column": column, "agg": agg}


@pytest.mark.parametrize(
    ("prompt", "basis"),
    [
        ("What is the average amount for each region?", _cmp("AVG")),
        ("Show the maximum amount per region.", _cmp("MAX")),
        ("What is the lowest price?", _ins("price", "MIN")),
    ],
)
def test_intent_accepts_plain_summaries(prompt: str, basis: dict[str, object]) -> None:
    assert intent.english_problem(prompt, basis) is None


@pytest.mark.parametrize(
    ("prompt", "basis", "reason"),
    [
        # SUM is not a Core descriptive kernel
        ("What is the total amount for each region?", _cmp("SUM"), "unsupported_aggregate"),
        ("How many amounts are there for each region?", _cmp("COUNT"), "unsupported_aggregate"),
        # a value filter, a year, a named place, a clause about other rows
        ("What is the average amount for each region with amount above fifty?", _cmp("AVG"), None),
        ("What is the average amount for each region in 2020?", _cmp("AVG"), None),
        ("What is the average amount for each Region in France?", _cmp("AVG"), None),
        (
            "What is the average price of products priced above the mean?",
            _ins("price", "AVG"),
            None,
        ),
        # a ranking asks for an entity, not a summary
        ("Which region has the highest amount?", _cmp("MAX"), None),
        # the wrong measure, or the wrong aggregate word
        ("What is the average salary for each region?", _cmp("AVG"), None),
        ("What is the average cost?", _ins("price", "AVG"), None),
        ("What is the maximum amount for each region?", _cmp("AVG"), None),
    ],
)
def test_intent_rejects_what_the_plan_would_misstate(
    prompt: str, basis: dict[str, object], reason: str | None
) -> None:
    got = intent.english_problem(prompt, basis)
    assert got is not None, prompt
    if reason:
        assert got == reason


def test_intent_translation_must_keep_the_aggregate_and_add_no_constraint() -> None:
    assert intent.turkish_problem("Her bölge için ortalama tutar nedir?", _cmp("AVG")) is None
    # a translation that turned an average into a total, or lost the aggregate, is dropped
    assert intent.turkish_problem("Her bölge için toplam tutar nedir?", _cmp("AVG"))
    assert intent.turkish_problem("Ortalama tutar nedir?", _cmp("MAX"))
    assert intent.turkish_problem("Her bölge için tutar nedir?", _cmp("AVG"))
    # an added number or a lost group marker
    assert intent.turkish_problem(
        "Her bölge için ortalama tutar 50'den fazla olanlar?", _cmp("AVG")
    )
    assert intent.turkish_problem("Ortalama tutar nedir?", _cmp("AVG"))  # group marker missing
    assert intent.turkish_problem("Her bölge için toplam tutar nedir?", _cmp("SUM")) == (
        "total_count_or_ranking_word"
    )


def test_modification_prompts_must_ask_for_a_change() -> None:
    assert intent.modification_problem("Delete all rows older than the cutoff") is None
    assert intent.modification_problem("Which rows were deleted?")  # a question about the past
    assert intent.modification_problem("Create a table of sales")  # not a data change
    assert intent.modification_problem("Remove the old rows?")  # a question


# ---------------------------------------------------------- committed corpus: adapted records


def _corpus_records() -> list[dict]:
    return [
        json.loads(line)
        for split in ("train", "validation")
        for line in (CORPUS / f"{split}.jsonl").read_text("utf-8").splitlines()
    ]


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")
def test_adapted_plans_use_only_core_summaries_and_bind_named_columns() -> None:
    """Independent of the adapter: re-derive each adapted PLAN from what the record itself says."""
    adapted = [r for r in _corpus_records() if r["provenance"]["origin"] == "external_adapted"]
    plans = [r for r in adapted if r["gold"]["action"] == "PLAN"]
    assert plans and {r["gold"]["operation"] for r in plans} == {"inspect", "compare"}
    for rec in plans:
        agg = rec["provenance"]["gold_derivation"]["aggregate"]
        assert agg in {"AVG", "MIN", "MAX"}, rec["id"]  # never SUM or COUNT
        names = {c["id"]: c["name"] for c in rec["schema"]["columns"]}
        kinds = {c["id"]: c["type"] for c in rec["schema"]["columns"]}
        gold = rec["gold"]
        if gold["operation"] == "inspect":
            bound = [gold["columns"][0]]
            basis = {"op": "inspect", "column": names[bound[0]], "agg": agg}
            assert kinds[bound[0]] == "numeric"
        else:
            bound = [gold["group_by"], gold["measure"]]
            basis = {
                "op": "compare",
                "group_by": names[gold["group_by"]],
                "measure": names[gold["measure"]],
                "agg": agg,
            }
            assert kinds[gold["group_by"]] == "text" and kinds[gold["measure"]] == "numeric"
        assert len(set(bound)) == len(bound)
        check = intent.english_problem if rec["language"] == "en" else intent.turkish_problem
        assert check(rec["request"], basis) is None, (rec["id"], rec["request"])
        assert not re.search(r"\d", rec["request"]), rec["id"]
    for rec in adapted:
        if rec["gold"]["action"] == "REJECT":
            assert rec["gold"]["reason"] == "data_modification"
            assert not rec["request"].rstrip().endswith("?"), rec["id"]


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")
def test_audit_records_why_upstream_rows_were_excluded() -> None:
    audit = json.loads((CORPUS / "AUDIT.json").read_text("utf-8"))["external_adapter_audit"]
    reasons = audit["excluded_by_reason"]
    assert reasons["select:shape_not_a_supported_summary"] > 0  # SUM/COUNT/filtered SELECTs
    assert any(k.startswith("compare:extra") or k.startswith("inspect:") for k in reasons)
    assert audit["qualified_upstream_rows"]["compare"] > 0


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")
def test_no_schema_fingerprint_or_family_is_shared_across_splits_for_any_origin() -> None:
    by_split: dict[str, dict[str, set[str]]] = {"fingerprint": {}, "family": {}}
    for rec in _corpus_records():
        by_split["fingerprint"].setdefault(validate.schema_fingerprint(rec), set()).add(
            rec["split"]
        )
        by_split["family"].setdefault(rec["family"], set()).add(rec["split"])
    for kind, groups in by_split.items():
        shared = [k for k, v in groups.items() if len(v) > 1]
        assert not shared, (kind, shared[:3])


def test_corpus_report_flags_a_shared_schema_fingerprint_across_origins(
    sample: dict[str, list[dict]],
) -> None:
    leaked = {k: copy.deepcopy(v) for k, v in sample.items()}
    clone = copy.deepcopy(leaked["train"][0])
    clone.update({"id": "leak-2", "split": "validation", "lineage_id": "L-other"})
    clone["family"] = "cognipeer/other"
    clone["request"] = "A completely different sentence about nothing in particular"
    leaked["validation"].append(clone)
    report = validate.corpus_report(leaked)
    assert report["cross_split_shared"]["schema_fingerprints"] >= 1


# ------------------------------------------------------------- the superseded draft stays as it was


@pytest.mark.skipif(not DRAFT.exists(), reason="draft not present")
def test_original_draft_is_preserved_unchanged() -> None:
    manifest = json.loads((DRAFT / "MANIFEST.json").read_text("utf-8"))
    assert manifest["corpus_version"] == "axbench-tr/0.2-draft"
    total = 0
    for split in ("train", "validation"):
        data = (DRAFT / f"{split}.jsonl").read_bytes()
        assert fetch.sha256_bytes(data) == manifest["files"][f"{split}.jsonl"]["sha256"]
        total += manifest["files"][f"{split}.jsonl"]["records"]
    assert total == 8625
    assert CURRENT_VERSION != "v0.2-draft"


def test_the_sealed_test_is_never_touched_by_the_tooling() -> None:
    sources = "".join(p.read_text("utf-8") for p in (ROOT / "src" / "axbench_tr").glob("*.py"))
    assert "contamination --sealed" not in sources
    assert not any(
        "sealed" in str(p).lower() for p in (ROOT / "data" / "axbench-tr").rglob("*.jsonl")
    )


# ----------------------------------------------- adversarial mutations of accepted requests

_BASE_EN = "What is the average amount for each region?"
_BASE_TR = "Her bölge için ortalama tutar nedir?"


@pytest.mark.parametrize(
    "mutated",
    [
        "What is the average amount for each region only?",  # a qualifier
        "What is the average amount for each region excluding north?",  # exclusion
        "What is the average amount for each region but not the south?",  # negation
        "What is the average amount for each region except east?",
        "What is the average amount for each region without outliers?",
        "What is the average amount for each region where amount is above fifty?",  # filter
        "What is the average amount for each region with amount over one hundred?",
        "What is the average of the top amount for each region?",  # ranking word
        "What is the average amount for each region and each year?",  # an extra group
        "What is the average large amount for each region?",  # an extra modifier word
        "What is the average salary for each region?",  # wrong measure
        "What is the average amount for each country?",  # wrong group
        "What is the average amount for each region not in the west",
    ],
)
def test_english_gate_rejects_mutated_requests(mutated: str) -> None:
    assert intent.english_problem(_BASE_EN, _cmp("AVG")) is None
    assert intent.english_problem(mutated, _cmp("AVG")) is not None, mutated


def test_english_gate_matches_whole_words_not_five_character_prefixes() -> None:
    inspect_price = _ins("price", "AVG")
    assert intent.english_problem("What is the average price?", inspect_price) is None
    assert intent.english_problem("What is the average prices?", inspect_price) is None
    for wrong in ("What is the average priced?", "What is the average pricey?"):
        assert intent.english_problem(wrong, inspect_price) is not None, wrong


@pytest.mark.parametrize(
    "mutated",
    [
        "Her bölge için ortalama tutar nedir, kuzey hariç?",  # exclusion
        "Her bölge için ortalama tutar kuzey olmayan?",  # negation
        "Her bölge için ortalama tutarı güneyde olan?",  # a filter
        "Her bölge için ortalama tutar elli üzerinde?",  # lowercase threshold
        "Her bölge için ortalama tutar yalnızca kuzey?",  # only
        "Her bölge için en iyi ortalama tutar nedir?",  # ranking
        "Her bölge için toplam tutar nedir?",  # a total
        "Her bölge için ortalama tutar değil mi?",  # negation
    ],
)
def test_turkish_gate_rejects_negation_filters_and_rankings(mutated: str) -> None:
    assert intent.turkish_problem(_BASE_TR, _cmp("AVG")) is None
    assert intent.turkish_problem(mutated, _cmp("AVG")) is not None, mutated


def test_turkish_gate_cannot_see_a_wrong_translated_measure_or_group() -> None:
    """Known limit, asserted so nobody claims otherwise: the gate checks the aggregate word and
    the group marker, not the translated column words. Such records are tagged and kept out of the
    smoke subset instead."""
    wrong_measure = "Her bölge için ortalama maaş nedir?"  # "salary" for a column named amount
    wrong_group = "Her ülke için ortalama tutar nedir?"  # "country" for a column named region
    assert intent.turkish_problem(wrong_measure, _cmp("AVG")) is None
    assert intent.turkish_problem(wrong_group, _cmp("AVG")) is None


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")
def test_every_external_turkish_plan_is_tagged_unverified() -> None:
    plans = [
        r
        for r in _corpus_records()
        if r["provenance"]["origin"] == "external_adapted"
        and r["language"] == "tr"
        and r["gold"]["action"] == "PLAN"
    ]
    assert plans and all("translation_unverified" in r["tags"] for r in plans)
    english = [r for r in _corpus_records() if r["language"] == "en"]
    assert not any("translation_unverified" in r["tags"] for r in english)


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")
def test_smoke_and_probe_exports_hold_only_rights_cleared_template_rows(tmp_path: Path) -> None:
    from axbench_tr import export

    manifest = export.export(tmp_path)
    ids = json.loads((tmp_path / "ROW_IDS.json").read_text("utf-8"))
    origin = {r["id"]: r["provenance"]["origin"] for r in _corpus_records()}
    tags = {r["id"]: r["tags"] for r in _corpus_records()}
    for name in ("smoke/train.jsonl", "smoke/valid.jsonl", "peak_probe/train.jsonl"):
        assert {origin[i] for i in ids[name]} == {"synthetic_template"}, name
        assert not any("translation_unverified" in tags[i] for i in ids[name])
    assert "external_adapted" in manifest["rights_cleared_subsets"]["full"]


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")
def test_probe_and_smoke_use_only_schema_sources_with_documented_terms(tmp_path: Path) -> None:
    from axbench_tr import export
    from axbench_tr.licences_addendum import ADDENDUM_PATH, coverage_problems, tables_used

    assert export.RIGHTS_VERIFIED_SCHEMA_SOURCES == ("SRC-O01", "SRC-O02", "SRC-O06")
    export.export(tmp_path)
    ids = json.loads((tmp_path / "ROW_IDS.json").read_text("utf-8"))
    records = {r["id"]: r for r in _corpus_records()}
    for name in ("smoke/train.jsonl", "smoke/valid.jsonl", "peak_probe/train.jsonl"):
        sources = {records[i]["provenance"]["schema_source_id"] for i in ids[name]}
        assert sources <= set(export.RIGHTS_VERIFIED_SCHEMA_SOURCES), (name, sources)
    # no overlap between the probe and the smoke training rows; validation stays validation
    assert not set(ids["peak_probe/train.jsonl"]) & set(ids["smoke/train.jsonl"])
    assert {records[i]["split"] for i in ids["smoke/train.jsonl"]} == {"train"}
    assert {records[i]["split"] for i in ids["smoke/valid.jsonl"]} == {"validation"}
    assert len(ids["smoke/train.jsonl"]) == export.SMOKE_TRAIN
    assert len(ids["smoke/valid.jsonl"]) == export.SMOKE_VALID
    # the committed addendum documents the terms of every dataset those rows use
    addendum = json.loads(ADDENDUM_PATH.read_text("utf-8"))
    assert addendum["tables_used"] == tables_used()
    assert coverage_problems(addendum, tables_used()) == []


def test_addendum_does_not_modify_the_original_licence_evidence() -> None:
    import hashlib

    from axbench_tr.licences import EVIDENCE_PATH

    digest = hashlib.sha256(EVIDENCE_PATH.read_bytes()).hexdigest()
    assert digest == "9304939da99058fe8ade25f6a0829304c388295956d81318a7fee01b063b1337"


def test_addendum_coverage_names_a_missing_or_unverified_source() -> None:
    from axbench_tr.licences_addendum import coverage_problems

    addendum = {
        "entries": [
            {"id": "AD-01", "scope": "portal", "source": "SRC-O01", "status": "excerpt_found"},
            {"id": "AD-04", "scope": "portal", "source": "SRC-O02", "status": "excerpt_found"},
            {"id": "AD-06", "scope": "portal", "source": "SRC-O06", "status": "excerpt_not_found"},
            {
                "id": "IZ-a",
                "scope": "dataset",
                "source": "SRC-O06",
                "status": "needs_review",
                "tables": ["izmir/a/1"],
            },
        ]
    }
    problems = coverage_problems(addendum, {"SRC-O03": ["ecb/x"], "SRC-O06": ["izmir/a/1"]})
    assert any("SRC-O03" in p for p in problems)
    assert any("AD-06" in p for p in problems)
    assert any("izmir/a/1" in p for p in problems)
