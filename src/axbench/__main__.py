"""AXBENCH command line: ``python -m axbench <command>`` (ADR 0011).

Commands never write the final test split into the repository and never export it for training.

* ``generate --out DIR``: write the development corpus (train, validation) and its manifest;
* ``audit --out FILE``: audit all splits (test generated in memory with the development seed);
* ``review-batch --out DIR``: the Turkish annotation batch for the owner (train/validation only);
* ``rules --split S --out FILE``: run the rule baseline and write raw predictions;
* ``score --split S --predictions FILE --out FILE``: score predictions (not the test split);
* ``export-mlx --out DIR``: chat-format train/valid JSONL for MLX-LM (no test file is written);
* ``seal --out DIR --seed-file FILE [--owner-lineages FILE]``: owner-only sealing of the test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from axiom_router.prompt import build_router_input, chat_messages
from axiom_router.schema import display_name, project_columns

from .audit import audit
from .evaluate import evaluate, rule_router, run_router
from .generate import (
    CORPUS_VERSION,
    GENERATOR_VERSION,
    TARGET_SIZES,
    generate_all_for_audit,
    generate_development_corpus,
)
from .score import load_jsonl


def _dump_jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _dev_records() -> dict[str, list[dict[str, Any]]]:
    return {s: [e.to_record() for e in exs] for s, exs in generate_development_corpus().items()}


def cmd_generate(args) -> int:
    out = Path(args.out)
    corpus = _dev_records()
    manifest: dict[str, Any] = {
        "corpus_version": CORPUS_VERSION,
        "generator_version": GENERATOR_VERSION,
        "status": "draft: unreviewed, not frozen",
        "target_sizes": TARGET_SIZES,
        "files": {},
        "test": "not in the repository; sealed by the owner (python -m axbench seal)",
    }
    for split, records in corpus.items():
        sha = _write(out / f"{split}.jsonl", _dump_jsonl(records))
        manifest["files"][f"{split}.jsonl"] = {"sha256": sha, "n": len(records)}
    _write(out / "MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def cmd_audit(args) -> int:
    corpus = {s: [e.to_record() for e in exs] for s, exs in generate_all_for_audit().items()}
    report = audit(corpus)
    report["note"] = (
        "test split generated in memory with the development seed for this audit; "
        "no test text is included; a sealed test is re-audited at sealing"
    )
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        _write(Path(args.out), text)
    else:
        sys.stdout.write(text)
    return 0 if report["checks"]["passed"] else 1


def _readable_gold(record: dict[str, Any]) -> str:
    names = {c["id"]: display_name(c["name"]) for c in record["schema"]["columns"]}
    g = record["gold"]

    def col(cid: str) -> str:
        return f"{cid} ({names[cid]})"

    if g["action"] == "PLAN":
        op = g["operation"]
        if op == "inspect":
            return f"PLAN inspect: {', '.join(col(c) for c in g['columns'])}"
        if op == "relate":
            return f"PLAN relate: x={col(g['x'])}, y={col(g['y'])}, method={g['method']}"
        if op == "compare":
            return f"PLAN compare: group_by={col(g['group_by'])}, measure={col(g['measure'])}"
        return f"PLAN cluster: features={', '.join(col(c) for c in g['features'])}, k={g['k']}"
    hint = f" (operation hint: {g['operation']})" if g.get("operation") else ""
    cols = f" (columns: {', '.join(col(c) for c in g['columns'])})" if g.get("columns") else ""
    return f"{g['action']} {g['reason']}{hint}{cols}"


def review_selection(
    corpus: dict[str, list[dict[str, Any]]], per_frame: int = 2, n_mixed: int = 8
) -> list[dict[str, Any]]:
    """Deterministic, representative Turkish batch from train and validation (never test)."""
    pool = [r for s in ("train", "validation") for r in corpus[s]]
    chosen: list[dict[str, Any]] = []
    frames = sorted({r["frame"] for r in pool})
    rare_tags = (
        "typo",
        "diacritics_stripped",
        "malicious_header",
        "generic_alias",
        "absent_column",
        "ambiguous_mention",
        "k_word",
        "quoted_name",
    )
    for frame in frames:
        cands = [r for r in pool if r["frame"] == frame and r["language"] == "tr"]
        cands.sort(key=lambda r: (-sum(t in rare_tags for t in r["tags"]), r["id"]))
        picked = []
        for r in cands:  # prefer one train and one validation example, then variety of tags
            if len(picked) == per_frame:
                break
            if (
                picked
                and r["split"] == picked[0]["split"]
                and any(c["split"] != picked[0]["split"] for c in cands)
            ):
                continue
            picked.append(r)
        chosen.extend(picked)
    mixed = sorted((r for r in pool if r["language"] == "mixed"), key=lambda r: r["id"])
    step = max(1, len(mixed) // n_mixed)
    chosen.extend(mixed[::step][:n_mixed])
    return chosen


def cmd_review(args) -> int:
    corpus = _dev_records()
    batch = review_selection(corpus)
    out = Path(args.out)
    rows = []
    md = [
        "# AXBENCH v0.1-draft: Turkish annotation batch for owner review",
        "",
        "Status: **generated by the implementing AI tool from templates; not reviewed by a "
        "human.**",
        "",
        f"{len(batch)} examples from the train and validation splits only (no test example).",
        "For each row, please record a decision (`approve`, `correct`, `reject`) and, if needed,",
        "the corrected gold label and a note on the naturalness of the Turkish wording. Your",
        "decisions are the first human review of this corpus and will be recorded with their",
        "actual count.",
        "",
        "Labeling rules being reviewed are in the dataset card (`data/axbench/DATASET_CARD.md`).",
        "",
    ]
    for i, r in enumerate(batch, start=1):
        cols = "; ".join(
            f"{c['id']}={display_name(c['name'])!r} ({c['type']})" for c in r["schema"]["columns"]
        )
        gold = _readable_gold(r)
        md += [
            f"## {i}. `{r['id']}` ({r['split']}, {r['language']}, frame `{r['frame']}`)",
            "",
            f"- Request: {display_name(r['request'])}",
            f"- Schema `{r['schema_id']}`: {cols}",
            f"- Gold: **{gold}**",
            f"- Tags: {', '.join(r['tags']) or 'none'}",
            "- Decision: ____  Corrected gold: ____  Note: ____",
            "",
        ]
        rows.append(
            {
                "n": i,
                "id": r["id"],
                "split": r["split"],
                "language": r["language"],
                "frame": r["frame"],
                "request": r["request"],
                "schema_id": r["schema_id"],
                "gold": gold,
                "gold_json": json.dumps(r["gold"], ensure_ascii=False),
                "decision": "",
                "corrected_gold": "",
                "note": "",
            }
        )
    _write(out / "owner_review_batch_tr.md", "\n".join(md))
    buf_path = out / "owner_review_batch_tr.csv"
    buf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(buf_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(batch)} examples written")
    return 0


def _split_records(split: str, path: str | None) -> list[dict[str, Any]]:
    if split == "test":
        raise SystemExit("the test split is sealed; use the frozen final-evaluation procedure")
    if path:
        return load_jsonl(path)
    return _dev_records()[split]


def cmd_rules(args) -> int:
    records = _split_records(args.split, args.data)
    preds = run_router(rule_router(), records, system="rules")
    _write(Path(args.out), _dump_jsonl(preds))
    return 0


def cmd_score(args) -> int:
    records = _split_records(args.split, args.data)
    preds = load_jsonl(args.predictions)
    report = evaluate(records, preds, system=args.system, split=args.split)
    _write(Path(args.out), json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return 0


def cmd_export_mlx(args) -> int:
    corpus = _dev_records()
    out = Path(args.out)
    for split, name in (("train", "train.jsonl"), ("validation", "valid.jsonl")):
        lines = []
        for r in corpus[split]:
            view = project_columns([(c["name"], c["type"]) for c in r["schema"]["columns"]])
            msgs = chat_messages(build_router_input(r["request"], view))
            target = json.dumps(r["gold"], ensure_ascii=False, separators=(",", ":"))
            lines.append({"messages": [*msgs, {"role": "assistant", "content": target}]})
        _write(out / name, _dump_jsonl(lines))
    print("wrote train.jsonl and valid.jsonl; no test file is exported")
    return 0


def cmd_seal(args) -> int:
    from .seal import load_owner_lineages, seal

    seed = Path(args.seed_file).read_text(encoding="utf-8").strip()
    owner = load_owner_lineages(args.owner_lineages) if args.owner_lineages else None
    manifest = seal(args.out, seed, owner_lineages=owner)
    print(json.dumps({k: manifest[k] for k in ("test_sha256", "n_test", "audit_passed")}))
    return 0 if manifest["audit_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m axbench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("audit")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_audit)
    p = sub.add_parser("review-batch")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_review)
    for name, fn in (("rules", cmd_rules), ("score", cmd_score)):
        p = sub.add_parser(name)
        p.add_argument("--split", required=True, choices=("train", "validation", "test"))
        p.add_argument("--data", help="split JSONL (default: regenerate the development corpus)")
        p.add_argument("--out", required=True)
        if name == "score":
            p.add_argument("--predictions", required=True)
            p.add_argument("--system", required=True)
        p.set_defaults(fn=fn)
    p = sub.add_parser("export-mlx")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_export_mlx)
    p = sub.add_parser("seal")
    p.add_argument("--out", required=True)
    p.add_argument("--seed-file", required=True)
    p.add_argument("--owner-lineages")
    p.set_defaults(fn=cmd_seal)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
