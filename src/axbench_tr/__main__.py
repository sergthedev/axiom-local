"""Command line: ``python -m axbench_tr <command>``."""

from __future__ import annotations

import argparse
import json
import sys

from .build import CURRENT_VERSION


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="axbench_tr", description="AXBENCH-TR data acquisition and corpus tools"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("acquire", help="download registered sources (raw files stay outside Git)")
    p.add_argument("--source", action="append", help="source id, repeatable (default: all)")
    sub.add_parser("verify", help="re-hash raw files against ACQUISITION.json")
    sub.add_parser("schemas", help="rebuild the committed table-schema inventory from raw files")
    sub.add_parser(
        "economic", help="copy redistributable real observations to data/axbench-tr/economic/"
    )
    p = sub.add_parser("build", help="generate, validate, audit and write the corpus")
    p.add_argument("--version", default=CURRENT_VERSION)
    sub.add_parser("validate", help="validate the committed corpus")
    sub.add_parser("docs", help="regenerate DATA_SOURCES.md from the registry")
    p = sub.add_parser("export-mlx", help="write MLX-LM chat files outside the repository")
    p.add_argument("--out", required=True)
    sub.add_parser("privacy-audit", help="scan committed economic CSVs for identifiers")
    sub.add_parser("licences", help="fetch licence pages and write LICENCE_EVIDENCE.json")
    sub.add_parser(
        "licences-addendum",
        help="fetch dataset-level terms for the probe/smoke subsets, write the addendum",
    )
    sub.add_parser(
        "review-sample", help="write the stratified 390-record review sample (300 Turkish)"
    )
    p = sub.add_parser("review-summary", help="count decisions in a filled-in review CSV")
    p.add_argument("--file", required=True)
    p = sub.add_parser(
        "contamination", help="count overlaps with an owner-supplied JSONL of requests"
    )
    p.add_argument("--file", required=True)
    p.add_argument("--field", default="request")
    p.add_argument("--version", default=CURRENT_VERSION)
    a = ap.parse_args(argv)

    if a.cmd == "acquire":
        from .acquire import acquire

        acquire(a.source)
    elif a.cmd == "verify":
        from .acquire import verify

        problems = verify()
        print("\n".join(problems) or "all raw files match the manifest")
        return 1 if problems else 0
    elif a.cmd == "schemas":
        from .schemas import export_inventory

        print(export_inventory(), "tables written")
    elif a.cmd == "economic":
        from .economic import build

        print(len(build()["files"]), "files")  # type: ignore[arg-type]
    elif a.cmd == "build":
        from .build import build_corpus, distributions, write

        corpus = build_corpus()
        out = write(corpus, a.version)
        print(json.dumps(distributions(corpus)["overall"]), "->", out.name)
    elif a.cmd == "validate":
        from .build import OUT_ROOT
        from .validate import corpus_report

        corpus = {}
        for split in ("train", "validation"):
            path = OUT_ROOT / CURRENT_VERSION / f"{split}.jsonl"
            corpus[split] = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        report = corpus_report(corpus)
        print(json.dumps({k: v for k, v in report.items() if k != "problems"}, indent=1))
        return 1 if report["problem_count"] else 0
    elif a.cmd == "export-mlx":
        from pathlib import Path

        from .export import export

        manifest = export(Path(a.out))
        print(
            json.dumps(
                {k: v["examples"] for k, v in manifest["files"].items() if "examples" in v},
                indent=1,
            )
        )
    elif a.cmd == "privacy-audit":
        from .economic import OUT_DIR, privacy_and_licence_audit

        report = privacy_and_licence_audit()
        (OUT_DIR / "PRIVACY_AUDIT.json").write_text(
            json.dumps(report, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(report["files_scanned"], "files;", len(report["files_with_candidates"]), "flagged")
    elif a.cmd == "licences":
        from .licences import write as write_licences

        print(write_licences().name)
    elif a.cmd == "licences-addendum":
        from .licences_addendum import write as write_addendum

        print(write_addendum().name)
    elif a.cmd == "review-sample":
        from .review import build as build_sample

        print(json.dumps(build_sample()["by_origin"]))
    elif a.cmd == "review-summary":
        from pathlib import Path

        from .review import reviewed_summary

        print(json.dumps(reviewed_summary(Path(a.file)), indent=1))
    elif a.cmd == "docs":
        from .docs import write

        print(write().name)
    elif a.cmd == "contamination":
        from .build import OUT_ROOT
        from .contamination import check_external_file

        reqs = []
        for split in ("train", "validation"):
            path = OUT_ROOT / a.version / f"{split}.jsonl"
            reqs += [json.loads(line)["request"] for line in path.read_text("utf-8").splitlines()]
        print(json.dumps(check_external_file(reqs, a.file, a.field), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
