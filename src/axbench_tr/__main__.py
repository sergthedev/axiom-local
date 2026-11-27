"""Command line: ``python -m axbench_tr <command>``."""

from __future__ import annotations

import argparse
import json
import sys


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
    p.add_argument("--version", default="v0.2-draft")
    sub.add_parser("validate", help="validate the committed corpus")
    sub.add_parser("docs", help="regenerate DATA_SOURCES.md from the registry")
    p = sub.add_parser(
        "contamination", help="count overlaps with an owner-supplied JSONL of requests"
    )
    p.add_argument("--file", required=True)
    p.add_argument("--field", default="request")
    p.add_argument("--version", default="v0.2-draft")
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
            path = OUT_ROOT / "v0.2-draft" / f"{split}.jsonl"
            corpus[split] = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        report = corpus_report(corpus)
        print(json.dumps({k: v for k, v in report.items() if k != "problems"}, indent=1))
        return 1 if report["problem_count"] else 0
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
