"""Explicit, pinned provisioning of an AX-01 candidate checkpoint (ADR 0012).

Run only when the owner has approved the download (sizes below), from this directory:

    uv run --locked python provision.py --model qwen3-0.6b --out <directory outside the repository>
    uv run --locked python provision.py --model qwen3-0.6b --out <dir> --tokenizer-only

* Downloads exactly the listed files at the pinned revision (no other files, no "latest").
* Verifies the SHA-256 of every file whose hash the Hub publishes (the LFS files) and records the
  SHA-256 of all other files in ``PROVENANCE.json`` (trust on first use for those).
* Refuses a directory inside the repository and an existing non-empty directory.
* Never runs at import time and is never called by AXIOM, AXBENCH or the tests; inference later
  reads the directory with ``HF_HUB_OFFLINE=1`` and never downloads on failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TOKENIZER_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
)

# Revisions, file sizes and LFS SHA-256 from the Hub API on 2026-11-26 (REF-032, REF-033).
CHECKPOINTS = {
    "qwen3-0.6b": {
        "repo_id": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "weights": {
            "model.safetensors": (
                1503300328,
                "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
            ),
        },
    },
    "qwen3-1.7b": {
        "repo_id": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "weights": {
            "model-00001-of-00002.safetensors": (
                3441185608,
                "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
            ),
            "model-00002-of-00002.safetensors": (
                622329984,
                "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
            ),
            "model.safetensors.index.json": (25605, None),
        },
    },
}
KNOWN_SHA256 = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
}
CONFIG_SHA256 = {
    "qwen3-0.6b": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "qwen3-1.7b": "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provision(model: str, out: Path, tokenizer_only: bool) -> dict:
    from huggingface_hub import hf_hub_download

    spec = CHECKPOINTS[model]
    out = out.resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        raise SystemExit("refusing to place model files inside the repository")
    if out.exists() and any(out.iterdir()):
        raise SystemExit("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    files = list(TOKENIZER_FILES)
    if not tokenizer_only:
        files += list(spec["weights"])
    expected = {**KNOWN_SHA256, "config.json": CONFIG_SHA256[model]}
    expected |= {name: sha for name, (_, sha) in spec["weights"].items() if sha}
    records = {}
    for name in files:
        path = Path(
            hf_hub_download(spec["repo_id"], name, revision=spec["revision"], local_dir=out)
        )
        digest = _sha256(path)
        want = expected.get(name)
        if want is not None and digest != want:
            raise SystemExit(f"SHA-256 mismatch for {name}: refusing to use this checkpoint")
        size = path.stat().st_size
        if name in spec["weights"] and size != spec["weights"][name][0]:
            raise SystemExit(f"size mismatch for {name}")
        records[name] = {
            "sha256": digest,
            "bytes": size,
            "verified_against": "published hash" if want else "recorded on first use",
        }
    provenance = {
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "tokenizer_only": tokenizer_only,
        "files": records,
        "licence": "Apache-2.0 (LICENSE file included)",
    }
    (out / "PROVENANCE.json").write_bytes(
        (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=sorted(CHECKPOINTS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer-only", action="store_true")
    args = parser.parse_args()
    result = provision(args.model, Path(args.out), args.tokenizer_only)
    json.dump({k: v for k, v in result.items() if k != "files"}, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
