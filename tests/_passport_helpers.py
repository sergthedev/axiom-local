"""Helpers for verification tests: example inputs and re-sealing of edited passports.

``reseal`` recomputes every digest a passport states after a test edits it. It models an adversary
who edits a passport *consistently*: such a passport passes its own integrity checks, so only
recomputation from the data (replay) can expose it. Digests are not signatures.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from axiom.canonical import canonical_sha256
from axiom.passport import passport_digest, profile_from_dict
from axiom.plan import parse_plan_object, plan_sha256, plan_to_dict

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"
DATA = EXAMPLES / "synthetic_regions.csv"
PROFILE = profile_from_dict(
    json.loads((EXAMPLES / "profiles" / "synthetic_regions.json").read_text("utf-8"))
)


def example_plan(name: str):
    return parse_plan_object(json.loads((EXAMPLES / "plans" / f"{name}.json").read_text("utf-8")))


def reseal(passport: dict[str, Any], *, plan_too: bool = False) -> dict[str, Any]:
    """Return a copy whose result, plan, identity and passport digests are all recomputed."""
    p = copy.deepcopy(passport)
    if plan_too:
        plan = parse_plan_object(p["inputs"]["plan"])
        p["inputs"]["plan"] = plan_to_dict(plan)
        p["result"]["plan"] = plan_to_dict(plan)
        digest = plan_sha256(plan)
        p["identity"]["plan_sha256"] = digest
        p["result"]["inputs"]["plan_sha256"] = digest
    p["identity"]["result_sha256"] = canonical_sha256(p["result"])
    p["passport_sha256"] = passport_digest(p)
    return p


def edited(passport: dict[str, Any], edit) -> dict[str, Any]:
    p = copy.deepcopy(passport)
    edit(p)
    return p
