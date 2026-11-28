"""EXP-TR04 data side: rights screening and automated label audit of the AXBENCH-TR v0.2-rev1 corpus.

``python -m axbench_tr.tr04 all|check|rights-check|export-mlx --out DIR``

Status: **automated and exploratory. Nothing here is human-reviewed or human-validated**, no model
is loaded and nothing is trained. The module reads the committed corpus ``data/axbench-tr/v0.2-rev1/``,
the committed licence evidence and the committed İzmir package screen
(``tr04_rights``), and writes ``data/axbench-tr/tr04/``. It never opens a test split (none exists
in this corpus; the sealed AXBENCH test is not referenced).

Two filters, applied in this order, decide which records the EXP-TR04 run may train on or be
validated on:

1. **Rights** (:func:`rights_decision`): a record is kept only if its request text is this
   project's own template text on a table layout whose terms were documented *and* screened here:
   İzmir open-data packages with the municipality licence, no free-text restriction found by the
   screen of ``tr04_rights``. Everything else is excluded as uncertain: World Bank layouts (upstream
   compilers' terms unread), Eurostat (dataset notes not found, exception clause is a legal
   reading), ECB/OECD/TCMB, external adapted rows, unverified translations.
2. **Labels** (:func:`row_problems`, :func:`tr04_cues.lineage_problems`): the template wording
   contract (every lineage, both languages) and a row-level re-derivation: the request must be
   exactly the lineage template filled with the recorded mentions, ``k`` and method; every mention
   must be a generator rendering of exactly the gold column it is bound to; gold structure, operation
   hints, language label, lineage, frame and split must agree. Any problem excludes the record.
   The audit's detection power is measured on synthetic mutations (:func:`mutation_report`).

The audit is an independent restatement of the generator's own rules by the same tool that wrote
the generator. It finds inconsistencies; it does not prove that a request is natural, that a label
is what a human would choose or that a table layout is representative.

Everything written is a pure function of the committed inputs (sorted keys, LF endings).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from axiom_router.schema import deasciify, turkish_lower

from . import tr04_cues, tr04_rights
from .build import CURRENT_VERSION, OUT_ROOT
from .fetch import sha256_bytes
from .synth import BAD_K, K_WORDS_EN, K_WORDS_TR, _alias, _join, normalize_request
from .templates import LINEAGES, Lineage, split_of
from .validate import record_problems, schema_fingerprint

TR04_DIR = OUT_ROOT / "tr04"
MAX_K = 64  # axiom.limits.Limits.max_k, the Core cap that makes a cluster count invalid
BATCH = 4  # gradient accumulation of the frozen configuration: kept rows are a multiple of it
TRIM_SEED = "axbench-tr-0.2-rev1-tr04-batch-trim"
MUTATION_SEED = "axbench-tr-0.2-rev1-tr04-mutations"
MUTATION_SAMPLE = 300
LIN: dict[str, Lineage] = {lin.id: lin for lin in LINEAGES}

# Source-level rights decisions (reason is the exclusion code). Anything not listed is uncertain.
SOURCE_DECISIONS: dict[str, str] = {
    "SRC-O06": "",  # included subject to the package screen (tr04_rights.izmir_status)
    "SRC-O01": "wdi_upstream_compilers_terms_unread",
    "SRC-O02": "eurostat_dataset_notes_not_found",
    "SRC-O03": "terms_unverified",
    "SRC-O04": "terms_unverified",
    "SRC-O05": "terms_unverified",
}


# ------------------------------------------------------------------------------------- loading


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def load(version: str = CURRENT_VERSION) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = OUT_ROOT / version
    return read_jsonl(root / "train.jsonl"), read_jsonl(root / "validation.jsonl")


# --------------------------------------------------------------------------------------- rights


def rights_decision(rec: Mapping[str, Any], izmir: Mapping[str, Mapping[str, Any]]) -> str:
    """Empty string if the record's rights are documented and screened, else the exclusion code."""
    prov = rec["provenance"]
    if prov["origin"] != "synthetic_template":
        return f"origin:{prov['origin']}"
    if "translation_unverified" in rec.get("tags", []):
        return "tag:translation_unverified"
    source = prov.get("schema_source_id")
    if source not in SOURCE_DECISIONS:
        return f"schema_source_unlisted:{source}"
    if SOURCE_DECISIONS[source]:
        return f"{source}:{SOURCE_DECISIONS[source]}"
    status = izmir.get(rec["family"])
    if status is None:
        return "izmir_group_not_screened"
    if not status["usable"]:
        return "izmir_package_not_usable:" + ",".join(status["problems"])
    return ""


# -------------------------------------------------------------------------------------- bindings


def _is_swap(a: str, b: str) -> bool:
    """``a`` is ``b`` with two adjacent characters exchanged (the generator's typo)."""
    if len(a) != len(b) or a == b:
        return False
    diff = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
    return (
        len(diff) == 2
        and diff[1] == diff[0] + 1
        and a[diff[0]] == b[diff[1]]
        and (a[diff[1]] == b[diff[0]])
    )


def is_rendering(mention: str, name: str) -> bool:
    """Is ``mention`` one of the ways the generator writes column ``name`` into a request?"""
    alias = _alias(name)
    exact = {name, f'"{name}"', alias, turkish_lower(name), deasciify(name)}
    if mention in exact:
        return True
    return any(_is_swap(mention, base) for base in (alias, name))


def resolve(mention: str, columns: Sequence[Mapping[str, Any]]) -> list[str]:
    """Ids of every column the mention could be a rendering of."""
    return [c["id"] for c in columns if is_rendering(mention, c["name"])]


# ----------------------------------------------------------------------------- request rebuild


def expected_pattern(
    rec: Mapping[str, Any], lin: Lineage, lang: str
) -> tuple[re.Pattern[str], list[str]]:
    """Regex of the request the generator would write; ``kbad`` is the only free slot."""
    text = lin.tr if lang == "tr" else lin.en
    gold, mentions, tags = rec["gold"], rec["mentions"], rec["tags"]
    parts = re.split(r"\{(\w+)\}", text)
    out, problems = [], []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            out.append(re.escape(part))
            continue
        if part in mentions:
            out.append(re.escape(_join(mentions[part], lang)))
        elif part == "k":
            k = gold.get("k")
            words = K_WORDS_TR if lang == "tr" else K_WORDS_EN
            if isinstance(k, int):
                out.append(re.escape(words.get(k, "?") if "k_word" in tags else str(k)))
            else:  # a clarify row records no count: the written count must be a valid one
                out.append(r"(?P<kfree>\d+|" + "|".join(map(re.escape, words.values())) + ")")
        elif part == "m":
            method = gold.get("method")
            out.append(re.escape(str(method).capitalize()) if method else "(?!)")
        elif part == "kbad":
            out.append(r"(?P<kbad>\d+)")
        else:
            problems.append(f"slot_without_value:{part}")
            out.append(".+?")
    return re.compile("^" + "".join(out) + "$", re.S), problems


def _kinds(rec: Mapping[str, Any]) -> dict[str, str]:
    return {c["id"]: c["type"] for c in rec["schema"]["columns"]}


# slot -> required column kind in a row; "a" and "n" are not columns
SLOT_KIND = {"x": "numeric", "y": "numeric", "v": "numeric", "feats": "numeric", "g": "text"}


def _bind_plan(rec: Mapping[str, Any], problems: list[str]) -> None:
    gold, mentions, cols = rec["gold"], rec["mentions"], rec["schema"]["columns"]
    op = gold["operation"]
    expected: dict[str, list[str]] = {
        "inspect": {"cols": list(gold.get("columns", []))},
        "relate": {"x": [gold.get("x")], "y": [gold.get("y")]},
        "compare": {"g": [gold.get("group_by")], "v": [gold.get("measure")]},
        "cluster": {"feats": list(gold.get("features", []))},
    }[op]
    if rec["frame"] == "inspect_all":
        if mentions:
            problems.append("inspect_all_with_mentions")
        if gold.get("columns") != [c["id"] for c in cols]:
            problems.append("inspect_all_columns_are_not_every_column")
        return
    if set(mentions) != set(expected):
        problems.append(f"mention_slots_differ_from_gold_slots:{sorted(mentions)}")
        return
    for slot, ids in expected.items():
        ms = mentions[slot]
        if len(ms) != len(ids):
            problems.append(f"mention_count_differs:{slot}")
            continue
        for mention, want in zip(ms, ids, strict=True):
            found = resolve(mention, cols)
            if found != [want]:
                problems.append(f"mention_not_bound_to_gold_column:{slot}:{found}!={want}")


def _bind_other(rec: Mapping[str, Any], lin: Lineage, problems: list[str]) -> None:
    mentions, cols, gold = rec["mentions"], rec["schema"]["columns"], rec["gold"]
    kinds = _kinds(rec)
    for slot, ms in mentions.items():
        if slot in ("a", "n"):
            continue
        for mention in ms:
            found = resolve(mention, cols)
            if len(found) != 1:
                problems.append(f"mention_resolves_to_{len(found)}_columns:{slot}")
                continue
            want = {"t": "text", "gn": "numeric"}.get(slot) or SLOT_KIND.get(slot)
            if want and kinds[found[0]] != want:
                problems.append(f"slot_{slot}_column_kind_is_{kinds[found[0]]}")
    reason = lin.frame.partition(":")[2]
    if reason == "column_not_found":
        absent = mentions.get("n", [""])[0]
        if resolve(absent, cols):
            problems.append("absent_column_resolves_to_a_column")
    if reason == "ambiguous_column":
        token = mentions.get("a", [""])[0]
        members = sorted(
            c["id"]
            for c in cols
            if token and token in re.split(r"[^0-9a-z]+", deasciify(c["name"]))
        )
        if len(members) < 2 or members != sorted(gold.get("columns", [])):
            problems.append("ambiguity_set_differs_from_token_members")
        if resolve(token, cols):
            problems.append("ambiguous_mention_is_also_an_exact_column")
    if reason == "column_type_mismatch" and not ({"t", "gn"} & set(mentions)):
        problems.append("type_mismatch_without_a_mismatching_slot")


# ------------------------------------------------------------------------------------ row audit


def row_problems(
    rec: Mapping[str, Any],
    lineages: Mapping[str, Lineage] | None = None,
    splits: Mapping[str, str] | None = None,
) -> list[str]:
    """Every inconsistency found in one record (empty list: none found by this audit).

    Fail closed: a record the audit cannot even read is a problem, never a pass.
    """
    try:
        return _row_problems(rec, lineages, splits)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, StopIteration) as exc:
        return [f"record_unreadable_by_the_audit:{type(exc).__name__}"]


def _row_problems(
    rec: Mapping[str, Any],
    lineages: Mapping[str, Lineage] | None,
    splits: Mapping[str, str] | None,
) -> list[str]:
    """Implementation of :func:`row_problems` (may raise on malformed records).

    ``lineages`` and ``splits`` default to the corpus templates and their pre-registered split;
    the held-out benchmark passes its own phrasings and the single split ``heldout``.
    """
    registry = LIN if lineages is None else lineages
    problems = list(record_problems(dict(rec)))  # gold validity and the corpus-build mention checks
    if rec.get("split") == "heldout":  # the held-out benchmark is not a corpus split
        problems = [p for p in problems if p not in ("bad_split", "bad_origin")]
    lin = registry.get(rec.get("lineage_id", ""))
    if lin is None:
        return [*problems, "unknown_lineage"]
    expected_split = split_of(lin.id) if splits is None else splits.get(lin.id)
    prov = rec["provenance"]
    if rec["frame"] != lin.frame:
        problems.append("frame_differs_from_lineage")
    if rec["split"] != expected_split:
        problems.append("split_differs_from_the_preregistered_lineage_split")
    if prov.get("request_authoring") != "ai_generated_from_templates":
        problems.append("request_authoring_is_not_template_generated")
    if prov.get("human_review") != "none":
        problems.append("human_review_claim")
    gold = rec["gold"]
    kind, _, reason = lin.frame.partition(":")
    if kind in ("inspect", "inspect_all", "relate", "compare", "cluster"):
        if gold["action"] != "PLAN" or gold.get("operation") != (
            "inspect" if kind == "inspect_all" else kind
        ):
            problems.append("gold_operation_differs_from_lineage")
    elif gold.get("action") != kind.upper() or gold.get("reason") != reason:
        problems.append("gold_action_or_reason_differs_from_lineage")
    if kind == "clarify" and gold.get("operation") != lin.hint:
        problems.append("clarify_operation_differs_from_lineage_hint")
    if kind == "relate" and lin.fixed.get("method") and gold.get("method") != lin.fixed["method"]:
        problems.append("method_differs_from_the_method_fixed_by_the_template")
    # the request must be the template filled with the recorded values, in the labelled language
    lang = rec["language"]
    pattern, more = expected_pattern(rec, lin, lang)
    problems += more
    match = pattern.match(rec["request"])
    if match is None:
        other = "en" if lang == "tr" else "tr"
        other_pat, _ = expected_pattern(rec, lin, other)
        problems.append(
            "language_label_is_for_the_other_language"
            if other_pat.match(rec["request"])
            else "request_is_not_the_lineage_template_filled_with_the_recorded_values"
        )
    else:
        found = match.groupdict()
        if found.get("kbad") is not None:
            bad_k = int(found["kbad"])
            if 2 <= bad_k <= MAX_K or bad_k not in BAD_K:
                problems.append("invalid_parameter_count_is_not_an_invalid_count")
        if found.get("kfree") is not None:
            raw = found["kfree"]
            words = {w: n for n, w in (K_WORDS_TR if lang == "tr" else K_WORDS_EN).items()}
            count = int(raw) if raw.isdigit() else words[raw]
            if not 2 <= count <= MAX_K:
                problems.append("count_in_request_is_not_a_valid_count")
    # bindings of mentions to columns
    if gold.get("action") == "PLAN":
        _bind_plan(rec, problems)
    else:
        _bind_other(rec, lin, problems)
    return sorted(set(problems))


# ------------------------------------------------------------------------------------ pipeline


def template_verdicts() -> dict[str, list[str]]:
    return {lin.id: tr04_cues.lineage_problems(lin) for lin in LINEAGES}


def _rank(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()


def audit_split(
    rows: Sequence[Mapping[str, Any]],
    izmir: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Sequence[str]],
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    """``(kept records, [(id, stage, reason)])``; one reason per excluded record, first filter wins."""
    kept: list[dict[str, Any]] = []
    excluded: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for rec in rows:
        why = rights_decision(rec, izmir)
        if why:
            excluded.append((rec["id"], "rights", why))
            continue
        if contract.get(rec["lineage_id"]):
            excluded.append((rec["id"], "label", f"template_contract:{rec['lineage_id']}"))
            continue
        problems = row_problems(rec)
        if problems:
            excluded.append((rec["id"], "label", problems[0]))
            continue
        key = (normalize_request(rec["request"]), schema_fingerprint(rec))
        if key in seen:
            excluded.append((rec["id"], "label", "duplicate_request_on_the_same_layout"))
            continue
        seen.add(key)
        kept.append(dict(rec))
    return kept, excluded


def trim_to_batch(
    kept: list[dict[str, Any]], batch: int = BATCH
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    """Drop the highest-ranked rows so that the kept count is a multiple of ``batch``."""
    extra = len(kept) % batch
    if not extra:
        return kept, []
    drop = {r["id"] for r in sorted(kept, key=lambda r: _rank(TRIM_SEED, r["id"]))[-extra:]}
    return (
        [r for r in kept if r["id"] not in drop],
        [(i, "trim", "partial_batch_trim") for i in sorted(drop)],
    )


def independence(
    train: Sequence[Mapping[str, Any]], valid: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """Cross-split overlaps among kept records (all must be zero)."""

    def keys(rows: Iterable[Mapping[str, Any]], f) -> set[str]:
        return {f(r) for r in rows}

    out = {}
    for name, f in (
        ("lineages", lambda r: r["lineage_id"]),
        ("table_families", lambda r: r["family"]),
        ("schema_fingerprints", schema_fingerprint),
        ("normalized_requests", lambda r: normalize_request(r["request"])),
    ):
        out[name] = len(keys(train, f) & keys(valid, f))
    return out


def _count(items: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(items).items()))


def _distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "language": _count(r["language"] for r in rows),
        "action": _count(r["gold"]["action"] for r in rows),
        "frame": _count(r["frame"] for r in rows),
        "domain": _count(r["domain"] for r in rows),
        "lineages": len({r["lineage_id"] for r in rows}),
        "table_families": len({r["family"] for r in rows}),
        "schema_fingerprints": len({schema_fingerprint(r) for r in rows}),
    }


# ------------------------------------------------------------------------------------ mutations


def _other_column(
    rec: Mapping[str, Any], avoid: set[str], kind: str | None, rank: str
) -> str | None:
    cols = [
        c
        for c in rec["schema"]["columns"]
        if c["id"] not in avoid and (kind is None or c["type"] == kind)
    ]
    cols.sort(key=lambda c: _rank(rank, c["id"]))
    return cols[0]["id"] if cols else None


def mutate(
    rec: Mapping[str, Any], kind: str, partner: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    """A copy of ``rec`` with one label or text field made wrong; ``None`` if not applicable."""
    m = json.loads(json.dumps(rec))
    g = m["gold"]
    cols = m["schema"]["columns"]
    if kind == "flip_method" and g.get("method"):
        g["method"] = "spearman" if g["method"] == "pearson" else "pearson"
    elif kind == "change_k" and isinstance(g.get("k"), int):
        g["k"] = g["k"] + 1 if g["k"] < MAX_K else g["k"] - 1
    elif kind == "swap_gold_column" and g.get("action") == "PLAN":
        key = {"inspect": "columns", "relate": "x", "compare": "measure", "cluster": "features"}[
            g["operation"]
        ]
        if m["frame"] == "inspect_all":
            return None
        cur = g[key][0] if isinstance(g[key], list) else g[key]
        kind_wanted = next(c["type"] for c in cols if c["id"] == cur)
        new = _other_column(
            m,
            {cur} | set(g.get(key, [])) if isinstance(g[key], list) else {cur},
            kind_wanted,
            MUTATION_SEED,
        )
        if new is None:
            return None
        if isinstance(g[key], list):
            g[key] = [new, *g[key][1:]]
        else:
            g[key] = new
    elif (
        kind == "exchange_group_measure"
        and g.get("action") == "PLAN"
        and g.get("operation") == "compare"
    ):
        g["group_by"], g["measure"] = g["measure"], g["group_by"]
    elif kind == "action_to_reject" and g.get("action") == "PLAN":
        m["gold"] = {"version": g["version"], "action": "REJECT", "reason": "out_of_scope"}
    elif kind == "plan_to_clarify" and g.get("action") == "PLAN":
        m["gold"] = {"version": g["version"], "action": "CLARIFY", "reason": "missing_operation"}
    elif kind == "reject_to_plan" and g.get("action") == "REJECT":
        m["gold"] = {
            "version": g["version"],
            "action": "PLAN",
            "operation": "inspect",
            "columns": [cols[0]["id"]],
        }
    elif kind == "reason_changed" and g.get("reason"):
        pool = sorted(
            {
                lin.frame.partition(":")[2]
                for lin in LINEAGES
                if lin.frame.startswith(m["frame"].split(":")[0])
            }
            - {g["reason"]}
        )
        g["reason"] = pool[int(_rank(MUTATION_SEED, m["id"]), 16) % len(pool)]
    elif kind == "language_label_swapped":
        m["language"] = "en" if m["language"] == "tr" else "tr"
    elif kind == "frame_changed":
        others = sorted({lin.frame for lin in LINEAGES} - {m["frame"]})
        m["frame"] = others[int(_rank(MUTATION_SEED, m["id"]), 16) % len(others)]
    elif kind == "request_from_other_row" and partner is not None:
        if normalize_request(partner["request"]) == normalize_request(m["request"]):
            return None
        m["request"] = partner["request"]
    elif kind == "mention_names_other_column" and g.get("action") == "PLAN" and m["mentions"]:
        slot = next((s for s in sorted(m["mentions"]) if s not in ("a", "n")), None)
        if slot is None:
            return None
        own = set(resolve(m["mentions"][slot][0], cols))  # never "replace" a column by itself
        names = [c["name"] for c in cols if c["id"] not in own]
        if not names:
            return None
        new = sorted(names, key=lambda n: _rank(MUTATION_SEED, n))[0]
        m["request"] = m["request"].replace(m["mentions"][slot][0], new, 1)
        m["mentions"][slot][0] = new
    elif (
        kind == "clarify_operation_changed"
        and m["frame"].startswith("clarify")
        and g.get("operation")
    ):
        g["operation"] = "inspect" if g["operation"] != "inspect" else "relate"
    elif (
        kind == "drop_gold_column"
        and g.get("operation") == "inspect"
        and len(g.get("columns", [])) > 1
    ):
        g["columns"] = g["columns"][:-1]
    else:
        return None
    return m


# "mention_names_other_column" is applied to PLAN rows only: for CLARIFY and REJECT rows the label
# does not depend on which real column the request names, so swapping it is not a fault.
MUTATION_KINDS = (
    "flip_method",
    "change_k",
    "swap_gold_column",
    "exchange_group_measure",
    "action_to_reject",
    "plan_to_clarify",
    "reject_to_plan",
    "reason_changed",
    "language_label_swapped",
    "frame_changed",
    "request_from_other_row",
    "mention_names_other_column",
    "clarify_operation_changed",
    "drop_gold_column",
)


def mutation_report(
    kept: Sequence[Mapping[str, Any]], sample: int = MUTATION_SAMPLE
) -> dict[str, Any]:
    """How many deliberately wrong copies of kept rows does the row audit flag? (synthetic faults)"""
    ordered = sorted(kept, key=lambda r: _rank(MUTATION_SEED, r["id"]))
    out: dict[str, Any] = {}
    for kind in MUTATION_KINDS:
        tried = caught = 0
        by_action: dict[str, list[int]] = {}
        missed: list[str] = []
        for i, rec in enumerate(ordered):
            if tried >= sample:
                break
            partner = ordered[(i + 1) % len(ordered)] if kind == "request_from_other_row" else None
            mutant = mutate(rec, kind, partner)
            if mutant is None:
                continue
            tried += 1
            flagged = bool(row_problems(mutant))
            caught += flagged
            cell = by_action.setdefault(rec["gold"]["action"], [0, 0])
            cell[0] += 1
            cell[1] += flagged
            if not flagged and len(missed) < 3:
                missed.append(rec["id"])
        out[kind] = {
            "applied": tried,
            "flagged": caught,
            "by_original_action_applied_flagged": dict(sorted(by_action.items())),
            "not_flagged_examples": missed,
        }
    return out


TEMPLATE_MUTATIONS = ("relabel_frame", "swap_languages", "drop_slot")


def template_mutation_report() -> dict[str, Any]:
    """Template-level fault injection: wrong frame label, a language swapped in, a slot removed."""
    frames = sorted({lin.frame for lin in LINEAGES})
    tried = Counter()
    caught = Counter()
    missed: dict[str, list[str]] = {k: [] for k in TEMPLATE_MUTATIONS}
    for lin in LINEAGES:
        others = [
            f for f in frames if f != lin.frame and f.split(":")[0] != lin.frame.split(":")[0]
        ]
        wrong = others[int(_rank(MUTATION_SEED, lin.id), 16) % len(others)]
        variants = {
            "relabel_frame": Lineage(lin.id, wrong, lin.tr, lin.en, None, {}),
            "swap_languages": Lineage(lin.id, lin.frame, lin.en, lin.tr, lin.hint, lin.fixed),
        }
        slot = next(iter(sorted(tr04_cues.slots(lin.tr))), None)
        if slot:
            variants["drop_slot"] = Lineage(
                lin.id, lin.frame, lin.tr.replace("{" + slot + "}", ""), lin.en, lin.hint, lin.fixed
            )
        for name, mutant in variants.items():
            if name == "swap_languages":
                continue  # a language swap is a row-level fault, covered by the row mutations
            tried[name] += 1
            if tr04_cues.lineage_problems(mutant):
                caught[name] += 1
            elif len(missed[name]) < 5:
                missed[name].append(f"{lin.id}->{mutant.frame}")
    return {
        k: {"applied": tried[k], "flagged": caught[k], "not_flagged_examples": missed[k]}
        for k in TEMPLATE_MUTATIONS
        if tried[k]
    }


# ---------------------------------------------------------------------------------------- output


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _csv_bytes(header: Sequence[str], rows: Iterable[Sequence[str]]) -> bytes:
    buf = io.StringIO(newline="")
    import csv

    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def build(version: str = CURRENT_VERSION) -> dict[str, bytes]:
    """Every generated file of ``data/axbench-tr/tr04/`` (relative path -> bytes), deterministically."""
    train, valid = load(version)
    izmir_doc = tr04_rights.load_izmir()
    izmir = tr04_rights.izmir_status(izmir_doc)
    contract = template_verdicts()
    kept_train, ex_train = audit_split(train, izmir, contract)
    kept_valid, ex_valid = audit_split(valid, izmir, contract)
    kept_train, trim_train = trim_to_batch(kept_train)
    kept_valid, trim_valid = trim_to_batch(kept_valid)
    excluded = [("train", *e) for e in ex_train + trim_train] + [
        ("validation", *e) for e in ex_valid + trim_valid
    ]
    reasons = Counter(
        (split, stage, why.split(":")[0] if stage == "rights" else why)
        for split, _, stage, why in excluded
    )
    audit = {
        "schema": "axbench-tr.tr04-audit/1",
        "status": "automated and exploratory; not human-reviewed; not human-validated; no model involved",
        "corpus": f"axbench-tr/{version}",
        "input_records": {"train": len(train), "validation": len(valid)},
        "kept_records": {"train": len(kept_train), "validation": len(kept_valid)},
        "excluded_by_stage": {
            stage: {
                s: sum(1 for sp, _, st, _ in excluded if sp == s and st == stage)
                for s in ("train", "validation")
            }
            for stage in ("rights", "label", "trim")
        },
        "excluded_by_reason": {
            f"{s}|{stage}|{why}": n for (s, stage, why), n in sorted(reasons.items())
        },
        "template_contract": {
            "lineages": len(contract),
            "lineages_with_problems": sorted(k for k, v in contract.items() if v),
        },
        "rights": {
            "izmir_packages": len(izmir_doc["packages"]),
            "izmir_packages_flagged_by_the_screen": sum(
                bool(p["flags"]) for p in izmir_doc["packages"].values()
            ),
            "wdi_indicators": tr04_rights.wdi_evidence(),
            "source_decisions": {
                k: v or "included_if_package_screen_clean" for k, v in SOURCE_DECISIONS.items()
            },
            "rule": "kept only if origin synthetic_template, schema source SRC-O06 and the package screen is clean",
        },
        "kept_distribution": {
            "train": _distribution(kept_train),
            "validation": _distribution(kept_valid),
        },
        "independence_of_kept_splits": independence(kept_train, kept_valid),
        "limitations": [
            "the audit restates the generator's rules; it finds inconsistencies, it does not show that a request is natural or a label is what a human would choose",
            "the template cue lists were written by the same AI tool as the templates (correlated blind spots)",
            "the İzmir package screen is a pattern match over text fields, not a legal opinion; no hit does not mean no restriction",
            "no kept record is human-reviewed; provenance.human_review stays 'none'",
            "dropping World Bank and Eurostat layouts removes all economic layouts from training and validation",
        ],
    }
    mutation = {
        "schema": "axbench-tr.tr04-mutation-report/1",
        "meaning": "detection of deliberately injected faults on synthetic copies; not accuracy on real label errors",
        "row_mutations_on_kept_train": mutation_report(kept_train),
        "template_mutations": template_mutation_report(),
    }
    files: dict[str, bytes] = {
        "AUDIT.json": _json_bytes(audit),
        "MUTATION_REPORT.json": _json_bytes(mutation),
        "TEMPLATE_CONTRACT.json": _json_bytes(
            {
                "cue_patterns": {
                    "family": tr04_cues.FAMILY_CUES,
                    "reject": tr04_cues.REJECT_CUES,
                    "method": tr04_cues.METHOD_CUES,
                },
                "verdicts": contract,
            }
        ),
        "EXCLUSIONS.csv": _csv_bytes(("split", "id", "stage", "reason"), excluded),
        "KEEP_TRAIN.txt": ("\n".join(r["id"] for r in kept_train) + "\n").encode(),
        "KEEP_VALIDATION.txt": ("\n".join(r["id"] for r in kept_valid) + "\n").encode(),
    }
    inputs = {
        f"{version}/{name}": sha256_bytes((OUT_ROOT / version / name).read_bytes())
        for name in ("train.jsonl", "validation.jsonl", "token_lengths_per_example.jsonl")
    }
    inputs["tr04/rights/IZMIR_PACKAGE_NOTES.json"] = sha256_bytes(
        tr04_rights.IZMIR_NOTES_PATH.read_bytes()
    )
    files["MANIFEST.json"] = _json_bytes(
        {
            "schema": "axbench-tr.tr04-manifest/1",
            "inputs_sha256": inputs,
            "files_sha256": {k: sha256_bytes(v) for k, v in sorted(files.items())},
        }
    )
    return files


def write(out: Path = TR04_DIR) -> dict[str, str]:
    files = build()
    out.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (out / name).write_bytes(data)
    return {k: sha256_bytes(v) for k, v in files.items()}


def check(out: Path = TR04_DIR) -> list[str]:
    problems = []
    for name, data in build().items():
        path = out / name
        if not path.is_file():
            problems.append(f"missing:{name}")
        elif path.read_bytes() != data:
            problems.append(f"differs_from_rebuild:{name}")
    return problems


# --------------------------------------------------------------------------------- training export


def kept_ids(out: Path = TR04_DIR) -> dict[str, list[str]]:
    return {
        "train": (out / "KEEP_TRAIN.txt").read_text("utf-8").split(),
        "validation": (out / "KEEP_VALIDATION.txt").read_text("utf-8").split(),
    }


def export_mlx(out: Path, version: str = CURRENT_VERSION) -> dict[str, Any]:
    """MLX chat export of the kept rows (``tr04/train.jsonl``, ``tr04/valid.jsonl``, ROW_IDS, manifest).

    Written outside the repository; hashes are reproducible and are pinned in ``training/tr04/config.json``.
    """
    from .export import _write

    out = out.resolve()
    if OUT_ROOT.parents[1] in (out, *out.parents):
        raise SystemExit("refusing to export inside the repository; choose another directory")
    train, valid = load(version)
    ids = kept_ids()
    by_id = {r["id"]: r for r in train + valid}
    sets = {
        "train": [by_id[i] for i in ids["train"]],
        "valid": [by_id[i] for i in ids["validation"]],
    }
    files: dict[str, Any] = {}
    row_ids: dict[str, list[str]] = {}
    for name, recs in sets.items():
        files[f"tr04/{name}.jsonl"] = _write(out / "tr04" / f"{name}.jsonl", recs)
        row_ids[f"tr04/{name}.jsonl"] = [r["id"] for r in recs]
    ids_bytes = (json.dumps(row_ids, indent=0, sort_keys=True) + "\n").encode("utf-8")
    (out / "ROW_IDS.json").write_bytes(ids_bytes)
    files["ROW_IDS.json"] = {"bytes": len(ids_bytes), "sha256": sha256_bytes(ids_bytes)}
    manifest = {
        "schema": "axbench-tr.tr04-mlx-export/1",
        "corpus_version": f"axbench-tr/{version.removeprefix('v')}",
        "format": "mlx-lm chat: system, user (axiom.prompt/1 JSON), assistant (compact gold JSON)",
        "files": files,
        "test_file": "none: this corpus has no test split and none is exported",
        "rights": "kept rows only (tr04 rights screen and label audit); automated, not human-reviewed",
    }
    (out / "EXPORT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("all")
    sub.add_parser("check")
    sub.add_parser("rights-check")
    exp = sub.add_parser("export-mlx")
    exp.add_argument("--out", required=True)
    a = parser.parse_args(argv)
    if a.cmd == "all":
        print(json.dumps(write(), indent=1))
        return 0
    if a.cmd == "check":
        problems = check()
        print(json.dumps({"problems": problems}, indent=1))
        return 1 if problems else 0
    if a.cmd == "rights-check":
        problems = tr04_rights.check_evidence(tr04_rights.load_izmir())
        print(json.dumps({"problems": problems}, indent=1))
        return 1 if problems else 0
    manifest = export_mlx(Path(a.out))
    print(json.dumps({k: v["sha256"] for k, v in manifest["files"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
