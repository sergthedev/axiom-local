"""Template-based generation of AXBENCH-TR examples over real table schemas.

Gold labels are derived from the filled slots (never typed by hand) and every record is verified by
:mod:`axbench_tr.validate`. Output is deterministic for a given ``seed``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from axiom_router.contract import CANDIDATE_VERSION
from axiom_router.schema import SchemaView, deasciify, project_columns, turkish_lower

from .schemas import RealTable, table_partition, time_like, tokens, usable
from .templates import LINEAGES, Lineage, split_of

GENERATOR = "axbench_tr.synth/0.2"
CORPUS_VERSION = "axbench-tr/0.2-rev1"

FRAME_WEIGHTS: dict[str, float] = {
    "inspect": 8,
    "inspect_all": 3,
    "relate": 15,
    "compare": 14,
    "cluster": 10,
    **{
        f"clarify:{r}": 25 / 9
        for r in (
            "missing_operation",
            "multiple_operations",
            "missing_column",
            "column_not_found",
            "ambiguous_column",
            "column_type_mismatch",
            "missing_correlation_method",
            "missing_cluster_count",
            "invalid_parameter",
        )
    },
    **{
        f"reject:{r}": 20 / 9
        for r in (
            "instruction_override",
            "code_or_system_request",
            "data_modification",
            "causal_inference",
            "forecasting",
            "regression",
            "hypothesis_test",
            "unsupported_analysis",
            "out_of_scope",
        )
    },
}
# Sampling weights: tables of official/international statistical sources (World Bank, Eurostat,
# ECB, OECD, TCMB) are drawn most often, then economic municipal tables, then all other
# municipal tables.
OFFICIAL_SOURCES = frozenset({"SRC-O01", "SRC-O02", "SRC-O03", "SRC-O04", "SRC-O05"})
OFFICIAL_TABLE_WEIGHT = 5.0
ECONOMIC_TABLE_WEIGHT = 3.0
TR_SHARE = 0.80

ABSENT_TR = (
    "Reel Faiz Oranı",
    "Hane Halkı Tüketimi",
    "Vergi Gelirleri",
    "Dış Ticaret Açığı",
    "Kişi Başı Gelir",
    "Üretici Fiyat Endeksi",
    "Tasarruf Oranı",
    "Kamu Harcamaları",
    "Sanayi Üretim Endeksi",
    "Asgari Ücret Düzeyi",
    "Konut Fiyat Endeksi",
    "Borsa Getirisi",
)
ABSENT_EN = (
    "Producer Price Index",
    "Household Savings Rate",
    "Real Interest Rate",
    "Tax Revenue",
    "Trade Balance",
    "Industrial Production Index",
    "Minimum Wage Level",
    "Housing Price Index",
    "Consumer Confidence",
    "Stock Market Return",
)
K_WORDS_TR = {
    2: "iki",
    3: "üç",
    4: "dört",
    5: "beş",
    6: "altı",
    7: "yedi",
    8: "sekiz",
    9: "dokuz",
    10: "on",
}
K_WORDS_EN = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}
BAD_K = (1, 65, 100, 128, 200, 500)
GENERIC_TOKENS = frozenset(
    "sayi sayisi toplam oran orani yil ay gun total number rate year month day".split()
)


def normalize_request(text: str) -> str:
    """Duplicate key: case, diacritics, punctuation and spacing removed (same rule as AXBENCH)."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", deasciify(text)).split())


@dataclass(slots=True)
class _Ctx:
    table: RealTable
    view: SchemaView
    rng: random.Random
    lang: str
    tags: set[str]
    mentions: dict[str, list[str]]

    def idx_ok(self, kind: str | None) -> list[int]:
        bad = {int(cid[1:]) - 1 for grp in self.view.collisions for cid in grp}
        return [
            i
            for i, c in enumerate(self.table.columns)
            if c.slot_ok and i not in bad and (kind is None or c.kind == kind)
        ]


def _alias(name: str) -> str:
    head = re.split(r"\s*[(,]", name, maxsplit=1)[0].strip()
    if re.fullmatch(r"[A-Z0-9_ /.-]+", head):
        return head.replace("_", " ").lower()
    return turkish_lower(head)


def _swap_typo(text: str, rng: random.Random) -> str | None:
    words = [(m.start(), m.group()) for m in re.finditer(r"[^\W\d_]{6,}", text)]
    if not words:
        return None
    start, word = rng.choice(words)
    i = rng.randrange(1, len(word) - 2)
    return (
        text[:start] + word[:i] + word[i + 1] + word[i] + word[i + 2 :] + text[start + len(word) :]
    )


def _render_mention(ctx: _Ctx, index: int) -> str:
    name = ctx.table.columns[index].name
    rng = ctx.rng
    style = rng.choices(
        ["exact", "quoted", "alias", "case", "ascii", "typo"], [40, 20, 20, 8, 7, 5]
    )[0]
    others = [c.name for i, c in enumerate(ctx.table.columns) if i != index]

    def unique(text: str) -> bool:
        key = deasciify(text)
        return all(deasciify(o) != key and deasciify(_alias(o)) != key for o in others)

    out, tag = name, None
    if style == "quoted":
        out, tag = f'"{name}"', "quoted"
    elif style == "alias" and _alias(name) != name and unique(_alias(name)):
        out, tag = _alias(name), "alias"
    elif style == "case" and turkish_lower(name) != name and unique(turkish_lower(name)):
        out, tag = turkish_lower(name), "case"
    elif style == "ascii" and name != deasciify(name):
        out, tag = deasciify(name), "diacritics_stripped"
    elif style == "typo":
        typo = _swap_typo(_alias(name) if unique(_alias(name)) else name, rng)
        if typo and unique(typo):
            out, tag = typo, "typo"
    # A mention that is also a whole-word part of another column name could name either column;
    # such mentions are written as the exact quoted name instead.
    if not out.startswith('"') and any(_word_contained(out, o) for o in others):
        out, tag = f'"{name}"', "quoted"
    if tag:
        ctx.tags.add(tag)
    return out


def _word_contained(mention: str, name: str) -> bool:
    m = " ".join(tokens(mention))
    return bool(m) and f" {m} " in f" {' '.join(tokens(name))} " and " ".join(tokens(name)) != m


def _join(items: list[str], lang: str) -> str:
    if len(items) == 1:
        return items[0]
    conj = "ve" if lang == "tr" else "and"
    return ", ".join(items[:-1]) + f" {conj} {items[-1]}"


def _pick(ctx: _Ctx, kind: str | None, n: int) -> list[int] | None:
    pool = ctx.idx_ok(kind)
    return ctx.rng.sample(pool, n) if len(pool) >= n else None


def _mentions(ctx: _Ctx, slot: str, idxs: list[int]) -> str:
    rendered = [_render_mention(ctx, i) for i in idxs]
    ctx.mentions[slot] = rendered
    if len(idxs) > 1:
        ctx.tags.add("multi_mention")
    return _join(rendered, ctx.lang)


def _ids(idxs: list[int]) -> list[str]:
    return [f"c{i + 1}" for i in idxs]


def _ambiguous_token(ctx: _Ctx) -> tuple[str, list[int], int] | None:
    """A word shared by 2-8 column names (and not generic), plus one other numeric column."""
    cols = ctx.table.columns
    toks = [set(t for t in tokens(c.name) if len(t) >= 4 and t not in GENERIC_TOKENS) for c in cols]
    counts: dict[str, list[int]] = {}
    for i, ts in enumerate(toks):
        for t in ts:
            counts.setdefault(t, []).append(i)
    options = [(t, ix) for t, ix in sorted(counts.items()) if 2 <= len(ix) <= 8]
    ctx.rng.shuffle(options)
    numeric = ctx.idx_ok("numeric")
    for t, members in options:
        others = [i for i in numeric if i not in members]
        if others:
            return t, members, ctx.rng.choice(others)
    return None


def _absent(ctx: _Ctx) -> str | None:
    pool = list(ABSENT_TR if ctx.lang == "tr" else ABSENT_EN)
    ctx.rng.shuffle(pool)
    col_tokens = {t for c in ctx.table.columns for t in tokens(c.name) if len(t) >= 4}
    for cand in pool:
        if not ({t for t in tokens(cand) if len(t) >= 4} & col_tokens):
            return cand
    return None


_SLOT_RE = re.compile(r"\{(\w+)\}")


def _fill(lin: Lineage, ctx: _Ctx) -> tuple[dict[str, str], dict[str, Any]] | None:
    """Slot values for the template text and the gold candidate, or None if the table cannot host
    it."""
    frame, rng = lin.frame, ctx.rng
    text = lin.tr if ctx.lang == "tr" else lin.en
    needs = set(_SLOT_RE.findall(text))
    kind_, _, reason = frame.partition(":")
    vals: dict[str, str] = {}
    sel: dict[str, list[int]] = {}  # slot -> chosen column indexes
    used: set[int] = set()

    def choose(slot: str, kind: str | None, n: int = 1) -> bool:
        # A calendar column (year, month, ...) is a legitimate column to summarize, but not a
        # meaningful measure, correlate or cluster feature.
        pool = [
            i
            for i in ctx.idx_ok(kind)
            if i not in used and (slot == "cols" or not time_like(ctx.table.columns[i].name))
        ]
        if len(pool) < n:
            return False
        sel[slot] = rng.sample(pool, n)
        used.update(sel[slot])
        return True

    # --- columns -------------------------------------------------------------------------
    if reason == "ambiguous_column":
        amb = _ambiguous_token(ctx)
        if amb is None:
            return None
        token, members, other = amb
        sel["y"] = [other]
        used.update(sel["y"])
        vals["a"], ctx.mentions["a"] = token, [token]
        ambiguous_ids = _ids(sorted(members))
        used.update(members)
        if "g" in needs and not choose("g", "text"):
            return None
    else:
        ambiguous_ids = []
        if "cols" in needs and not choose("cols", None, rng.choices([1, 2, 3], [50, 35, 15])[0]):
            return None
        if "feats" in needs and not choose("feats", "numeric", rng.choice([2, 2, 3])):
            return None
        if reason == "column_type_mismatch":
            if "t" in needs and not (choose("t", "text") and choose("y", "numeric")):
                return None
            if "gn" in needs and not (choose("gn", "numeric") and choose("v", "numeric")):
                return None
        else:
            if "x" in needs and not choose("x", "numeric"):
                return None
            if "y" in needs and not choose("y", "numeric"):
                return None
            if "g" in needs and not choose("g", "text"):
                return None
            if "v" in needs and not choose("v", "numeric"):
                return None
    if frame == "relate" and not ({"x", "y"} <= sel.keys()):
        return None
    if frame == "compare" and not ({"g", "v"} <= sel.keys()):
        return None
    if frame == "cluster" and "feats" not in sel:
        return None
    if frame == "inspect" and "cols" not in sel:
        return None
    if reason == "column_not_found":
        absent = _absent(ctx)
        if absent is None:
            return None
        vals["n"], ctx.mentions["n"] = absent, [absent]
    for slot, idxs in sel.items():
        if slot in needs:
            vals[slot] = _mentions(ctx, slot, idxs)
    # --- parameters ----------------------------------------------------------------------
    k = 0
    if "k" in needs:
        k = rng.choice([2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12])
        words = K_WORDS_TR if ctx.lang == "tr" else K_WORDS_EN
        if k in words and rng.random() < 0.25 and "k={k}" not in text:
            vals["k"] = words[k]
            ctx.tags.add("k_word")
        else:
            vals["k"] = str(k)
    if "kbad" in needs:
        k = rng.choice(BAD_K)
        vals["kbad"] = str(k)
    method = lin.fixed.get("method") or rng.choice(["pearson", "spearman"])
    if "m" in needs:
        vals["m"] = method.capitalize()
    if needs - vals.keys():
        return None
    # --- gold ------------------------------------------------------------------------------
    head = {"version": CANDIDATE_VERSION}
    if frame == "inspect":
        gold: dict[str, Any] = {
            **head,
            "action": "PLAN",
            "operation": "inspect",
            "columns": _ids(sel["cols"]),
        }
    elif frame == "inspect_all":
        if (
            ctx.view.collisions
        ):  # "all columns" would include colliding names, which Core-side checks refuse
            return None
        gold = {
            **head,
            "action": "PLAN",
            "operation": "inspect",
            "columns": _ids(list(range(len(ctx.table.columns)))),
        }
    elif frame == "relate":
        gold = {
            **head,
            "action": "PLAN",
            "operation": "relate",
            "x": _ids(sel["x"])[0],
            "y": _ids(sel["y"])[0],
            "method": method,
        }
    elif frame == "compare":
        gold = {
            **head,
            "action": "PLAN",
            "operation": "compare",
            "group_by": _ids(sel["g"])[0],
            "measure": _ids(sel["v"])[0],
        }
    elif frame == "cluster":
        gold = {
            **head,
            "action": "PLAN",
            "operation": "cluster",
            "features": _ids(sel["feats"]),
            "k": k,
        }
    elif kind_ == "reject":
        gold = {**head, "action": "REJECT", "reason": reason}
    else:
        gold = {**head, "action": "CLARIFY", "reason": reason}
        if lin.hint:
            gold["operation"] = lin.hint
        if ambiguous_ids:
            gold["columns"] = ambiguous_ids
    return vals, gold


def _build_ctx(table: RealTable, lang: str, rng: random.Random) -> _Ctx:
    view = project_columns([(c.name, c.kind) for c in table.columns])
    return _Ctx(table, view, rng, lang, set(), {})


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    norm = sum(weights.values())
    raw = {k: total * w / norm for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def _interleave(counts: dict[str, int]) -> list[str]:
    """Smooth weighted round robin: exactly ``counts`` items, evenly spread."""
    total = sum(counts.values())
    current = dict.fromkeys(counts, 0)
    out = []
    for _ in range(total):
        for key, n in counts.items():
            current[key] += n
        best = max(sorted(current), key=lambda k: current[k])
        current[best] -= total
        out.append(best)
    return out


def generate(tables: list[RealTable], split: str, total: int, seed: str) -> list[dict[str, Any]]:
    """``total`` deduplicated examples for ``split`` (tables are filtered to the split's
    partition)."""
    rng = random.Random(f"{seed}/{split}")
    pool = [t for t in tables if table_partition(t) == split and usable(t)]
    if not pool:
        raise ValueError(f"no usable tables for split {split!r}")
    lineages: dict[str, list[Lineage]] = {}
    for lin in LINEAGES:
        if split_of(lin.id) == split:
            lineages.setdefault(lin.frame, []).append(lin)
    frames = [f for f in FRAME_WEIGHTS if f in lineages]
    tw = [
        OFFICIAL_TABLE_WEIGHT
        if t.source_id in OFFICIAL_SOURCES
        else ECONOMIC_TABLE_WEIGHT
        if t.domain == "economic"
        else 1.0
        for t in pool
    ]
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    n_tr = round(total * TR_SHARE)
    langs = ["tr"] * n_tr + ["en"] * (total - n_tr)
    rng.shuffle(langs)
    counts = _largest_remainder(total, {f: FRAME_WEIGHTS[f] for f in frames})
    sequence = _interleave(counts)
    for frame in sequence:
        lang = langs[len(out)]
        record = None
        for _ in range(200):
            lin = rng.choice(lineages[frame])
            table = rng.choices(pool, tw)[0]
            ctx = _build_ctx(table, lang, rng)
            filled = _fill(lin, ctx)
            if filled is None:
                continue
            vals, gold = filled
            request = (lin.tr if lang == "tr" else lin.en).format(**vals)
            key = (normalize_request(request), table.table_id)
            if key in seen:
                continue
            seen.add(key)
            if lang != table.header_language and ctx.mentions:
                ctx.tags.add("cross_language_mention")
            record = {
                "id": f"t2-{split}-{len(out) + 1:05d}",
                "corpus_version": CORPUS_VERSION,
                "split": split,
                "stratum": "template_on_real_schema",
                "language": lang,
                "family": table.group,
                "domain": table.domain,
                "schema_id": table.table_id,
                "lineage_id": lin.id,
                "frame": lin.frame,
                "request": request,
                "schema": ctx.view.to_prompt_payload(),
                "gold": gold,
                "accepted_variants": [],
                "tags": sorted(ctx.tags),
                "mentions": ctx.mentions,
                "provenance": {
                    "origin": "synthetic_template",
                    "request_authoring": "ai_generated_from_templates",
                    "schema_source_id": table.source_id,
                    "schema_origin": "real_table_headers_localized"
                    if table.localized
                    else "real_table_headers",
                    "header_language": table.header_language,
                    "human_review": "none",
                    "generator": GENERATOR,
                    "seed_label": f"{seed}/{split}",
                },
            }
            from .validate import record_problems  # lazy: validate imports this module

            if record_problems(record):
                seen.discard(key)  # e.g. a template word equals another column's name: retry
                record = None
                continue
            break
        if record is None:
            raise ValueError(
                f"could not generate an example for frame {frame!r} in split {split!r}"
            )
        out.append(record)
    return out
