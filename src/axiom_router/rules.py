"""Deterministic bilingual (Turkish/English) rule-based router, version ``axiom.rules/0.1``.

The mandatory baseline of DR-003/DR-005 and a model-free router. It receives exactly the router
input contract (request text and schema projection) and emits ``axiom.candidate/1`` JSON text,
which then goes through the same strict parser, semantic validation and Guard as any model output.
It never sees rows. It is not a fallback that is silently mixed into model results.

Optionally it takes a *lexicon* of synonym groups (surface forms that name the same concept, for
example ``hane geliri`` / ``household income``). The benchmark builds that lexicon from the
training partition only. Without a lexicon it matches column names (case, diacritics, separators,
small typos and Turkish suffixes tolerated).

Known, accepted limitations: keyword heuristics, no morphology beyond suffix tolerance, no
semantic understanding of paraphrases outside the keyword lists.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .contract import CANDIDATE_VERSION
from .prompt import RouterInput
from .schema import deasciify, loose_key, split_identifier

RULES_VERSION = "axiom.rules/0.1"
MAX_K = 64

_WORD = re.compile(r"[0-9a-z]+")


def _norm_words(text: str) -> list[str]:
    return _WORD.findall(deasciify(split_identifier(text)))


@dataclass(frozen=True, slots=True)
class Lexicon:
    """Synonym groups and generic (ambiguous) forms learned from a training partition."""

    groups: tuple[tuple[str, ...], ...] = ()
    generic: dict[str, tuple[int, ...]] = field(default_factory=dict)  # form -> group indices


def _keywords(*items: str) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(_norm_words(i)) for i in items)


REJECT_CUES: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    (
        "instruction_override",
        _keywords(
            "yok say",
            "ignore",
            "talimat",
            "instructions",
            "onay adımı",
            "confirmation step",
            "system note",
            "sistem notu",
            "mark this plan",
            "approved",
            "onaylanmış",
        ),
    ),
    (
        "code_or_system_request",
        _keywords(
            "python",
            "kodu",
            "code",
            "sql",
            "select",
            "upload",
            "http",
            "script",
        ),
    ),
    (
        "data_modification",
        _keywords(
            "sil",
            "delete",
            "doldur",
            "fill",
            "kaydet",
            "save",
            "drop",
            "replace",
        ),
    ),
    (
        "causal_inference",
        _keywords(
            "neden oluyor",
            "nedensel",
            "causal",
            "cause",
            "causing",
        ),
    ),
    (
        "regression",
        _keywords(
            "regresyon",
            "regress",
            "regression",
            "eğimi",
            "slope",
        ),
    ),
    (
        "forecasting",
        _keywords(
            "forecast",
            "gelecek",
            "next year",
            "önümüzdeki",
            "projeksiyon",
            "project",
            "tahmin et",
        ),
    ),
    (
        "hypothesis_test",
        _keywords(
            "anlamlı",
            "significant",
            "p değeri",
            "p değerini",
            "p value",
            "hipotez",
            "hypothesis",
        ),
    ),
    (
        "unsupported_analysis",
        _keywords(
            "matrisi",
            "matrix",
            "histogram",
            "grafik",
            "plot",
            "chart",
        ),
    ),
    (
        "out_of_scope",
        _keywords(
            "şiir",
            "poem",
            "hava nasıl",
            "weather",
            "film",
            "recipe",
        ),
    ),
)

OP_CUES: dict[str, tuple[tuple[str, ...], ...]] = {
    "relate": _keywords(
        "korelasyon",
        "korelasyonu",
        "korelasyonunu",
        "correlation",
        "correlate",
        "correlated",
        "ilişki",
        "ilişkili",
        "ilişkiyi",
        "ilişkisini",
        "relates",
        "pearson",
        "spearman",
        "katsayısı",
        "coefficient",
    ),
    "cluster": _keywords(
        "küme",
        "kümeye",
        "kümele",
        "kümeleme",
        "cluster",
        "clusters",
        "k means",
        "segment",
        "segments",
    ),
    "compare": _keywords(
        "karşılaştır",
        "karşılaştırma",
        "compare",
        "comparison",
        "bazında",
        "gruplarına göre",
        "grubu için",
        "for each",
        "across",
        "differ",
        "group",
    ),
    "inspect": _keywords(
        "özet",
        "özetle",
        "özetini",
        "özetini çıkar",
        "summarize",
        "summary",
        "betimsel",
        "descriptive",
        "istatistik",
        "istatistikleri",
        "statistics",
        "incele",
        "eksik",
        "missing",
        "dağılım",
        "dağılımı",
        "distribution",
        "describe",
    ),
}
_SEQUENCE_CUES = _keywords("ve ayrıca", "and also", "then", "ardından", "önce", "also", "da", "de")
_ALL_CUES = _keywords(
    "tüm sütunların",
    "tüm sütunları",
    "every column",
    "tüm column",
)
_PEARSON = _keywords("pearson", "doğrusal", "linear")
_SPEARMAN = _keywords(
    "spearman",
    "sıra korelasyonu",
    "rank",
)
_K_WORDS = {
    "iki": 2,
    "uc": 3,
    "dort": 4,
    "bes": 5,
    "alti": 6,
    "yedi": 7,
    "sekiz": 8,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}
_K_NOUNS = (
    "kume",
    "kumeye",
    "cluster",
    "clusters",
    "group",
    "groups",
    "segment",
    "segments",
)
_STOP_WORDS = frozenset(
    {
        "ile",
        "ve",
        "and",
        "by",
        "the",
        "into",
        "using",
        "groups",
        "group",
        "kumeye",
        "ayir",
        "bazinda",
        "karsilastir",
        "compare",
        "cluster",
        "with",
        "of",
        "for",
        "make",
        "create",
        "olustur",
        "kullanarak",
        "column",
        "show",
    }
)
_NOT_FOUND_CUES = _keywords("ile", "between", "and", "ve", "using", "kullanarak", "column")


def _find(words: list[str], phrase: Sequence[str]) -> list[int]:
    n = len(phrase)
    return [
        i
        for i in range(len(words) - n + 1)
        if all(_word_match(words[i + j], phrase[j], prefix=(j == n - 1)) for j in range(n))
    ]


def _word_match(word: str, form: str, *, prefix: bool) -> bool:
    if word == form:
        return True
    if prefix and len(form) >= 3 and word.startswith(form) and len(word) - len(form) <= 4:
        return True  # Turkish suffixes: korelasyonunu, gelirini
    return False


def _has(words: list[str], cues: Sequence[Sequence[str]]) -> bool:
    return any(_find(words, c) for c in cues)


def _osa1(a: str, b: str) -> bool:
    """Optimal string alignment distance <= 1 (one edit or one adjacent transposition)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return (
            len(diff) == 2
            and diff[1] == diff[0] + 1
            and a[diff[0]] == b[diff[1]]
            and a[diff[1]] == b[diff[0]]
        )
    if la > lb:
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1 :]


@dataclass(slots=True)
class _Mention:
    start: int
    end: int
    ids: tuple[str, ...]
    score: int


class RuleRouter:
    """``route(router_input) -> str``; deterministic for a given input and lexicon."""

    version = RULES_VERSION

    def __init__(self, lexicon: Lexicon | None = None) -> None:
        self.lexicon = lexicon or Lexicon()
        self._group_keys = [{loose_key(s) for s in group} for group in self.lexicon.groups]

    # ---------------------------------------------------------------- mentions

    def _forms(self, name: str) -> tuple[list[list[str]], set[int]]:
        forms = [_norm_words(name)]
        key = loose_key(name)
        groups = {i for i, keys in enumerate(self._group_keys) if key in keys}
        for i in sorted(groups):
            forms.extend(_norm_words(s) for s in self.lexicon.groups[i])
        return [f for f in forms if f], groups

    def _mentions(self, words: list[str], columns: list[dict]) -> list[_Mention]:
        found: list[_Mention] = []
        col_groups: dict[str, set[int]] = {}
        for col in columns:
            forms, groups = self._forms(col["name"])
            col_groups[col["id"]] = groups
            for form in forms:
                n = len(form)
                for i in range(len(words) - n + 1):
                    window = words[i : i + n]
                    exact = all(
                        _word_match(w, f, prefix=(j == n - 1))
                        for j, (w, f) in enumerate(zip(window, form, strict=True))
                    )
                    fuzzy = not exact and all(
                        w == f
                        or (len(f) >= 5 and _osa1(w, f))
                        or (j == n - 1 and _word_match(w, f, prefix=True))
                        for j, (w, f) in enumerate(zip(window, form, strict=True))
                    )
                    if exact or fuzzy:
                        found.append(_Mention(i, i + n, (col["id"],), 2 * n + (1 if exact else 0)))
        for form, group_ids in self.lexicon.generic.items():
            ids = tuple(c["id"] for c in columns if col_groups[c["id"]] & set(group_ids))
            fw = _norm_words(form)
            if ids:  # one id: a usable alias; two or more: an ambiguous mention
                for i in _find(words, fw):
                    found.append(_Mention(i, i + len(fw), ids, 1))
        # Keep the best non-overlapping spans (longest/strongest first, then leftmost).
        found.sort(key=lambda m: (-m.score, m.start))
        chosen: list[_Mention] = []
        for m in found:
            if all(m.end <= c.start or m.start >= c.end for c in chosen):
                same = [c for c in chosen if c.ids == m.ids]
                if not same:
                    chosen.append(m)
        chosen.sort(key=lambda m: m.start)
        return chosen

    # ---------------------------------------------------------------- routing

    def route(self, router_input: RouterInput) -> str:
        return json.dumps(self.decide(router_input), ensure_ascii=False, separators=(",", ":"))

    def decide(self, router_input: RouterInput) -> dict:
        words = _norm_words(router_input.request)
        columns = router_input.schema["columns"]
        kinds = {c["id"]: c["type"] for c in columns}

        def out(action: str, **fields) -> dict:
            return {"version": CANDIDATE_VERSION, "action": action, **fields}

        def clarify(reason: str, op: str | None = None, cols: Sequence[str] | None = None):
            fields: dict = {"reason": reason}
            if op:
                fields["operation"] = op
            if cols:
                fields["columns"] = sorted(cols, key=lambda c: int(c[1:]))[:8]
            return out("CLARIFY", **fields)

        for reason, cues in REJECT_CUES:
            if _has(words, cues):
                return out("REJECT", reason=reason)

        mentions = self._mentions(words, columns)
        ambiguous = [m for m in mentions if len(m.ids) > 1]
        single = [m.ids[0] for m in mentions if len(m.ids) == 1]
        ordered: list[str] = []
        for cid in single:
            if cid not in ordered:
                ordered.append(cid)
        nums = [c for c in ordered if kinds[c] == "numeric"]
        texts = [c for c in ordered if kinds[c] == "text"]

        ops = [op for op, cues in OP_CUES.items() if _has(words, cues)]
        k_value = self._k(words)
        if "cluster" in ops and "compare" in ops and not texts:
            ops.remove("compare")  # "into k groups" is a cluster count, not a comparison
        if "compare" in ops and "inspect" in ops:
            ops.remove("inspect")  # "summary of v for each g" is a group summary
        sequence = _has(words, _SEQUENCE_CUES)
        all_cols = _has(words, _ALL_CUES)
        if len(ops) >= 2 and (sequence or all_cols):
            return clarify("multiple_operations")
        for preferred in ("cluster", "relate", "compare", "inspect"):
            if preferred in ops:
                op = preferred
                break
        else:
            if ordered or ambiguous:
                return clarify("missing_operation")
            return out("REJECT", reason="out_of_scope")

        if ambiguous:
            return clarify("ambiguous_column", op, ambiguous[0].ids)

        if op == "inspect":
            if all_cols:
                return out("PLAN", operation="inspect", columns=[c["id"] for c in columns])
            if not ordered:
                return clarify("missing_column", op)
            return out("PLAN", operation="inspect", columns=ordered)

        if op == "relate":
            method = (
                "spearman"
                if _has(words, _SPEARMAN)
                else ("pearson" if _has(words, _PEARSON) else None)
            )
            if texts and len(nums) < 2:
                return clarify("column_type_mismatch", op)
            if len(nums) < 2:
                if nums and _has(words, _NOT_FOUND_CUES[:2]):
                    return clarify("column_not_found", op)
                return clarify("missing_column", op)
            if method is None:
                return clarify("missing_correlation_method", op)
            return out("PLAN", operation="relate", x=nums[0], y=nums[1], method=method)

        if op == "compare":
            if not texts:
                if len(nums) >= 2:
                    return clarify("column_type_mismatch", op)
                return clarify("missing_column", op)
            if not nums:
                if self._unmatched_content(words, mentions):
                    return clarify("column_not_found", op)
                return clarify("missing_column", op)
            return out("PLAN", operation="compare", group_by=texts[0], measure=nums[0])

        # cluster
        if texts:
            return clarify("column_type_mismatch", op)
        if not nums:
            return clarify("missing_column", op)
        if k_value is not None and not 2 <= k_value <= MAX_K:
            return clarify("invalid_parameter", op)
        if len(nums) == 1 and self._unmatched_content(words, mentions):
            return clarify("column_not_found", op)
        if k_value is None:
            return clarify("missing_cluster_count", op)
        return out("PLAN", operation="cluster", features=nums, k=k_value)

    @staticmethod
    def _unmatched_content(words: list[str], mentions: list[_Mention]) -> bool:
        covered = {i for m in mentions for i in range(m.start, m.end)}
        stop = {w for cue_list in OP_CUES.values() for cue in cue_list for w in cue}
        stop |= _STOP_WORDS
        rest = [
            w
            for i, w in enumerate(words)
            if i not in covered and w not in stop and not w.isdigit() and len(w) >= 4
        ]
        return bool(rest)

    @staticmethod
    def _k(words: list[str]) -> int | None:
        for i, w in enumerate(words):
            if w == "k" and i + 1 < len(words) and words[i + 1].isdigit():
                return int(words[i + 1][:6])
            m = re.fullmatch(r"k=?(\d+)", w)
            if m:
                return int(m.group(1)[:6])
        for i, w in enumerate(words):
            value = int(w[:6]) if w.isdigit() else _K_WORDS.get(w)
            if value is None:
                continue
            nxt = words[i + 1] if i + 1 < len(words) else ""
            prev = words[i - 1] if i > 0 else ""
            if any(nxt.startswith(n) for n in _K_NOUNS) or prev in ("k", "with"):
                return value
        return None
