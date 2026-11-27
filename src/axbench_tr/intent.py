"""Does a natural-language request say exactly what the derived plan does?

The router sees only a schema and a request. A plan derived from the SQL of an upstream row is
correct for the router only if the *whole request* is satisfied by that plan. So SQL shape is not
accepted as evidence of intent. These checks are deliberately narrow: a request is kept only if it
is a plain "<aggregate> <measure> [for each <group>]" question whose measure and group words match
the bound columns, and it carries no filter, ranking, count, total, proper noun, number, quoted or
SQL-looking text. Anything else is excluded with a reason, never relabelled.

Supported summaries are those of the Core descriptive kernels: min, max, mean (no sum, no count).
The Turkish side is a machine translation of the English prompt, so it is checked only for the
presence of the matching aggregate word and for absence of added constraints; it is not verified
for meaning.
"""

from __future__ import annotations

import re

AGG_WORDS = {
    "AVG": r"(?:average|mean|avg)",
    "MIN": r"(?:minimum|min|lowest|smallest)",
    "MAX": r"(?:maximum|max|highest|largest|greatest)",
}
TR_AGG_WORDS = {
    "AVG": r"ortalama",
    "MIN": r"minimum|en düşük|en küçük|asgari|en az",
    "MAX": r"maksimum|en yüksek|en büyük|azami|en fazla",
}
_LEAD = r"(?:what (?:is|are)|show|find|calculate|get|display|determine|report|list)(?: me)?"
_PREPOSITIONS = {
    "for", "in", "of", "at", "among", "within", "from", "by", "per", "across", "between",
    "during", "with", "where", "that", "who", "which", "when", "on", "to", "and", "or",
    "after", "before", "since", "than", "above", "below", "over", "under", "having", "has",
    "have", "had", "is", "are", "was", "were",
}  # fmt: skip
_BAD_TR = re.compile(r"toplam|kaç|sayısı|adet|sırala|yüzde|ilk \d|en popüler|en çok", re.I)
_UNITS = {"m", "km", "kg", "usd", "pct", "amt", "id", "num", "no", "cnt"}
_AGG_IN_NAME = {"max", "min", "avg", "mean", "average", "total", "maximum", "minimum"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _stem(word: str) -> str:
    return word.rstrip("s'")[:5]


def column_tokens(name: str) -> list[str]:
    """Content words of a column or table name (units and aggregate words dropped)."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    toks = [t.lower() for t in re.split(r"[^A-Za-z]+", spaced) if t]
    return [t for t in toks if len(t) > 2 and t not in _UNITS and t not in _AGG_IN_NAME]


def _covers(phrase_words: list[str], required: list[str]) -> bool:
    have = {_stem(w) for w in phrase_words}
    return bool(required) and all(_stem(t) in have for t in required)


def _extras_ok(phrase_words: list[str], required: list[str]) -> bool:
    exact = set(required)
    extra = [w for w in phrase_words if _stem(w) not in {_stem(t) for t in required}]
    # no extra word may be a participle ("priced", "yielding": asks for the entity, not the value)
    participle = any(w.endswith(("ed", "ing")) and w not in exact for w in phrase_words)
    return len(extra) <= 1 and not participle


def english_problem(prompt: str, basis: dict[str, object]) -> str | None:
    """Reason the English prompt is not fully expressed by ``basis`` (or None if it is)."""
    text = prompt.strip()
    if re.search(r"\d|['\"`_]|\btable\b|\bcolumn\b|\bdatabase\b", re.sub(r"'s\b", "", text)):
        return "digits_quotes_or_sql_words"
    body = text.rstrip("?.! ")
    # a capitalised word after the first is a proper noun (a named place, team, brand, ...)
    if any(w[:1].isupper() for w in body.split()[1:]):
        return "proper_noun_or_acronym"
    low = body.lower()
    agg = str(basis["agg"])
    if agg not in AGG_WORDS:  # SUM and COUNT are not Core descriptive kernels
        return "unsupported_aggregate"
    op = basis["op"]
    if op == "inspect":
        m = re.fullmatch(rf"{_LEAD} (?:the )?{AGG_WORDS[agg]} (?P<x>[a-z' ]+)", low)
        if not m:
            return "not_a_plain_aggregate_question"
        measure = _words(m["x"])
        required = column_tokens(str(basis["column"]))
        if _PREPOSITIONS & set(measure):
            return "extra_clause"
        if not _covers(measure, required):
            return "measure_words_do_not_match_column"
        return None if _extras_ok(measure, required) else "extra_or_changed_measure_words"
    m = re.fullmatch(
        rf"{_LEAD} (?:the )?{AGG_WORDS[agg]} (?P<x>[a-z' ]+?)"
        r" (?:for each|for every|in each|across each|per|by|of each) (?:the )?(?P<g>[a-z' ]+)",
        low,
    )
    if not m:
        return "not_a_plain_per_group_aggregate_question"
    measure, group = _words(m["x"]), _words(m["g"])
    if _PREPOSITIONS & set(measure) or _PREPOSITIONS & set(group):
        return "extra_clause"
    need_m = column_tokens(str(basis["measure"]))
    need_g = column_tokens(str(basis["group_by"]))
    if not _covers(measure, need_m):
        return "measure_words_do_not_match_column"
    if not _covers(group, need_g):
        return "group_words_do_not_match_column"
    if not (_extras_ok(measure, need_m) and _extras_ok(group, need_g)):
        return "extra_or_changed_words"
    return None


def turkish_problem(text: str, basis: dict[str, object]) -> str | None:
    """Reason the machine-translated prompt adds or loses what the plan does (or None)."""
    body = text.strip()
    if re.search(r"\d|['\"`_]", body.replace("'nın", "").replace("'nin", "")):
        return "digits_quotes_or_sql_words"
    if _BAD_TR.search(body):
        return "total_count_or_ranking_word"
    if any(w[:1].isupper() for w in body.split()[1:]):
        return "proper_noun_or_acronym"
    if str(basis["agg"]) not in TR_AGG_WORDS:
        return "unsupported_aggregate"
    if not re.search(TR_AGG_WORDS[str(basis["agg"])], body, re.I):
        return "aggregate_word_missing"
    if basis["op"] == "compare" and not re.search(
        r"\bher\b|başına|göre|bazında|bazlı|için", body, re.I
    ):
        return "group_marker_missing"
    return None


def modification_problem(prompt: str) -> str | None:
    """An INSERT/UPDATE/DELETE row is a data_modification request only if the prompt says so."""
    head = " ".join(prompt.lower().split()[:4])
    verbs = r"\b(insert|add|update|delete|remove|change|modify|set|increase|decrease)\b"
    if not re.search(verbs, head):
        return "prompt_does_not_ask_for_a_change"
    if prompt.strip().endswith("?"):
        return "prompt_is_a_question"
    return None
