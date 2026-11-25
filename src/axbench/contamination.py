"""Rule-baseline contamination audit (ADR 0011).

The rule baseline and every benchmark template were written by the same AI tool, so the tool was
not blind to validation or test phrasings. As a mechanical safeguard, every keyword, phrase and
word list of the rule router must be motivated by *training* templates or be general vocabulary
that occurs in no template at all: an entry that occurs only in validation or test templates is a
contamination finding. Passing this audit does not make the baseline blind; it removes the
vocabulary that could only have come from held-out phrasings.
"""

from __future__ import annotations

from axiom_router import rules as R

from .lineages import LINEAGES


def _template_words(split_filter) -> list[list[str]]:
    return [
        R._norm_words(text)
        for lin in LINEAGES
        if split_filter(lin.split)
        for text in (lin.tr, lin.en, lin.mix or "")
        if text
    ]


def rule_vocabulary() -> dict[str, list[tuple[str, ...]]]:
    out: dict[str, list[tuple[str, ...]]] = {}
    for reason, cues in R.REJECT_CUES:
        out[f"reject:{reason}"] = list(cues)
    for op, cues in R.OP_CUES.items():
        out[f"op:{op}"] = list(cues)
    for name in ("_SEQUENCE_CUES", "_ALL_CUES", "_PEARSON", "_SPEARMAN", "_NOT_FOUND_CUES"):
        out[name] = list(getattr(R, name))
    out["_K_NOUNS"] = [(w,) for w in R._K_NOUNS]
    out["_STOP_WORDS"] = [(w,) for w in sorted(R._STOP_WORDS)]
    return out


def contamination_findings() -> list[str]:
    train = _template_words(lambda s: s == "train")
    held_out = _template_words(lambda s: s != "train")
    findings = []
    for name, cues in rule_vocabulary().items():
        for cue in cues:
            in_train = any(R._find(w, cue) for w in train)
            in_held_out = any(R._find(w, cue) for w in held_out)
            if in_held_out and not in_train:
                findings.append(f"{name}: {' '.join(cue)}")
    return findings
