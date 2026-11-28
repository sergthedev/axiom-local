"""EXP-TR04 template contract: does the wording of a request template say what its label says?

The 122 lineages of ``axbench_tr.templates`` are the only source of request wording in the
rights-screened corpus, so the semantic audit is exhaustive at the template level: every lineage,
in Turkish and in English, is checked against an explicit, reviewable contract of cue patterns.
Rows are then checked against their template (``tr04.row_problems``).

The cue lists below were written by the implementing AI tool for this audit. They are an
independent restatement of the labelling conventions in the AXBENCH-TR dataset card, **not** a
human review and not independent of the author of the templates; a template that passes carries
"no contradiction found by this contract", never "verified correct". Their detection power is
measured on synthetic mutations (``tr04.mutation_report``), which is also not a measure of
accuracy on real mistakes.

Matching is on Turkish-aware lower-cased text with placeholders replaced by ``_``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from axiom_router.schema import turkish_lower

# Operation families that a PLAN request may express. A PLAN template must express exactly its
# own family; a CLARIFY ``multiple_operations`` template must express at least two.
FAMILY_CUES: dict[str, str] = {
    "inspect": (
        r"summar|descriptive|statistic|distribut|missing|quartile|overview|data quality"
        r"|\bmean\b|\bminimum\b|\bmaximum\b"
        r"|özet|betimsel|istatistik|dağıl|eksik|çeyrek|veri kalitesi|genel bakış"
        r"|ortalama|en küçük|en büyük|minimum|maksimum"
    ),
    "relate": (
        r"correlat|relationship|related|move together|monoton|coefficient"
        r"|korelasyon|ilişki|birlikte hareket|katsayı"
    ),
    "compare": (
        r"compar|\bgroups?\b|across|by group|broken down|\bper\b|group by group"
        r"|karşılaştır|kıyasla|grup|bazında|bazlı|başına|kırılım|sınıf"
    ),
    "cluster": (
        r"cluster|segment|k-means|homogeneous|homojen|kümele|küme|kümeli|\bkümeye|\bkümelik"
    ),
}
# Cues of each REJECT reason; the template must contain its own reason's cue.
REJECT_CUES: dict[str, str] = {
    "causal_inference": r"caus|lead to|reason for|nedensel|neden-sonuç|sebe[pb]|yol aç",
    "regression": r"regress|regresyon|r-squared|r-kare|equation|denklem",
    "forecasting": (
        r"forecast|projection|ahead|coming period|next \d|year end|evolve"
        r"|tahmin|projeksiyon|gelecek|önümüzdeki|yıl sonunda|ileriye|seyreder"
    ),
    "hypothesis_test": (
        r"t-test|anova|significan|null hypothesis|confidence interval"
        r"|anlamlı|güven aralığı|hipotez|sına"
    ),
    "unsupported_analysis": (
        r"boxplot|kutu grafi|outlier|aykırı|olağandışı|unusual|heat map|ısı haritası"
        r"|scatter|dağılım grafiği|pca|principal component|temel bileşen|matrix|matris"
        r"|histogram|bar chart|çubuk grafik|pasta grafi"
    ),
    "data_modification": (
        r"fill|log-transform|add it to the table|remove|delete|rename|change the header|clean"
        r"|\breplace|\bsave\b|\bdrop\b|overwrite"
        r"|doldur|dönüştür|tabloya ekle|temizle|\bsil\b|yeniden adlandır|kaydet|kaldır|değiştirip"
    ),
    "code_or_system_request": (
        r"\bscript|query|select |terminal|/etc|function|komut|betik|sorgu|fonksiyon"
        r"|dosyasını aç|pandas|\bcommand|\bshell\b|\bbash\b|kabuk"
    ),
    "instruction_override": (
        r"forget|disable|admin|system|skip|from now on|unut|yönetici|devre dışı"
        r"|sistem|atla|bundan sonra|\bignore|görmezden|bypass"
    ),
}
# Method cues: a relate template must carry one iff its gold method is fixed by the template.
METHOD_CUES: dict[str, str] = {
    "pearson": r"pearson|linear|doğrusal|çizgisel",
    "spearman": r"spearman|rank|sıra|monoton",
}
ANY_METHOD_CUE = r"pearson|spearman|linear|rank|doğrusal|çizgisel|sıra|monoton"
K_WORD_CUE = {
    "en": r"\b(two|three|four|five|six|seven|eight|nine|ten)\b",
    "tr": r"\b(iki|üç|dört|beş|altı|yedi|sekiz|dokuz|on)\b",
}
# Cues a PLAN family may legitimately share with another family in the same sentence: a group
# comparison is described as a summary by group, and a cluster request says "groups".
ALLOWED_OVERLAP: dict[str, frozenset[str]] = {
    "compare": frozenset({"inspect"}),
    "cluster": frozenset({"compare"}),
}
_SLOT = re.compile(r"\{(\w+)\}")

_FAMILY_RE = {k: re.compile(v) for k, v in FAMILY_CUES.items()}
_REJECT_RE = {k: re.compile(v) for k, v in REJECT_CUES.items()}
_METHOD_RE = {k: re.compile(v) for k, v in METHOD_CUES.items()}
_ANY_METHOD_RE = re.compile(ANY_METHOD_CUE)


def slots(text: str) -> set[str]:
    return set(_SLOT.findall(text))


def _plain(text: str) -> str:
    return turkish_lower(_SLOT.sub("_", text))


def families(text: str) -> set[str]:
    plain = _plain(text)
    return {k for k, rx in _FAMILY_RE.items() if rx.search(plain)}


def reject_reasons(text: str) -> set[str]:
    plain = _plain(text)
    return {k for k, rx in _REJECT_RE.items() if rx.search(plain)}


def template_problems(
    frame: str, hint: str | None, fixed: Mapping[str, str], text: str, lang: str
) -> list[str]:
    """Contradictions between one template text and its label (empty: none found)."""
    problems: list[str] = []
    s = slots(text)
    plain = _plain(text)
    fam = families(text)
    kind, _, reason = frame.partition(":")
    has_cols = bool(s & {"x", "y", "g", "v", "cols", "feats", "t", "gn", "a", "n"})
    if frame in ("inspect", "inspect_all", "relate", "compare", "cluster"):
        want = "inspect" if frame == "inspect_all" else frame
        if want not in fam:
            problems.append(f"own_family_cue_missing:{want}")
        for other in sorted(fam - {want} - ALLOWED_OVERLAP.get(want, frozenset())):
            problems.append(f"other_family_cue:{other}")
        bad = reject_reasons(text)
        problems += [f"reject_cue_in_plan:{r}" for r in sorted(bad)]
        if s & {"n", "a", "t", "gn", "kbad"}:
            problems.append("clarify_slot_in_plan_template")
        if frame == "inspect_all":
            if not re.search(r"\ball\b|\bevery\b|\beach\b|tüm|bütün|\bher\b", plain):
                problems.append("inspect_all_without_an_all_cue")
            if has_cols:
                problems.append("inspect_all_with_a_column_slot")
        elif frame == "inspect" and "cols" not in s:
            problems.append("inspect_without_cols_slot")
        elif frame == "relate":
            if not {"x", "y"} <= s:
                problems.append("relate_without_x_y")
            fixed_method = fixed.get("method")
            if fixed_method:
                if not _METHOD_RE[fixed_method].search(plain):
                    problems.append(f"fixed_method_not_in_text:{fixed_method}")
                other = "spearman" if fixed_method == "pearson" else "pearson"
                if _METHOD_RE[other].search(plain):
                    problems.append(f"contradicting_method_cue:{other}")
            elif "m" not in s:
                problems.append("method_neither_fixed_nor_in_text")
        elif frame == "compare" and not {"g", "v"} <= s:
            problems.append("compare_without_g_v")
        elif frame == "cluster":
            if "feats" not in s or "k" not in s:
                problems.append("cluster_without_feats_or_k")
        if frame != "relate" and "m" in s:
            problems.append("method_slot_outside_relate")
        return problems
    if kind == "reject":
        if reason not in _REJECT_RE:
            if reason != "out_of_scope":
                problems.append(f"unknown_reject_reason:{reason}")
            elif fam or reject_reasons(text) or has_cols:
                problems.append("out_of_scope_with_data_cues")
        elif reason not in reject_reasons(text):
            problems.append(f"own_reject_cue_missing:{reason}")
        if s & {"n", "a", "t", "gn", "kbad", "m"}:
            problems.append("clarify_slot_in_reject_template")
        return problems
    # clarify
    if reason == "missing_operation":
        if fam or reject_reasons(text):
            problems.append(f"operation_cue_in_missing_operation:{sorted(fam)}")
        if not has_cols:
            problems.append("missing_operation_without_a_column")
    elif reason == "multiple_operations":
        if len(fam) < 2:
            problems.append("multiple_operations_with_fewer_than_two_families")
    elif reason == "missing_column":
        if has_cols:
            problems.append("missing_column_with_a_column_slot")
        if not fam:
            problems.append("missing_column_without_an_operation_cue")
    elif reason == "column_not_found":
        if "n" not in s:
            problems.append("column_not_found_without_n_slot")
    elif reason == "ambiguous_column":
        if "a" not in s:
            problems.append("ambiguous_column_without_a_slot")
    elif reason == "column_type_mismatch":
        if not s & {"t", "gn"}:
            problems.append("type_mismatch_without_t_or_gn_slot")
    elif reason == "missing_correlation_method":
        if "relate" not in fam or not {"x", "y"} <= s:
            problems.append("missing_method_without_relate_cue_or_x_y")
        if "m" in s or _ANY_METHOD_RE.search(plain):
            problems.append("method_given_in_missing_method_template")
    elif reason == "missing_cluster_count":
        if "cluster" not in fam or "feats" not in s:
            problems.append("missing_count_without_cluster_cue_or_feats")
        if s & {"k", "kbad"} or re.search(r"\d|" + K_WORD_CUE[lang], plain):
            problems.append("count_given_in_missing_count_template")
    elif reason == "invalid_parameter":
        if "cluster" not in fam or "feats" not in s or "kbad" not in s:
            problems.append("invalid_parameter_without_cluster_feats_kbad")
    else:
        problems.append(f"unknown_clarify_reason:{reason}")
    if hint is not None and hint not in {"inspect", "relate", "compare", "cluster"}:
        problems.append(f"unknown_operation_hint:{hint}")
    return problems


def lineage_problems(lin: Any) -> list[str]:
    """Both languages of one lineage plus their parity (same slots, same label)."""
    out: list[str] = []
    for lang, text in (("tr", lin.tr), ("en", lin.en)):
        out += [
            f"{lang}:{p}" for p in template_problems(lin.frame, lin.hint, lin.fixed, text, lang)
        ]
    if slots(lin.tr) != slots(lin.en):
        out.append("slot_sets_differ_between_languages")
    return out
