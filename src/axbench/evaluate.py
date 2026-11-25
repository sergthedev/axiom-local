"""AXBENCH evaluation harness: run a router, score it, pass its outputs through the Guard.

Three layers are reported separately (DR-005):

1. **raw candidate** metrics (:mod:`axbench.score`), on the full denominator;
2. **after Guard**: each raw output is reviewed by a real :class:`axiom_router.Guard` bound to a
   small synthetic dataset with the example's exact schema (no confirmation is ever given);
3. **dispatch**: the Guard's executor is instrumented; every call is counted as an unauthorized
   dispatch (the harness never confirms). The expected count is zero; zero observed is not proof.

Systems receive only :class:`axiom_router.RouterInput` (request + schema projection).
"""

from __future__ import annotations

import csv
import hashlib
import io
import random
import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from axiom.ingest import ColumnSpec, CsvProfile, load_csv_bytes
from axiom_router import Guard, ReviewStatus, build_router_input
from axiom_router.prompt import RouterInputError
from axiom_router.rules import Lexicon, RuleRouter
from axiom_router.schema import project_columns

from .families import FAMILIES, GENERIC, STYLES, column_name
from .score import SCORE_VERSION, score_one, slices, summarize

HARNESS_VERSION = "axbench.evaluate/0.1"
Router = Callable[[Any], str]


def training_lexicon() -> Lexicon:
    """Synonym groups from the families that have training instances (the ``shared`` partition).

    This is the rule baseline's "training": it uses only concepts visible in the training split.
    Held-out validation/test families contribute nothing.
    """
    groups: list[tuple[str, ...]] = []
    generic: dict[str, list[int]] = {}
    for fam in FAMILIES:
        if fam.partition != "shared":
            continue
        for concept in fam.concepts:
            forms = {concept.tr, concept.en, *concept.tr_phrases, *concept.en_phrases}
            forms |= {column_name(concept, s, 0) for s in STYLES if s != "mixed"}
            groups.append(tuple(sorted(forms)))
            if concept.generic:
                for form in GENERIC[concept.generic]:
                    generic.setdefault(form, []).append(len(groups) - 1)
    return Lexicon(tuple(groups), {k: tuple(v) for k, v in generic.items()})


def rule_router(lexicon: Lexicon | None = None) -> Router:
    router = RuleRouter(training_lexicon() if lexicon is None else lexicon)
    return router.route


def run_router(
    router: Router, examples: Sequence[dict[str, Any]], *, system: str
) -> list[dict[str, Any]]:
    """Raw predictions. Exceptions become ``error`` statuses; nothing is retried or repaired."""
    preds = []
    for e in examples:
        view = project_columns([(c["name"], c["type"]) for c in e["schema"]["columns"]])
        record: dict[str, Any] = {"example_id": e["id"], "system": system}
        try:
            router_input = build_router_input(e["request"], view)
        except RouterInputError as exc:
            preds.append(
                {
                    **record,
                    "status": "error",
                    "raw_output": None,
                    "elapsed_ms": None,
                    "error": exc.code,
                }
            )
            continue
        start = time.perf_counter()
        try:
            raw = router(router_input)
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - recorded as a technical failure
            raw, status = None, "error"
            record["error"] = type(exc).__name__
        record.update(
            status=status, raw_output=raw, elapsed_ms=(time.perf_counter() - start) * 1000.0
        )
        preds.append(record)
    return preds


def synthetic_dataset(schema: dict[str, Any], rows: int = 24):
    """A small deterministic dataset with exactly this schema (values are arbitrary)."""
    cols = schema["columns"]
    seed = hashlib.sha256(repr([(c["name"], c["type"]) for c in cols]).encode()).hexdigest()
    rng = random.Random(seed)
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([c["name"] for c in cols])
    for _ in range(rows):
        writer.writerow(
            [
                f"{rng.uniform(1, 100):.3f}" if c["type"] == "numeric" else f"g{rng.randrange(4)}"
                for c in cols
            ]
        )
    profile = CsvProfile(columns=tuple(ColumnSpec(name=c["name"], kind=c["type"]) for c in cols))
    return load_csv_bytes(buf.getvalue().encode("utf-8"), profile)


def guard_layer(
    examples: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Review every raw output with a real Guard; never confirm; count executor calls."""
    dispatches = 0

    def spy(dataset, bound, limits):  # pragma: no cover - must never run
        nonlocal dispatches
        dispatches += 1

    statuses: Counter[str] = Counter()
    awaiting_correct = awaiting_wrong = 0
    from .score import semantic_key

    for e, p in zip(examples, predictions, strict=True):
        guard = Guard(synthetic_dataset(e["schema"]), executor=spy)
        if p.get("status") != "ok":
            status = {
                "timeout": ReviewStatus.ROUTER_TIMEOUT,
                "unavailable": ReviewStatus.ROUTER_UNAVAILABLE,
            }.get(p.get("status"), ReviewStatus.ROUTER_ERROR)
            review = Guard.router_failure(status)
        else:
            review = guard.review_candidate_text(p.get("raw_output"))
        statuses[review.status.value] += 1
        if review.status is ReviewStatus.AWAITING_CONFIRMATION:
            from axiom_router import parse_candidate
            from axiom_router.contract import candidate_to_dict

            pred_key = semantic_key(candidate_to_dict(parse_candidate(p["raw_output"])))
            if pred_key == semantic_key(e["gold"]):
                awaiting_correct += 1
            else:
                awaiting_wrong += 1
    return {
        "n_reviews": len(examples),
        "guard_status": dict(sorted(statuses.items())),
        "awaiting_confirmation_correct": awaiting_correct,
        "awaiting_confirmation_wrong": awaiting_wrong,
        "confirmations_given": 0,
        "unauthorized_dispatch_count": dispatches,
        "note": "No confirmation is given by the harness; any executor call would be unauthorized.",
    }


def evaluate(
    examples: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    *,
    system: str,
    split: str,
    bootstrap_reps: int = 2000,
) -> dict[str, Any]:
    by_id = {p["example_id"]: p for p in predictions}
    ordered = [by_id.get(e["id"], {"status": "error", "raw_output": None}) for e in examples]
    missing = sum(1 for e in examples if e["id"] not in by_id)
    scored = [score_one(e, p) for e, p in zip(examples, ordered, strict=True)]
    return {
        "harness_version": HARNESS_VERSION,
        "score_version": SCORE_VERSION,
        "system": system,
        "split": split,
        "missing_predictions_counted_as_error": missing,
        "raw_candidate": summarize(examples, scored, bootstrap_reps=bootstrap_reps),
        "slices": slices(examples, scored),
        "after_guard": guard_layer(examples, ordered),
        "memory": None,
        "memory_note": "not measured for this system",
        "per_example": [s.to_record() for s in scored],
    }
