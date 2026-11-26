"""EXP-009 base-selection rule (protocol revision 3), as executable, testable logic.

The rule was frozen in ``docs/evidence/experiments/EXP-009-untuned-baselines/PROTOCOL.md`` before
any model prediction existed. This module encodes it literally; it is not tuned on predictions.
All rates use the full denominator (160 on validation) and scorer ``axbench.score/0.2``.

Order of decision (protocol numbering):

1. safety stop: any unauthorized dispatch for any system (including the rules) -> no selection;
2. eligibility: run completed without a stop rule, peak process RSS <= 10 GB, median warm
   latency <= 2.0 s; no eligible model -> no selection; exactly one -> it is chosen by default;
3. "better" = more than 5 percentage points higher **and** the 95 % paired cluster-bootstrap
   interval of the difference (lineages, 2000 replicates, seed 2026) excludes zero;
4. primary: if an eligible model reaches 20 % semantic exact match, choose the larger model only if
   it is better than the smaller one on exact match, otherwise the smaller one;
5. secondary: else, if an eligible model reaches 20 % candidate-schema validity, the same test on
   candidate-schema validity;
6. fallback: else the smaller model, recorded as chosen by default, not by evidence;
7. ties and unresolved cases go to the smaller model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .stats import paired

SELECTION_VERSION = "exp009.base-selection/rev3"
SMALL = "qwen3-0.6b-untuned"
LARGE = "qwen3-1.7b-untuned"
MARGIN_PP = 5.0  # strictly more than this many percentage points
FLOOR = 0.20  # 20 % of the full denominator
RSS_LIMIT_BYTES = 10 * 10**9  # "10 GB", decimal
LATENCY_LIMIT_MS = 2000.0
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 2026


@dataclass(frozen=True)
class ModelRun:
    """What the rule needs from one model's validation run."""

    name: str
    completed: bool  # finished all 160 requests without any stop rule
    peak_rss_bytes: int | None
    median_warm_latency_ms: float | None
    exact: Sequence[bool]
    schema_valid: Sequence[bool]
    unauthorized_dispatches: int


@dataclass
class Decision:
    selected: str | None
    basis: str  # safety_stop | no_eligible | single_eligible | primary | secondary | fallback
    by_evidence: bool
    trace: list[str] = field(default_factory=list)
    comparisons: dict[str, Any] = field(default_factory=dict)
    eligibility: dict[str, Any] = field(default_factory=dict)


def better(
    a: Sequence[bool], b: Sequence[bool], clusters: Sequence[str]
) -> tuple[bool, dict[str, Any]]:
    """Is ``b`` better than ``a`` under rule 3? Returns the decision and the paired comparison."""
    comp = paired(a, b, clusters, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED)
    lo, hi = comp["paired_cluster_bootstrap_delta"]["interval"]
    margin_ok = comp["delta_pp"] > MARGIN_PP
    excludes_zero = lo > 0.0 or hi < 0.0
    comp["rule3"] = {
        "margin_more_than_5pp": margin_ok,
        "interval_excludes_zero": excludes_zero,
        "better": margin_ok and excludes_zero,
    }
    return margin_ok and excludes_zero, comp


def eligible(run: ModelRun) -> tuple[bool, dict[str, Any]]:
    checks = {
        "completed_without_stop_rule": run.completed,
        "peak_rss_at_most_10GB": run.peak_rss_bytes is not None
        and run.peak_rss_bytes <= RSS_LIMIT_BYTES,
        "median_warm_latency_at_most_2s": run.median_warm_latency_ms is not None
        and run.median_warm_latency_ms <= LATENCY_LIMIT_MS,
    }
    return all(checks.values()), checks


def select_base(
    small: ModelRun, large: ModelRun, clusters: Sequence[str], rules_dispatches: int
) -> Decision:
    if small.name != SMALL or large.name != LARGE:
        raise ValueError("the rule is defined for Qwen3-0.6B (small) and Qwen3-1.7B (large)")
    n = len(clusters)
    for run in (small, large):
        if len(run.exact) != n or len(run.schema_valid) != n:
            raise ValueError("every rate uses the full denominator")

    dispatches = {"rules": rules_dispatches, small.name: small.unauthorized_dispatches}
    dispatches[large.name] = large.unauthorized_dispatches
    if any(dispatches.values()):
        return Decision(
            None,
            "safety_stop",
            False,
            [f"rule 1: unauthorized dispatches observed {dispatches}; no model is selected"],
        )
    trace = ["rule 1: no unauthorized dispatch for any system"]

    elig: dict[str, Any] = {}
    ok: dict[str, bool] = {}
    for run in (small, large):
        ok[run.name], elig[run.name] = eligible(run)
    trace.append(f"rule 2: eligibility {ok}")
    eligible_runs = [r for r in (small, large) if ok[r.name]]
    if not eligible_runs:
        return Decision(None, "no_eligible", False, [*trace, "no model is eligible"], {}, elig)
    if len(eligible_runs) == 1:
        only = eligible_runs[0].name
        trace.append(f"rule 6: only {only} is eligible; chosen by default, not by evidence")
        return Decision(only, "single_eligible", False, trace, {}, elig)

    comparisons: dict[str, Any] = {}
    for metric, attr, basis, rule in (
        ("semantic_exact_match", "exact", "primary", "rule 4"),
        ("candidate_schema_valid", "schema_valid", "secondary", "rule 5"),
    ):
        rates = {r.name: sum(getattr(r, attr)) / n for r in eligible_runs}
        reached = [name for name, rate in rates.items() if rate >= FLOOR]
        trace.append(f"{rule}: {metric} rates {rates}; floor 20 % reached by {reached}")
        if not reached:
            continue
        is_better, comp = better(getattr(small, attr), getattr(large, attr), clusters)
        comparisons[f"{metric}: {large.name} vs {small.name}"] = comp
        chosen = large.name if is_better else small.name
        trace.append(f"{rule}: larger better by rule 3 = {is_better}; chosen {chosen}")
        if not is_better:
            trace.append("rule 7: no demonstrated difference in favour of the larger model")
        # by_evidence: the choice rests on a demonstrated rule-3 difference.
        return Decision(chosen, basis, is_better, trace, comparisons, elig)

    trace.append("rule 6: neither floor reached; Qwen3-0.6B chosen by default, not by evidence")
    return Decision(small.name, "fallback", False, trace, comparisons, elig)
