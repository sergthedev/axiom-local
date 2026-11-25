"""Phase 3A review regressions (PR #8 review of 51a6a3a), Guard blocker 1.

Router answers are bound to a trusted request context created before inference. Each test was
written against the reviewed head first: there, a late candidate resolved ``c1`` against the new
dataset and opened a valid preview, and a non-ASCII confirmation digest raised ``TypeError``.
Every test instruments the executor and requires zero dispatches unless a confirmation is given.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from _helpers import make_dataset
from axiom import RelatePlan
from axiom_router import Guard, GuardError, ReviewStatus, build_router_input
from axiom_router.guard import GUARD_POLICY_VERSION

SPEC_A = {"income": "numeric", "age": "numeric", "region": "text"}
SPEC_B = {"debt": "numeric", "score": "numeric", "city": "text"}
A = make_dataset("income,age,region\n1,2,x\n3,5,y\n4,4,x\n6,7,y\n", SPEC_A)
A_CHANGED = make_dataset("income,age,region\n10,2,x\n30,5,y\n40,4,x\n60,7,y\n", SPEC_A)
B = make_dataset("debt,score,city\n9,1,p\n8,3,q\n7,2,p\n5,5,q\n", SPEC_B)
PLAN_TEXT = json.dumps(
    {
        "version": "axiom.candidate/1",
        "action": "PLAN",
        "operation": "relate",
        "x": "c1",
        "y": "c2",
        "method": "pearson",
    }
)


class Spy:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, dataset, bound, limits):
        self.calls += 1


def _guard(ds=A):
    spy = Spy()
    return Guard(ds, executor=spy), spy


def _stale(review, code: str) -> None:
    assert review.status is ReviewStatus.STALE_RESPONSE
    assert review.code == code
    assert review.confirmation is None
    assert not review.structurally_valid and not review.semantically_valid


def test_policy_version_was_bumped() -> None:
    assert GUARD_POLICY_VERSION == "axiom.guard-policy/2"


def test_context_carries_router_input_and_no_authority_to_the_router() -> None:
    guard, _ = _guard()
    ctx = guard.begin_request("Gelir ile yaş arasındaki Pearson korelasyonu")
    assert ctx.router_input == build_router_input(ctx.router_input.request, guard.schema_view)
    payload = ctx.router_input.to_json()
    assert ctx.request_id not in payload  # the router never sees the request id
    assert ctx.dataset_raw_sha256 not in payload
    assert ctx.schema_sha256 == guard.schema_view.schema_sha256


@pytest.mark.parametrize("new", [B, A_CHANGED], ids=["other_schema", "same_schema_changed_data"])
def test_late_answer_after_dataset_switch_is_refused(new) -> None:
    guard, spy = _guard()
    ctx = guard.begin_request("relate the first two columns with Pearson")
    guard.set_dataset(new)  # the person switched data while the router was running
    _stale(guard.review_candidate_text(ctx, PLAN_TEXT), "request_stale")
    assert spy.calls == 0
    # A fresh request against the new dataset works normally.
    review = guard.review_candidate_text(guard.begin_request("again"), PLAN_TEXT)
    assert review.status is ReviewStatus.AWAITING_CONFIRMATION
    assert review.confirmation.dataset_raw_sha256 == new.raw_sha256


def test_switch_back_to_the_same_dataset_does_not_revive_a_request() -> None:
    guard, spy = _guard()
    ctx = guard.begin_request("q")
    guard.set_dataset(B)
    guard.set_dataset(A)  # same digests as when the request was made
    _stale(guard.review_candidate_text(ctx, PLAN_TEXT), "request_stale")
    assert spy.calls == 0


def test_out_of_order_answers() -> None:
    guard, spy = _guard()
    first = guard.begin_request("first question")
    second = guard.begin_request("second question")
    _stale(guard.review_candidate_text(first, PLAN_TEXT), "request_superseded")
    assert guard.review_candidate_text(second, PLAN_TEXT).status is (
        ReviewStatus.AWAITING_CONFIRMATION
    )
    assert spy.calls == 0


def test_one_answer_per_request() -> None:
    guard, spy = _guard()
    ctx = guard.begin_request("q")
    view = guard.review_candidate_text(ctx, PLAN_TEXT).confirmation
    _stale(guard.review_candidate_text(ctx, PLAN_TEXT), "request_answered")
    guard.confirm(view.token, view.confirmation_digest)  # the first preview is unaffected
    assert spy.calls == 1


def test_cancelled_and_failed_requests_refuse_late_answers() -> None:
    guard, spy = _guard()
    ctx = guard.begin_request("q")
    guard.cancel_request(ctx)
    _stale(guard.review_candidate_text(ctx, PLAN_TEXT), "request_cancelled")
    ctx2 = guard.begin_request("q2")
    timeout = guard.router_failure(ctx2, ReviewStatus.ROUTER_TIMEOUT)
    assert timeout.status is ReviewStatus.ROUTER_TIMEOUT
    _stale(guard.review_candidate_text(ctx2, PLAN_TEXT), "request_failed")
    _stale(guard.router_failure(ctx2, ReviewStatus.ROUTER_ERROR), "request_failed")
    assert spy.calls == 0


def test_forged_or_foreign_contexts_are_unknown() -> None:
    guard, spy = _guard()
    other, _ = _guard()
    ctx = guard.begin_request("q")
    foreign = other.begin_request("q")
    _stale(guard.review_candidate_text(foreign, PLAN_TEXT), "unknown_request")
    forged = dataclasses.replace(ctx, schema_sha256="0" * 64)
    _stale(guard.review_candidate_text(forged, PLAN_TEXT), "unknown_request")
    for junk in (None, "q", {"request_id": ctx.request_id}):
        _stale(guard.review_candidate_text(junk, PLAN_TEXT), "unknown_request")
    # the genuine context is still open and usable once
    assert guard.review_candidate_text(ctx, PLAN_TEXT).status is (
        ReviewStatus.AWAITING_CONFIRMATION
    )
    assert spy.calls == 0


def test_new_request_invalidates_open_preview() -> None:
    guard, spy = _guard()
    view = guard.review_candidate_text(guard.begin_request("q"), PLAN_TEXT).confirmation
    guard.begin_request("a different question")
    with pytest.raises(GuardError) as info:
        guard.confirm(view.token, view.confirmation_digest)
    assert info.value.code == "preview_superseded"
    assert spy.calls == 0


def test_late_model_answer_cannot_replace_a_manual_preview() -> None:
    guard, spy = _guard()
    ctx = guard.begin_request("q")
    manual = guard.propose_manual(
        RelatePlan(
            schema_version="axiom.plan/1",
            operation="relate",
            x="age",
            y="income",
            method="spearman",
        )
    ).confirmation
    _stale(guard.review_candidate_text(ctx, PLAN_TEXT), "request_superseded")
    guard.confirm(manual.token, manual.confirmation_digest)
    assert spy.calls == 1


def test_stale_check_happens_before_parsing() -> None:
    guard, _ = _guard()
    ctx = guard.begin_request("q")
    guard.set_dataset(B)
    _stale(guard.review_candidate_text(ctx, "not json at all"), "request_stale")


def test_router_input_errors_leave_no_open_request() -> None:
    from axiom_router import RouterInputError

    guard, _ = _guard()
    ctx = guard.begin_request("q")
    with pytest.raises(RouterInputError):
        guard.begin_request("")
    # a rejected request does not supersede the open one
    assert guard.review_candidate_text(ctx, PLAN_TEXT).status is (
        ReviewStatus.AWAITING_CONFIRMATION
    )


@pytest.mark.parametrize(
    "digest",
    [
        "ü" * 64,  # non-ASCII: hmac.compare_digest raised TypeError on the reviewed head
        "é",
        "A" * 64,  # uppercase hex
        "0" * 63,
        "0" * 65,
        "",
        "0" * 64 + "\n",
        b"0" * 64,
        None,
        64,
    ],
)
def test_malformed_digest_is_a_structured_refusal(digest) -> None:
    guard, spy = _guard()
    view = guard.review_candidate_text(guard.begin_request("q"), PLAN_TEXT).confirmation
    with pytest.raises(GuardError) as info:
        guard.confirm(view.token, digest)
    assert info.value.code == "malformed_confirmation"
    assert spy.calls == 0
    guard.confirm(view.token, view.confirmation_digest)  # the preview stays valid
    assert spy.calls == 1


@pytest.mark.parametrize("token", ["ü" * 32, "x" * 31, "x" * 33, "a b" + "x" * 29, "", None])
def test_malformed_token_is_a_structured_refusal(token) -> None:
    guard, spy = _guard()
    view = guard.review_candidate_text(guard.begin_request("q"), PLAN_TEXT).confirmation
    with pytest.raises(GuardError) as info:
        guard.confirm(token, view.confirmation_digest)
    assert info.value.code == "malformed_confirmation"
    assert spy.calls == 0
