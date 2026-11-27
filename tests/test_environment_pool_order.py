"""The recorded native-pool list must not depend on the order threadpoolctl discovers pools in."""

from __future__ import annotations

import itertools

import pytest

from axiom import environment

SCIPY_OLD = {
    "user_api": "blas",
    "internal_api": "openblas",
    "prefix": "libscipy_openblas",
    "version": "0.3.31.dev",
    "num_threads": 1,
    "threading_layer": "pthreads",
}
SCIPY_NEW = {**SCIPY_OLD, "version": "0.3.34.237.2"}  # same prefix, other bundled build
OMP = {"user_api": "openmp", "internal_api": "openmp", "prefix": "libgomp", "version": None}


@pytest.fixture
def discovered(monkeypatch: pytest.MonkeyPatch):
    def install(pools: list[dict]) -> None:
        monkeypatch.setattr("threadpoolctl.threadpool_info", lambda: [dict(p) for p in pools])

    return install


def test_same_prefix_pools_are_recorded_in_one_order_whatever_the_discovery_order(
    discovered,
) -> None:
    seen = set()
    for order in itertools.permutations([SCIPY_OLD, SCIPY_NEW, OMP]):
        discovered(list(order))
        seen.add(repr(environment._pools()))
    assert len(seen) == 1


def test_same_prefix_pools_are_ordered_by_version(discovered) -> None:
    discovered([SCIPY_NEW, SCIPY_OLD])
    versions = [p["version"] for p in environment._pools()]
    assert versions == ["0.3.31.dev", "0.3.34.237.2"]


def test_a_missing_version_does_not_break_the_order(discovered) -> None:
    with_version = {**OMP, "version": "1"}
    discovered([OMP, with_version])
    first = environment._pools()
    discovered([with_version, OMP])
    assert environment._pools() == first


def test_sort_by_api_then_prefix_is_unchanged_for_distinct_prefixes(discovered) -> None:
    discovered([OMP, SCIPY_OLD])
    assert [p["user_api"] for p in environment._pools()] == ["blas", "openmp"]


def test_an_absent_field_and_an_explicit_none_are_both_totally_ordered(discovered) -> None:
    """``str`` renders a missing version and ``None`` alike; the canonical JSON breaks the tie."""
    absent = {k: v for k, v in OMP.items() if k != "version"}  # no version key at all
    explicit = {**OMP, "version": None}  # version reported as None
    seen = set()
    for order in itertools.permutations([absent, explicit]):
        discovered(list(order))
        seen.add(repr(environment._pools()))
    assert len(seen) == 1
    discovered([explicit, absent])
    assert sorted(map(repr, environment._pools())) == sorted(map(repr, [absent, explicit]))
    assert len(environment._pools()) == 2  # both kept: neither is dropped as a duplicate


def test_numeric_and_string_representations_do_not_tie(discovered) -> None:
    as_int = {**SCIPY_OLD, "num_threads": 1}
    as_str = {**SCIPY_OLD, "num_threads": "1"}
    seen = set()
    for order in itertools.permutations([as_int, as_str]):
        discovered(list(order))
        seen.add(repr(environment._pools()))
    assert len(seen) == 1


def test_all_permutations_of_a_mixed_set_agree_and_keep_api_then_prefix_first(discovered) -> None:
    absent = {k: v for k, v in OMP.items() if k != "version"}
    pools = [SCIPY_NEW, OMP, SCIPY_OLD, absent, {**OMP, "version": "1"}]
    results = set()
    for order in itertools.permutations(pools):
        discovered(list(order))
        got = environment._pools()
        results.add(repr(got))
        assert [p["user_api"] for p in got] == sorted(p["user_api"] for p in got)
    assert len(results) == 1


def test_a_passport_recorded_in_legacy_discovery_order_is_compared_as_a_different_environment() -> (
    None
):
    """Passports written before the fix hold pools in discovery order. Compared with the same
    environment captured now (sorted), the list differs only by order, the status is ``different``
    on ``threads.detected_pools`` alone, and so the cross-environment numeric criteria apply, never
    the exact-bytes ones. The comparison only reads the environment record; result hashes and the
    numerical kernels are not involved, and the fingerprint schema is unchanged."""
    sorted_now = [SCIPY_OLD, SCIPY_NEW]
    legacy = [SCIPY_NEW, SCIPY_OLD]  # the other discovery order, as an old passport may hold it
    reference = {"threads": {"detected_pools": legacy}}
    current = {"threads": {"detected_pools": sorted_now}}
    result = environment.compare_environments(reference, current)
    assert result["status"] == "different"
    assert [d["field"] for d in result["differences"]] == ["threads.detected_pools"]
    same = environment.compare_environments(current, {"threads": {"detected_pools": sorted_now}})
    assert "threads.detected_pools" not in [d["field"] for d in same["differences"]]
