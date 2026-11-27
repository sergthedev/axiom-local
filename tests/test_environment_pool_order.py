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
