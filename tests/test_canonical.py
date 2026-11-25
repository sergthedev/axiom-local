"""Canonical JSON serializer v1."""

from __future__ import annotations

import json
import math

import pytest

from axiom.canonical import canonical_dumps, canonical_sha256, sha256_hex


def test_keys_sorted_by_codepoint_and_no_whitespace() -> None:
    assert canonical_dumps({"b": 1, "a": [1, 2], "é": 0, "Z": 3}) == '{"Z":3,"a":[1,2],"b":1,"é":0}'


def test_signed_zero_is_canonical() -> None:
    assert canonical_dumps({"v": -0.0}) == canonical_dumps({"v": 0.0}) == '{"v":0.0}'


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        canonical_dumps({"v": bad})


def test_unserializable_and_non_string_keys_rejected() -> None:
    with pytest.raises(TypeError):
        canonical_dumps({1: "a"})
    with pytest.raises(TypeError):
        canonical_dumps({"v": object()})


@pytest.mark.parametrize("value", [0.1, 1 / 3, 1e-320, 1.7976931348623157e308, 123456789.123456789])
def test_floats_keep_full_precision_round_trip(value: float) -> None:
    assert json.loads(canonical_dumps({"v": value}))["v"] == value


def test_bool_is_not_int_and_unicode_is_not_escaped() -> None:
    assert canonical_dumps([True, 1, None, "ş"]) == '[true,1,null,"ş"]'


def test_tuples_serialize_as_lists_and_hash_is_stable() -> None:
    assert canonical_dumps({"a": (1, 2)}) == canonical_dumps({"a": [1, 2]})
    assert canonical_sha256({"a": 1}) == sha256_hex(b'{"a":1}')
