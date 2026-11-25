"""Input and guard tests: strict, explicit, fail-closed CSV ingestion."""

from __future__ import annotations

import hashlib
import unicodedata

import pytest
from pydantic import ValidationError

from _helpers import make_dataset, make_profile
from axiom import InputError, Limits, load_csv, load_csv_bytes, read_header
from axiom.ingest import SAFE_INTEGER_MAX

NUM = {"x": "numeric"}


def _code(excinfo: pytest.ExceptionInfo[InputError]) -> str:
    return excinfo.value.code


def test_basic_typed_snapshot_and_missing() -> None:
    ds = make_dataset(
        "id,x,name\nA1,1.5,foo\nA2,,bar\nA3,-2,\n", {"id": "text", "x": "numeric", "name": "text"}
    )
    assert ds.n_rows == 3
    assert ds.columns["x"] == (1.5, None, -2.0)
    assert ds.columns["name"] == ("foo", "bar", None)
    assert ds.column_kinds == ("text", "numeric", "text")


def test_semicolon_decimal_comma_utf8_sig_and_turkish_text() -> None:
    raw = "﻿şehir;maaş\nİstanbul;1234,5\nızmir;0,25\n".encode()
    profile = make_profile(
        {"şehir": "text", "maaş": "numeric"}, encoding="utf-8-sig", delimiter=";", decimal=","
    )
    ds = load_csv_bytes(raw, profile)
    assert ds.columns["şehir"] == ("İstanbul", "ızmir")
    assert ds.columns["maaş"] == (1234.5, 0.25)
    assert ds.raw_sha256 == hashlib.sha256(raw).hexdigest()


def test_decimal_comma_requires_semicolon_profile() -> None:
    with pytest.raises(ValidationError):
        make_profile(NUM, decimal=",")


def test_bom_with_plain_utf8_is_rejected() -> None:
    with pytest.raises(InputError) as e:
        load_csv_bytes("﻿x\n1\n".encode(), make_profile(NUM))
    assert _code(e) == "unexpected_bom"


def test_invalid_utf8_and_nul_rejected() -> None:
    with pytest.raises(InputError) as e:
        load_csv_bytes(b"x\n\xff\n", make_profile({"x": "text"}))
    assert _code(e) == "invalid_encoding"
    with pytest.raises(InputError) as e:
        load_csv_bytes(b"x\na\x00b\n", make_profile({"x": "text"}))
    assert _code(e) == "nul_character"


@pytest.mark.parametrize(
    ("token", "code"),
    [
        ("NA", "invalid_numeric"),
        ("N/A", "invalid_numeric"),
        ("null", "invalid_numeric"),
        ("1_000", "invalid_numeric"),
        (" 1", "invalid_numeric"),
        ("1 ", "invalid_numeric"),
        ("0x10", "invalid_numeric"),
        ("1e", "invalid_numeric"),
        ("--1", "invalid_numeric"),
        ("١٢٣", "invalid_numeric"),  # non-ASCII digits
        ("nan", "nonfinite_numeric"),
        ("NaN", "nonfinite_numeric"),
        ("inf", "nonfinite_numeric"),
        ("-Infinity", "nonfinite_numeric"),
        ("1e999", "numeric_overflow"),
        ("-1e999", "numeric_overflow"),
        ("1e-999", "numeric_underflow"),
        ("9007199254740993", "unsafe_integer"),
        ("-9007199254740993", "unsafe_integer"),
        ("123456789012345678901234567890", "unsafe_integer"),
    ],
)
def test_numeric_tokens_rejected_never_coerced(token: str, code: str) -> None:
    with pytest.raises(InputError) as e:
        make_dataset(f"x\n1\n{token}\n", NUM)
    assert _code(e) == code
    assert e.value.details["record_ordinal"] == 2
    assert e.value.details["column"] == "x"


def test_thousands_separator_with_semicolon_profile_is_rejected() -> None:
    with pytest.raises(InputError) as e:
        make_dataset("x\n1,000\n", NUM, delimiter=";")
    assert _code(e) == "invalid_numeric"


def test_dot_rejected_under_decimal_comma() -> None:
    with pytest.raises(InputError) as e:
        make_dataset("x\n1.5\n", NUM, delimiter=";", decimal=",")
    assert _code(e) == "invalid_numeric"


def test_accepted_numeric_forms_and_safe_integer_boundary() -> None:
    ds = make_dataset(
        f"x\n+1.5\n.5\n5.\n1e3\n-2E-2\n{SAFE_INTEGER_MAX}\n-{SAFE_INTEGER_MAX}\n007\n0e-999\n", NUM
    )
    assert ds.columns["x"] == (
        1.5,
        0.5,
        5.0,
        1000.0,
        -0.02,
        float(SAFE_INTEGER_MAX),
        -float(SAFE_INTEGER_MAX),
        7.0,
        0.0,
    )


def test_negative_zero_is_canonicalized() -> None:
    ds = make_dataset("x\n-0\n-0.0\n", NUM)
    assert all(str(v) == "0.0" for v in ds.columns["x"])  # type: ignore[union-attr]


def test_large_integer_identifier_survives_as_text() -> None:
    ds = make_dataset("id\n123456789012345678901234567890\n", {"id": "text"})
    assert ds.columns["id"] == ("123456789012345678901234567890",)


def test_subnormal_value_is_accepted() -> None:
    ds = make_dataset("x\n5e-324\n", NUM)
    assert ds.columns["x"] == (5e-324,)


def test_record_width_and_quoting_errors() -> None:
    with pytest.raises(InputError) as e:
        make_dataset("a,b\n1,2\n3\n", {"a": "numeric", "b": "numeric"})
    assert _code(e) == "record_width" and e.value.details["record_ordinal"] == 2
    with pytest.raises(InputError) as e:
        make_dataset("x\n1,2\n", NUM)
    assert _code(e) == "record_width"
    with pytest.raises(InputError) as e:
        make_dataset('x\n"unterminated\n', {"x": "text"})
    assert _code(e) == "malformed_csv"
    with pytest.raises(InputError) as e:
        make_dataset('x\n"a"b\n', {"x": "text"})
    assert _code(e) == "malformed_csv"


def test_quoted_fields_embedded_newlines_and_crlf() -> None:
    ds = make_dataset(
        't,x\r\n"a,b",1\r\n"line1\nline2",2\r\n"say ""hi""",3\r\n', {"t": "text", "x": "numeric"}
    )
    assert ds.columns["t"] == ("a,b", "line1\nline2", 'say "hi"')
    assert ds.n_rows == 3


def test_header_only_is_zero_rows_and_empty_input_is_an_error() -> None:
    ds = make_dataset("x\n", NUM)
    assert ds.n_rows == 0 and ds.columns["x"] == ()
    ds = make_dataset("x", NUM)
    assert ds.n_rows == 0
    for raw in ("", "\n\n"):
        with pytest.raises(InputError) as e:
            make_dataset(raw, NUM)
        assert _code(e) == "empty_input"


def test_empty_and_duplicate_header_names() -> None:
    with pytest.raises(InputError) as e:
        make_dataset("a,\n1,2\n", {"a": "numeric", "b": "numeric"})
    assert _code(e) == "empty_header_name"
    with pytest.raises(InputError) as e:
        make_dataset("a,a\n1,2\n", {"a": "numeric", "b": "numeric"})
    assert _code(e) == "duplicate_header"


def test_header_nfc_normalization_and_collision() -> None:
    decomposed = unicodedata.normalize("NFD", "é")
    composed = unicodedata.normalize("NFC", "é")
    assert decomposed != composed
    ds = make_dataset(f"{decomposed}\n1\n", {composed: "numeric"})
    assert ds.column_names == (composed,)
    with pytest.raises(InputError) as e:
        make_dataset(f"{decomposed},{composed}\n1,2\n", {"a": "numeric", "b": "numeric"})
    assert _code(e) == "duplicate_header"
    with pytest.raises(ValidationError):
        make_profile({decomposed: "numeric", composed: "numeric"})


def test_header_case_is_preserved_and_case_variants_are_distinct() -> None:
    ds = make_dataset("Ab,aB\n1,2\n", {"Ab": "numeric", "aB": "numeric"})
    assert ds.column_names == ("Ab", "aB")


def test_schema_must_match_header_exactly_and_in_order() -> None:
    for spec in (
        {"a": "numeric"},
        {"b": "numeric", "a": "numeric"},
        {"a": "numeric", "c": "numeric"},
    ):
        with pytest.raises(InputError) as e:
            make_dataset("a,b\n1,2\n", spec)
        assert _code(e) == "schema_mismatch"


def test_group_text_keys_are_not_normalized_or_stripped() -> None:
    nfd = unicodedata.normalize("NFD", "é")
    nfc = unicodedata.normalize("NFC", "é")
    ds = make_dataset(f"g\n{nfd}\n{nfc}\n a\n", {"g": "text"})
    assert ds.columns["g"] == (nfd, nfc, " a")


def test_blank_lines_versus_all_empty_records() -> None:
    ds = make_dataset("a,b\n1,2\n\n,\n3,4\n", {"a": "numeric", "b": "numeric"})
    assert ds.n_blank_lines == 1
    assert ds.n_all_empty_records == 1
    assert ds.n_rows == 3
    assert ds.columns["a"] == (1.0, None, 3.0)
    ds = make_dataset("x\n1\n\n\n2\n", NUM)
    assert ds.n_blank_lines == 2 and ds.n_rows == 2


def test_record_ordinal_is_logical_not_physical_line() -> None:
    raw = 't,x\n"multi\nline",1\n\n"more\nlines",bad\n'
    with pytest.raises(InputError) as e:
        make_dataset(raw, {"t": "text", "x": "numeric"})
    assert e.value.details["record_ordinal"] == 2  # physical line would be 5


def test_limits_are_enforced() -> None:
    profile = make_profile(NUM)
    with pytest.raises(InputError) as e:
        load_csv_bytes(b"x\n1\n2\n3\n", profile, Limits(max_rows=2))
    assert _code(e) == "too_many_rows"
    with pytest.raises(InputError) as e:
        load_csv_bytes(b"x\n1\n", profile, Limits(max_bytes=3))
    assert _code(e) == "input_too_large"
    with pytest.raises(InputError) as e:
        load_csv_bytes(b"x\n" + b"1" * 20 + b"\n", profile, Limits(max_field_chars=10))
    assert _code(e) == "field_too_long"
    with pytest.raises(InputError) as e:
        load_csv_bytes(
            b"a,b\n1,2\n", make_profile({"a": "numeric", "b": "numeric"}), Limits(max_columns=1)
        )
    assert _code(e) == "too_many_columns"


def test_raw_hash_covers_exact_parsed_bytes_and_file_is_read_once(tmp_path) -> None:
    raw = b"x\n1\n2\n"
    path = tmp_path / "d.csv"
    path.write_bytes(raw)
    ds = load_csv(path, make_profile(NUM))
    assert ds.raw_sha256 == hashlib.sha256(raw).hexdigest()
    path.write_bytes(b"x\n9\n9\n")  # later mutation cannot change the snapshot
    assert ds.columns["x"] == (1.0, 2.0)
    with pytest.raises(InputError) as e:
        load_csv(tmp_path / "missing.csv", make_profile(NUM))
    assert _code(e) == "unreadable_file"
    big = tmp_path / "big.csv"
    big.write_bytes(b"x\n" + b"1\n" * 10)
    with pytest.raises(InputError) as e:
        load_csv(big, make_profile(NUM), Limits(max_bytes=5))
    assert _code(e) == "input_too_large"


def test_snapshot_is_immutable() -> None:
    ds = make_dataset("x\n1\n", NUM)
    with pytest.raises(TypeError):
        ds.columns["x"] = (2.0,)  # type: ignore[index]
    with pytest.raises(TypeError):
        ds.columns["x"][0] = 2.0  # type: ignore[index]
    with pytest.raises(AttributeError):
        ds.n_rows = 5  # type: ignore[misc]


def test_logical_hash_is_sensitive_to_values_order_schema_and_framing() -> None:
    base = make_dataset("x\n1\n2\n", NUM).logical_sha256
    assert make_dataset("x\n2\n1\n", NUM).logical_sha256 != base  # order matters
    assert make_dataset("x\n1\n3\n", NUM).logical_sha256 != base  # value matters
    assert make_dataset("x\n1\n\n2\n", NUM).logical_sha256 == base  # blank lines are not data
    assert make_dataset("x\n1\n2\n", {"x": "text"}).logical_sha256 != base
    assert make_dataset("x\r\n1\r\n2\r\n", NUM).logical_sha256 == base  # same typed data
    assert (
        make_dataset("x\r\n1\r\n2\r\n", NUM).raw_sha256 != make_dataset("x\n1\n2\n", NUM).raw_sha256
    )
    # length framing: ("ab","c") vs ("a","bc") must differ, and missing vs empty-string-free text
    a = make_dataset("p,q\nab,c\n", {"p": "text", "q": "text"})
    b = make_dataset("p,q\na,bc\n", {"p": "text", "q": "text"})
    assert a.logical_sha256 != b.logical_sha256
    c = make_dataset("p,q\nab,\n", {"p": "text", "q": "text"})
    assert c.logical_sha256 != a.logical_sha256
    renamed = make_dataset("y\n1\n2\n", {"y": "numeric"})
    assert renamed.logical_sha256 != base


def test_read_header_helper() -> None:
    assert read_header("﻿a;b\n1;2\n".encode(), encoding="utf-8-sig", delimiter=";") == ("a", "b")
    with pytest.raises(InputError):
        read_header(b"")
