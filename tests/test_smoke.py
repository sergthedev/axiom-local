"""Smoke test: the package imports and exposes a version string."""

import axiom


def test_import_and_version() -> None:
    assert isinstance(axiom.__version__, str)
    assert axiom.__version__.count(".") == 2
