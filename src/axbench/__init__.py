"""AXBENCH: the AX-01 intent benchmark tooling (generation, audit, scoring, sealing).

Benchmark tooling only. It imports the router contract and the Core, never a model runtime.
"""

from .generate import CORPUS_VERSION, GENERATOR_VERSION

__all__ = ["CORPUS_VERSION", "GENERATOR_VERSION"]
