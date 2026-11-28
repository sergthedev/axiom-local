"""AXBENCH-TR: Turkey-first data acquisition and corpus engineering for AX-01.

Standard library only. This package never touches the deterministic engine (``axiom``), the router
contract (``axiom_router``) or the sealed AXBENCH test split; it only *reads* the router contract to
verify gold plans.
"""

__version__ = "0.2.0-dev"
