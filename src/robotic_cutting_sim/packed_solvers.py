"""Packed-code solver selection (ADR 0014 tlmpm_v5; ADR 0020 proposed: CANDIDATE tlmpm_v6).

``tlmpm_v5`` is the packed-code solver every production path uses. ``tlmpm_v6`` (grid-consistent position update,
WP-12 Phase 4 prototype decision P-A) is an opt-in CANDIDATE: it is selected only when a caller passes
``solver="tlmpm_v6"`` explicitly, never by a default, preset, scene or task (scene configs still declare
``tlmpm_v4``/``tlmpm_v5`` only). Both keep packed 2-bit cut codes with sparse per-node slots, so every packed
consumer (codes, slots, recorders, evaluators) treats them alike.
"""
import importlib

DEFAULT_PACKED_SOLVER = "tlmpm_v5"
CANDIDATE_SOLVERS = ("tlmpm_v6",)
PACKED_SOLVERS = (DEFAULT_PACKED_SOLVER, *CANDIDATE_SOLVERS)


def is_packed_solver(name):
    """True for 'tlmpm_v5' and the opt-in candidate 'tlmpm_v6' (packed codes, sparse slots)."""
    return name in PACKED_SOLVERS


def packed_solver_class(name):
    """The ``TLMPM`` class of a packed solver name (imports only the requested module)."""
    if not is_packed_solver(name):
        raise ValueError(f"packed solver is one of {PACKED_SOLVERS}, not {name!r}")
    return importlib.import_module(f"{__package__}.{name}").TLMPM
