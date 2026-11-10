"""RL-ready infrastructure (decision D-04): benchmark rows, GPU telemetry, measured-trajectory replay and the
batched vector env ``open_vector_env`` (ADR 0021 §8).

Nothing in the existing simulator imports this subpackage; frozen solvers, evaluators and runtimes are
consumed read-only. Submodules import Warp only where a material is actually built (``replay.build_material``),
and Torch only in the device modules; ``open_vector_env`` / ``TaskFamily`` are resolved lazily so that the
torch-free modules stay importable in the project ``.venv`` (no Torch).
"""

_LAZY = {"open_vector_env": "vector_env", "TaskFamily": "vector_env", "VectorCuttingEnv": "vector_env",
         "UnsupportedVectorConfiguration": "vector_env", "ConfiguredTaskFamily": "configured_family"}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
