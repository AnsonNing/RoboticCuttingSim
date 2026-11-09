"""RL-ready infrastructure (decision D-04): benchmark rows, GPU telemetry and measured-trajectory replay.

Nothing in the existing simulator imports this subpackage; frozen solvers, evaluators and runtimes are
consumed read-only. Submodules import Warp only where a material is actually built (``replay.build_material``).
"""
