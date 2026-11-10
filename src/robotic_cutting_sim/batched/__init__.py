"""Batched physics for the ``rl_gpu_headless`` runtime (ADR 0021, plan decision D-04).

``layout`` packs N frozen single ``tlmpm_v7_2`` solvers into one flat batch, ``tlmpm_v7_2_batch`` steps
all envs with one launch per operation, ``material_batch`` adds the CUDA graph, coupling, wrench handoff,
validity flags and masked partial reset, ``constitutive`` is the stress-model registry (ADR §11) and
``errors`` holds the explicit UNSUPPORTED refusals. Nothing in the existing simulator imports this package.
"""
from .errors import UnsupportedBatchConfiguration

__all__ = ["UnsupportedBatchConfiguration"]
