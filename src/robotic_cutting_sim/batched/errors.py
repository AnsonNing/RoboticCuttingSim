"""Explicit refusals of the batched runtime (ADR 0021 §10).

Every configuration the batched path does not implement raises
``UnsupportedBatchConfiguration``; there is never a silent fallback to a Python
loop over single environments. The message always starts with ``UNSUPPORTED``.
"""

CODES = (
    # solvers (§10 "Solvers")
    "UNSUPPORTED_SOLVER",                         # anything but tlmpm_v7_2 (including legacy v1-v4 tool paths)
    "UNSUPPORTED_V7_2_OPTIMIZATIONS_DISABLED",    # v7_2_optimizations=False / optimize_* False rollbacks
    # non-uniform batches (§10, §3.2 "Uniform per batch")
    "UNSUPPORTED_NONUNIFORM_BATCH",               # dt, dx, spacing, substeps, S, K, law, B, contact flags, pic, gravity
    "UNSUPPORTED_PER_ENV_TOOL_PROFILE",           # blade/tool profile differs between envs
    "UNSUPPORTED_LEGACY_FLOOR",                   # floor=True tray
    # recorders and sensors
    "UNSUPPORTED_RECORDER",                       # cut_events, cut_aware_surface, per-step surfaces
    "UNSUPPORTED_SENSOR",                         # camera / vision profiles / RTX sensors
    # geometry
    "UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET",
    "UNSUPPORTED_WORKPIECE_MOTION",
    # not yet built
    "UNSUPPORTED_BACKEND",                        # e.g. isaac_lab_newton before P1-1b
    "UNSUPPORTED_MIXED_CONSTITUTIVE_MODELS",
    # device and size
    "UNSUPPORTED_CPU_WITHOUT_PARITY_MODE",
    "UNSUPPORTED_INT32_OVERFLOW",                 # P*8 or nodes_total*S >= 2**31
    "UNSUPPORTED_EMPTY_BATCH",
    "UNSUPPORTED_SINGLE_ENV_STATE",               # a single solver that is not a fresh, unstepped reset state
)


class UnsupportedBatchConfiguration(ValueError):
    """A batch configuration that ADR 0021 declares unsupported (never a fallback)."""

    def __init__(self, code, detail=""):
        if code not in CODES:
            raise ValueError(f"unknown UNSUPPORTED code {code!r}")
        self.code = code
        self.detail = str(detail)
        message = code if not detail else f"{code}: {detail}"
        if not message.startswith("UNSUPPORTED"):
            raise AssertionError("UNSUPPORTED messages start with UNSUPPORTED")
        super().__init__(message)
