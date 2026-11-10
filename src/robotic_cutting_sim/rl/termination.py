"""``termination_reason`` bitfield of the vector envs (P1-2b) and infrastructure-failure classification (P1-4).

Every ``step`` of ``open_vector_env`` (both backends) returns ``info["termination_reason"]``: an ``int32 [N]``
device tensor of the bits below (sticky until the env is reset, except ``TRUNCATED_HORIZON``, which is set while
``step_count >= max_episode_steps``). ``terminated`` is ``(termination_reason & TERMINAL_BITS) != 0``;
``truncated`` is the horizon alone. Bit values 1/2/4/8 are the ones ``rl.lab_batched_env`` has used since S2.

Infrastructure failures (OOM, CUDA/Warp kernel error, MJWarp contact-pool overflow, process crash) are never task
failures: in-process ones become ``INFRASTRUCTURE_ERROR`` (a per-env bit for the shared contact pool, or a raised
:class:`VectorEnvInfrastructureError` with every env marked); a crash that kills the process (SIGSEGV) can only be
seen by the parent, which classifies it with :func:`classify_process_exit`.

Material ``invalid_reason`` -> ``termination_reason`` (``MATERIAL_TERMINATION_VERSION``, both backends): every
material bit (``MATERIAL_INVALID_REASONS``, a Warp-free copy of ``batched.material_batch.INVALID_REASONS`` checked by
``tests/test_rl_termination.py``) sets ``MATERIAL_INVALID`` (2); ``FOREIGN_DEVICE_HASHGRID`` (another live CUDA
HashGrid in the process, a process-level misuse, not a task outcome) also sets ``INFRASTRUCTURE_ERROR`` (64).
Which material bit fired (e.g. ``INPUT_OUT_OF_DOMAIN`` of ``material_input_domain_v1`` vs ``NONFINITE_STATE``) stays
visible in ``info["invalid_reason"]`` and :func:`decode_termination_detail`. Not to be confused with
``MATERIAL_INPUT_DOMAIN_GUARD`` (32), the RL-side input guard of ``rl.material_guard``.
"""
import re
import signal

import numpy as np

TERM_RIGID_INVALID = 1           # rigid state non-finite / edge pose invalid / grasp settle failed (workload B)
TERM_MATERIAL_INVALID = 2        # material invalid_reason != 0 (non-finite state, slot overflow, contact guard, ...)
TERM_GRASP_LOST = 4              # B2 grasp monitor latched GRASP_LOST
TERM_IK_STREAK = 8               # IK-failure streak >= max_consecutive_ik_failures (off by default, live_env semantics)
TERM_TRUNCATED = 16              # step_count >= max_episode_steps (truncation, not termination)
TERM_INPUT_DOMAIN = 32           # material input guard: edge/pad pose left the valid domain; pose held (rl.material_guard)
TERM_INFRASTRUCTURE = 64         # infrastructure, not task: OOM, CUDA/kernel error, shared contact-pool overflow

TERMINATION_REASONS = {TERM_RIGID_INVALID: "RIGID_INVALID", TERM_MATERIAL_INVALID: "MATERIAL_INVALID",
                       TERM_GRASP_LOST: "GRASP_LOST", TERM_IK_STREAK: "IK_FAILURE_STREAK",
                       TERM_TRUNCATED: "TRUNCATED_HORIZON", TERM_INPUT_DOMAIN: "MATERIAL_INPUT_DOMAIN_GUARD",
                       TERM_INFRASTRUCTURE: "INFRASTRUCTURE_ERROR"}
TERMINAL_BITS = sum(TERMINATION_REASONS) & ~TERM_TRUNCATED
# Bits that mean the physical state of the env is not valid (the shaping reward applies its invalid penalty).
INVALID_STATE_BITS = TERM_RIGID_INVALID | TERM_MATERIAL_INVALID | TERM_INPUT_DOMAIN | TERM_INFRASTRUCTURE
TASK_FAILURE_BITS = TERMINAL_BITS & ~TERM_INFRASTRUCTURE
TERMINATION_VERSION = "termination_reason_v1"

# Material invalid_reason bits: Warp-free copy of batched.material_batch.INVALID_REASONS (that module imports Warp);
# tests/test_rl_termination.py asserts equality, so a new material bit cannot stay unmapped.
MATERIAL_INVALID_REASONS = {1: "NONFINITE_STATE", 2: "SLOT_OVERFLOW", 4: "INVALID_MEASURED_POSE",
                            8: "INVALID_CUT_DIGIT", 16: "NONFINITE_WRENCH", 32: "INVALID_OBSTACLE_BOXES",
                            64: "CONTACT_SPEED_GUARD", 128: "INPUT_OUT_OF_DOMAIN", 256: "FOREIGN_DEVICE_HASHGRID"}
# termination bits per material reason (by name). Every reason sets MATERIAL_INVALID; the process-level misuse
# FOREIGN_DEVICE_HASHGRID also sets INFRASTRUCTURE_ERROR.
MATERIAL_REASON_TERMINATION = {
    "NONFINITE_STATE": TERM_MATERIAL_INVALID,
    "SLOT_OVERFLOW": TERM_MATERIAL_INVALID,
    "INVALID_MEASURED_POSE": TERM_MATERIAL_INVALID,
    "INVALID_CUT_DIGIT": TERM_MATERIAL_INVALID,
    "NONFINITE_WRENCH": TERM_MATERIAL_INVALID,
    "INVALID_OBSTACLE_BOXES": TERM_MATERIAL_INVALID,
    "CONTACT_SPEED_GUARD": TERM_MATERIAL_INVALID,               # alias_free_v1 speed certificate (P1-3a)
    "INPUT_OUT_OF_DOMAIN": TERM_MATERIAL_INVALID,               # material_input_domain_v1 (P1-1b B2 follow-up)
    "FOREIGN_DEVICE_HASHGRID": TERM_MATERIAL_INVALID | TERM_INFRASTRUCTURE,   # process-level misuse (review F1)
}
MATERIAL_TERMINATION_VERSION = "material_invalid_termination_v1"
if set(MATERIAL_REASON_TERMINATION) != set(MATERIAL_INVALID_REASONS.values()):   # import-time completeness check
    raise AssertionError("MATERIAL_REASON_TERMINATION must map every material invalid reason")
# material bits grouped by the termination bits they set: ((material mask, termination bits), ...)
MATERIAL_TERMINATION_GROUPS = tuple(
    (sum(bit for bit, name in MATERIAL_INVALID_REASONS.items() if MATERIAL_REASON_TERMINATION[name] == term), term)
    for term in sorted(set(MATERIAL_REASON_TERMINATION.values())))
MATERIAL_INFRASTRUCTURE_MASK = sum(bit for bit, name in MATERIAL_INVALID_REASONS.items()
                                   if MATERIAL_REASON_TERMINATION[name] & TERM_INFRASTRUCTURE)


def decode_termination(value):
    """Names of the bits set in one ``termination_reason`` value (ascending bit order)."""
    value = int(value)
    return [name for bit, name in sorted(TERMINATION_REASONS.items()) if value & bit]


def decode_material_invalid(value):
    """Names of the material ``invalid_reason`` bits set in ``value`` (ascending bit order)."""
    value = int(value)
    return [name for bit, name in sorted(MATERIAL_INVALID_REASONS.items()) if value & bit]


def material_termination_bits(invalid_reason):
    """``termination_reason`` bits of material ``invalid_reason`` values (``MATERIAL_REASON_TERMINATION``).

    Accepts a Torch tensor (stays on its device, no host sync; returns int32), a NumPy array (int32) or an int."""
    if hasattr(invalid_reason, "masked_fill"):          # torch.Tensor
        import torch
        out = torch.zeros_like(invalid_reason, dtype=torch.int32)
        for mask, term in MATERIAL_TERMINATION_GROUPS:
            out = out | torch.where((invalid_reason & mask) != 0, term, 0).to(torch.int32)
        return out
    if isinstance(invalid_reason, (int, np.integer)):
        value, out = int(invalid_reason), 0
        for mask, term in MATERIAL_TERMINATION_GROUPS:
            if value & mask:
                out |= term
        return out
    value = np.asarray(invalid_reason).astype(np.int64)
    out = np.zeros(value.shape, np.int32)
    for mask, term in MATERIAL_TERMINATION_GROUPS:
        out |= np.where((value & mask) != 0, term, 0).astype(np.int32)
    return out


def decode_termination_detail(termination_reason, invalid_reason=0):
    """``termination_reason`` names plus the material reasons behind ``MATERIAL_INVALID`` (host, one env)."""
    return {"termination": decode_termination(termination_reason),
            "material_invalid": decode_material_invalid(invalid_reason),
            "infrastructure": bool(int(termination_reason) & TERM_INFRASTRUCTURE)}


def termination_table():
    """Receipt form of the bitfield, including the material ``invalid_reason`` mapping."""
    return {"version": TERMINATION_VERSION, "bits": {str(bit): name for bit, name in sorted(TERMINATION_REASONS.items())},
            "terminal_mask": TERMINAL_BITS, "invalid_state_mask": INVALID_STATE_BITS,
            "task_failure_mask": TASK_FAILURE_BITS, "infrastructure_bit": TERM_INFRASTRUCTURE,
            "sticky": "until the env is reset (TRUNCATED_HORIZON follows step_count)",
            "material_invalid": {
                "version": MATERIAL_TERMINATION_VERSION,
                "bits": {str(bit): {"name": name, "termination": decode_termination(MATERIAL_REASON_TERMINATION[name])}
                         for bit, name in sorted(MATERIAL_INVALID_REASONS.items())},
                "infrastructure_mask": MATERIAL_INFRASTRUCTURE_MASK,
                "statement": "every material bit sets MATERIAL_INVALID; FOREIGN_DEVICE_HASHGRID (process-level "
                             "misuse) also INFRASTRUCTURE_ERROR; the material bits stay in info['invalid_reason']"}}


# ----------------------------------------------------------------------------- infrastructure failures

class VectorEnvInfrastructureError(RuntimeError):
    """An infrastructure failure inside a vector-env call (never a task outcome).

    ``kind``: ``OOM`` | ``CUDA_ERROR`` | ``WARP_ERROR`` | ``INFRASTRUCTURE_OTHER``. ``termination_reason`` is a host
    ``numpy.int32 [N]`` array with ``TERM_INFRASTRUCTURE`` set for every env (the device may be unusable). The env
    refuses further calls after this error; close it and open a new one (or a new process)."""

    def __init__(self, kind, detail, termination_reason=None):
        self.kind, self.detail = kind, str(detail)
        self.termination_reason = termination_reason
        super().__init__(f"INFRASTRUCTURE_ERROR[{kind}]: {detail}")


# Message patterns of infrastructure (not task) failures. Task failures are raised as InvalidBatchEnvs
# (material, strict mode) or FloatingPointError("Invalid rigid worlds ...") (rigid, strict mode) and are never
# matched here.
_INFRA_PATTERNS = (
    ("OOM", re.compile(r"out of memory|OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation", re.I)),
    ("CUDA_ERROR", re.compile(r"CUDA error|cudaError|CUDA_ERROR|illegal memory access|device-side assert|"
                              r"unspecified launch failure|misaligned address|an illegal instruction", re.I)),
    ("WARP_ERROR", re.compile(r"Warp CUDA error|Failed to (?:launch|capture|allocate)|Error launching kernel|"
                              r"wp\.capture_launch", re.I)),
)


def classify_exception(error):
    """``(category, kind)`` of an exception raised inside a vector-env call.

    ``category`` is ``"infrastructure"`` (OOM, CUDA/Warp error), ``"task"`` (a strict-mode invalid env) or
    ``"caller"`` (anything else: bad input, unsupported configuration, programming error)."""
    try:
        from ..batched.material_batch import InvalidBatchEnvs
    except ImportError:          # Warp not importable: nothing can have raised it
        InvalidBatchEnvs = ()
    if isinstance(error, VectorEnvInfrastructureError):
        return "infrastructure", error.kind
    if isinstance(error, InvalidBatchEnvs):
        names = {name for reasons in getattr(error, "reasons", {}).values() for name in reasons}
        if names & {n for b, n in MATERIAL_INVALID_REASONS.items() if b & MATERIAL_INFRASTRUCTURE_MASK}:
            return "infrastructure", "FOREIGN_DEVICE_HASHGRID"      # process-level misuse, not a task outcome
        return "task", "MATERIAL_INVALID"
    if isinstance(error, FloatingPointError) and str(error).startswith("Invalid rigid worlds"):
        if "CONTACT_POOL_OVERFLOW" in str(error):     # shared MJWarp pool (ADR 0022 risk 1): batch-level
            return "infrastructure", "CONTACT_POOL_OVERFLOW"
        return "task", "RIGID_INVALID"
    if isinstance(error, MemoryError) or type(error).__name__ == "OutOfMemoryError":
        return "infrastructure", "OOM"
    if isinstance(error, (RuntimeError, OSError, SystemError)):
        text = f"{type(error).__name__}: {error}"
        for kind, pattern in _INFRA_PATTERNS:
            if pattern.search(text):
                return "infrastructure", kind
    return "caller", type(error).__name__


def classify_process_exit(returncode, stderr_tail=""):
    """Classify a finished worker process from its return code and the tail of its stderr (P1-4 runners).

    Returns ``dict(category, kind, detail)``: ``ok`` (code 0), ``infrastructure`` (killed by a signal, e.g.
    SIGSEGV in a CUDA graph replay, SIGKILL by the OOM killer, or an uncaught OOM/CUDA error), ``task`` (an uncaught
    strict-mode invalid env) or ``unknown``. A zero exit code is not a task success; read the run's own result."""
    text = stderr_tail or ""
    if returncode == 0:
        return dict(category="ok", kind="EXIT_0", detail="process exited normally (task outcome is in its result)")
    if returncode is not None and returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"SIG{-returncode}"
        return dict(category="infrastructure", kind=f"CRASH_{name}", detail=f"terminated by {name}")
    if returncode in (128 + signal.SIGSEGV, 128 + signal.SIGKILL, 128 + signal.SIGABRT, 128 + signal.SIGBUS):
        name = signal.Signals(returncode - 128).name
        return dict(category="infrastructure", kind=f"CRASH_{name}", detail=f"exit {returncode} (128+{name})")
    if "Fatal Python error: Segmentation fault" in text:
        return dict(category="infrastructure", kind="CRASH_SIGSEGV", detail="faulthandler reported a segfault")
    if "INFRASTRUCTURE_ERROR[" in text:
        kind = re.search(r"INFRASTRUCTURE_ERROR\[(\w+)\]", text).group(1)
        return dict(category="infrastructure", kind=kind, detail="VectorEnvInfrastructureError")
    if ("InvalidBatchEnvs" in text or "Invalid batched envs" in text) and "FOREIGN_DEVICE_HASHGRID" in text:
        return dict(category="infrastructure", kind="FOREIGN_DEVICE_HASHGRID",
                    detail="strict-mode invalid env from another live CUDA HashGrid (process-level misuse)")
    if "InvalidBatchEnvs" in text or "Invalid rigid worlds" in text:
        return dict(category="task", kind="STRICT_INVALID_ENV", detail="strict-mode invalid env raised")
    for kind, pattern in _INFRA_PATTERNS:
        if pattern.search(text):
            return dict(category="infrastructure", kind=kind, detail="matched in stderr")
    return dict(category="unknown", kind=f"EXIT_{returncode}", detail="no infrastructure or task signature found")


__all__ = ["TERM_RIGID_INVALID", "TERM_MATERIAL_INVALID", "TERM_GRASP_LOST", "TERM_IK_STREAK", "TERM_TRUNCATED",
           "TERM_INPUT_DOMAIN", "TERM_INFRASTRUCTURE", "TERMINATION_REASONS", "TERMINAL_BITS",
           "INVALID_STATE_BITS", "TASK_FAILURE_BITS", "decode_termination", "termination_table",
           "MATERIAL_INVALID_REASONS", "MATERIAL_REASON_TERMINATION", "MATERIAL_TERMINATION_VERSION",
           "MATERIAL_TERMINATION_GROUPS", "MATERIAL_INFRASTRUCTURE_MASK", "decode_material_invalid",
           "material_termination_bits", "decode_termination_detail",
           "VectorEnvInfrastructureError", "classify_exception", "classify_process_exit"]
