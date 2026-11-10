"""``termination_reason`` bitfield of the vector envs (P1-2b) and infrastructure-failure classification (P1-4).

Every ``step`` of ``open_vector_env`` (both backends) returns ``info["termination_reason"]``: an ``int32 [N]``
device tensor of the bits below (sticky until the env is reset, except ``TRUNCATED_HORIZON``, which is set while
``step_count >= max_episode_steps``). ``terminated`` is ``(termination_reason & TERMINAL_BITS) != 0``;
``truncated`` is the horizon alone. Bit values 1/2/4/8 are the ones ``rl.lab_batched_env`` has used since S2.

Infrastructure failures (OOM, CUDA/Warp kernel error, MJWarp contact-pool overflow, process crash) are never task
failures: in-process ones become ``INFRASTRUCTURE_ERROR`` (a per-env bit for the shared contact pool, or a raised
:class:`VectorEnvInfrastructureError` with every env marked); a crash that kills the process (SIGSEGV) can only be
seen by the parent, which classifies it with :func:`classify_process_exit`.
"""
import re
import signal

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


def decode_termination(value):
    """Names of the bits set in one ``termination_reason`` value (ascending bit order)."""
    value = int(value)
    return [name for bit, name in sorted(TERMINATION_REASONS.items()) if value & bit]


def termination_table():
    """Receipt form of the bitfield."""
    return {"version": TERMINATION_VERSION, "bits": {str(bit): name for bit, name in sorted(TERMINATION_REASONS.items())},
            "terminal_mask": TERMINAL_BITS, "invalid_state_mask": INVALID_STATE_BITS,
            "task_failure_mask": TASK_FAILURE_BITS, "infrastructure_bit": TERM_INFRASTRUCTURE,
            "sticky": "until the env is reset (TRUNCATED_HORIZON follows step_count)"}


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
    if "InvalidBatchEnvs" in text or "Invalid rigid worlds" in text:
        return dict(category="task", kind="STRICT_INVALID_ENV", detail="strict-mode invalid env raised")
    for kind, pattern in _INFRA_PATTERNS:
        if pattern.search(text):
            return dict(category="infrastructure", kind=kind, detail="matched in stderr")
    return dict(category="unknown", kind=f"EXIT_{returncode}", detail="no infrastructure or task signature found")


__all__ = ["TERM_RIGID_INVALID", "TERM_MATERIAL_INVALID", "TERM_GRASP_LOST", "TERM_IK_STREAK", "TERM_TRUNCATED",
           "TERM_INPUT_DOMAIN", "TERM_INFRASTRUCTURE", "TERMINATION_REASONS", "TERMINAL_BITS",
           "INVALID_STATE_BITS", "TASK_FAILURE_BITS", "decode_termination", "termination_table",
           "VectorEnvInfrastructureError", "classify_exception", "classify_process_exit"]
