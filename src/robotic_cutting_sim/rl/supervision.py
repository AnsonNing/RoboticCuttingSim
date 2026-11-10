"""P1-4 supervision (T4.1): crash-safe ledger, failure classification, ``info`` adapter and a process supervisor.

Torch- and Warp-free (importable in the project ``.venv``). Plan: ``docs/reports/rl_ready/P1_5_PLAN.md`` §6.4.

* ``Ledger``: append-only JSONL, one ``os.write`` per record followed by ``fsync``; a crash loses at most the
  record being written. ``read_ledger`` tolerates a truncated last line.
* ``classify_process_end``: every child end is COMPLETE or exactly one infrastructure class (``INFRA_CLASSES``),
  from the return code / signal, the child's ``infra_error`` record, the stderr tail and the supervisor's own
  watchdog reason. ``classify_episode_end``: every episode end is a task outcome, a task-level termination, an
  env-level invalid state or a truncation; never an infrastructure class of the process.
* ``step_status``: the integration adapter over the vector env ``info``. It reads ``info["termination_reason"]``
  (``rl.termination``, ``termination_reason_v1``: bit 64 ``INFRASTRUCTURE_ERROR`` is the env's own
  infrastructure verdict) when present and falls back to the pre-P1-2b ``invalid_reason`` otherwise.
* Two pre-declared episode/process splits (config ``classification.split``):
  ``p1_4_v1`` (``long_rollout_v1``: numerical invalid bits are an env-level infrastructure class, a strict-mode
  invalid env is ``INFRA_INVALID_STATE``) and ``termination_reason_v1`` (``long_rollout_v2``: the env's own
  classification, ``rl.termination.classify_exception`` / ``classify_process_exit``: only bit 64 /
  ``VectorEnvInfrastructureError`` / a crash is infrastructure; ``MATERIAL_INVALID`` incl. non-finite state and a
  strict-mode invalid env are task-level).
* ``Supervisor``: runs each attempt in its own child process (one material = one device HashGrid per process,
  the Warp 1.17 static-descriptor hazard), watches heartbeats (hang) and wall time, samples per-PID VRAM and
  RSS, classifies the end and restarts per policy into a NEW attempt directory (never overwritten).

Fault injection lives only in the child script (``scripts/rl_ready/run_long_rollout.py``, ``RCS_FAULT_INJECT``),
never in library code.
"""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time

import numpy as np

LEDGER_SCHEMA = "rcs_rl_ledger_v1"
SUPERVISOR_SCHEMA = "rcs_rl_supervisor_v1"
MASK64 = (1 << 64) - 1

# ----------------------------------------------------------------------------- classes

COMPLETE = "COMPLETE"
# infrastructure class -> family (OOM, CUDA/kernel error, segfault/signal, timeout, NaN/invalid state, ...)
INFRA_CLASSES = {
    "INFRA_OOM_HOST": "OOM",
    "INFRA_OOM_DEVICE": "OOM",
    "INFRA_CUDA_ERROR": "CUDA_KERNEL_ERROR",
    "INFRA_CRASH_SIGNAL": "SEGFAULT_SIGNAL",
    "INFRA_KILLED": "SEGFAULT_SIGNAL",
    "INFRA_HANG": "TIMEOUT",
    "INFRA_TIMEOUT": "TIMEOUT",
    "INFRA_INVALID_STATE": "NAN_INVALID_STATE",
    "INFRA_DISK": "DISK",
    "INFRA_PYTHON_EXCEPTION": "PYTHON_EXCEPTION",
    "INFRA_CONTACT_POOL_OVERFLOW": "SIMULATOR_CAPACITY",
    "INFRA_UNKNOWN_EXIT": "UNKNOWN",
}
TASK_STRICT_INVALID_ENV = "TASK_STRICT_INVALID_ENV"   # split termination_reason_v1: uncaught strict invalid env
SPLITS = ("p1_4_v1", "termination_reason_v1")
INFRA_FAMILIES = tuple(dict.fromkeys(INFRA_CLASSES.values()))

# episode end classes (never mixed with the process classes above)
TASK_SUCCESS, TASK_FAILURE = "TASK_SUCCESS", "TASK_FAILURE"
TERMINATED = "TERMINATED"                    # task-level termination (policy drove the env out of its domain)
ENV_INVALID_STATE = "ENV_INVALID_STATE"      # split p1_4_v1: env-level numerical failure (NaN / overflow); infra
ENV_INFRASTRUCTURE = "ENV_INFRASTRUCTURE"    # split termination_reason_v1: env bit 64 (OOM / CUDA / pool); infra
TRUNCATED_BY_FORCED_RESET = "TRUNCATED_BY_FORCED_RESET"
TRUNCATED_RUN_END = "TRUNCATED_RUN_END"
INFRA_INCOMPLETE = "INFRA_INCOMPLETE"        # in flight when the process died (written by the supervisor)
EPISODE_CLASSES = (TASK_SUCCESS, TASK_FAILURE, TERMINATED, ENV_INVALID_STATE, ENV_INFRASTRUCTURE,
                   TRUNCATED_BY_FORCED_RESET, TRUNCATED_RUN_END, INFRA_INCOMPLETE)
QUOTA_CLASSES = (TASK_SUCCESS, TASK_FAILURE, TERMINATED, ENV_INVALID_STATE, ENV_INFRASTRUCTURE)  # episodes per env
FULL_LENGTH_CLASSES = (TASK_SUCCESS, TASK_FAILURE)
SUCCESS_DENOMINATOR_CLASSES = (TASK_SUCCESS, TASK_FAILURE)                  # invalid / terminated excluded

# termination_reason bits: the env's own module (torch- and Warp-free), termination_reason_v1
from .termination import (TERM_INFRASTRUCTURE, TERM_MATERIAL_INVALID, TERM_TRUNCATED, TERMINAL_BITS,  # noqa: E402
                          TERMINATION_REASONS, TERMINATION_VERSION, classify_process_exit, material_termination_bits)
# material invalid_reason bits: rl.termination's Warp-free copy of batched.material_batch.INVALID_REASONS (equality
# tested; until the P1 consolidation this module kept its own copy without 128 / 256)
from .termination import MATERIAL_INVALID_REASONS as INVALID_REASONS  # noqa: E402
DEFAULT_NUMERICAL_INVALID = ("NONFINITE_STATE", "SLOT_OVERFLOW", "NONFINITE_WRENCH")
# optional explicit per-env infrastructure keys (none in termination_reason_v1, where bit 64 carries it); read when
# present so a later info revision is not silently ignored
INFRA_INFO_KEYS = ("infra_error", "infrastructure_error", "infra_reason", "infrastructure_failure")
VECTOR_ENV_INFRA_KINDS = {"OOM": "INFRA_OOM_DEVICE", "CUDA_ERROR": "INFRA_CUDA_ERROR", "WARP_ERROR": "INFRA_CUDA_ERROR",
                          "CONTACT_POOL_OVERFLOW": "INFRA_CONTACT_POOL_OVERFLOW",
                          "INFRASTRUCTURE_OTHER": "INFRA_CUDA_ERROR",
                          # a second live CUDA HashGrid in the process: a harness/process misuse (no new class, the
                          # declared long_rollout configs fix the class set)
                          "FOREIGN_DEVICE_HASHGRID": "INFRA_PYTHON_EXCEPTION"}


def decode_bits(value, table):
    return [name for bit, name in table.items() if int(value) & bit]


def bits_of(names, table):
    lookup = {name: bit for bit, name in table.items()}
    unknown = [n for n in names if n not in lookup]
    if unknown:
        raise ValueError(f"unknown reason names {unknown}")
    out = 0
    for name in names:
        out |= lookup[name]
    return out


# ----------------------------------------------------------------------------- seeds

def splitmix64(x):
    """SplitMix64 of a Python int (same mixing as ``rl.vector_env.splitmix64``, unsigned 64-bit result)."""
    z = (int(x) + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def episode_seed(base_seed, env, episode):
    """Deterministic per-(env, episode) seed in [0, 2^63): SplitMix64(SplitMix64(base + env) + episode)."""
    return splitmix64((splitmix64(int(base_seed) + int(env)) + int(episode)) & MASK64) & ((1 << 63) - 1)


# ----------------------------------------------------------------------------- ledger

def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


class Ledger:
    """Append-only JSONL file; every record is one ``os.write`` then ``os.fsync`` (crash loses at most it).

    The file is created exclusively: an existing ledger is never reopened or overwritten."""

    def __init__(self, path, *, fsync=True, schema=LEDGER_SCHEMA):
        self.path = Path(path)
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o644)
        self.fsync, self.schema, self.seq = bool(fsync), schema, 0
        self._lock = threading.Lock()

    def write(self, record_type, **fields):
        with self._lock:
            record = {"type": str(record_type), "seq": self.seq, "t": time.time(), **fields}
            if self.seq == 0:
                record.setdefault("schema", self.schema)
            data = (json.dumps(record, sort_keys=True, default=_json_default) + "\n").encode()
            view = memoryview(data)
            while view:
                view = view[os.write(self._fd, view):]
            if self.fsync:
                os.fsync(self._fd)
            self.seq += 1
            return record

    def close(self):
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_ledger(path):
    """``(records, report)``; a last line without newline or unparsable is reported as ``truncated_tail``."""
    path = Path(path)
    report = dict(path=str(path), exists=path.exists(), truncated_tail=False, corrupt_lines=0, records=0)
    if not path.exists():
        return [], report
    data = path.read_bytes()
    lines = data.split(b"\n")
    tail = lines.pop()               # b"" when the file ends with a newline
    records = []
    for line in lines:
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            report["corrupt_lines"] += 1
    if tail.strip():
        report["truncated_tail"] = True
        try:                         # a complete record whose newline was lost is still a record
            records.append(json.loads(tail))
            report["truncated_tail_parsed"] = True
        except ValueError:
            report["truncated_tail_parsed"] = False
    report["records"] = len(records)
    return records, report


class LedgerFollower:
    """Incremental reader of a growing ledger (complete lines only), for the supervisor watchdog."""

    def __init__(self, path):
        self.path, self.offset, self._partial = Path(path), 0, b""

    def poll(self):
        if not self.path.exists():
            return []
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read()
        self.offset += len(data)
        data = self._partial + data
        lines = data.split(b"\n")
        self._partial = lines.pop()
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out


# ----------------------------------------------------------------------------- error text classification

_OOM_DEVICE = re.compile(r"OutOfMemoryError|CUDA out of memory|cudaErrorMemoryAllocation|CUDA error 2\b|"
                         r"CUDA_ERROR_OUT_OF_MEMORY|out of memory|Failed to allocate .*(device|cuda)|"
                         r"Warp CUDA error 2\b", re.IGNORECASE)
_OOM_HOST = re.compile(r"\bMemoryError\b|std::bad_alloc|Cannot allocate memory")
_DISK = re.compile(r"No space left on device|ENOSPC|Disk quota exceeded|\[Errno (28|122|5)\]|Input/output error")
_CUDA = re.compile(r"CUDA error|cudaError|CUDA_ERROR|illegal memory access|unspecified launch failure|"
                   r"Warp CUDA error|CUDA driver error|device-side assert|CUBLAS_STATUS|illegal instruction|"
                   r"misaligned address|launch timed out", re.IGNORECASE)
_INVALID = re.compile(r"\bFloatingPointError\b|\bInvalidBatchEnvs\b|Invalid batched envs|Invalid rigid worlds|"
                      r"non-?finite|NaN detected", re.IGNORECASE)


def classify_error_text(exc_type, message=""):
    """Infrastructure class of one exception (type name + message), or None if nothing specific matches."""
    text = f"{exc_type or ''}: {message or ''}"
    pool = re.search(r"CONTACT_POOL_OVERFLOW", text)
    if pool:
        return "INFRA_CONTACT_POOL_OVERFLOW"
    if exc_type == "MemoryError":                     # Python host allocation failure
        return "INFRA_OOM_HOST"
    if exc_type in ("OutOfMemoryError", "torch.OutOfMemoryError", "torch.cuda.OutOfMemoryError") \
            or _OOM_DEVICE.search(text):
        return "INFRA_OOM_DEVICE"
    if _OOM_HOST.search(text):
        return "INFRA_OOM_HOST"
    if _DISK.search(text):
        return "INFRA_DISK"
    if _CUDA.search(text):
        return "INFRA_CUDA_ERROR"
    if exc_type in ("FloatingPointError", "InvalidBatchEnvs") or _INVALID.search(text):
        return "INFRA_INVALID_STATE"
    return None


def signal_name(number):
    try:
        return signal.Signals(int(number)).name
    except ValueError:
        return f"SIG{int(number)}"


def kernel_oom_evidence(pid, *, runner=None):
    """Kernel OOM-killer evidence for ``pid`` from ``dmesg`` / ``journalctl -k`` when readable; else UNVERIFIED."""
    def run(args):
        if runner is not None:
            return runner(args)
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout if out.returncode == 0 else None
    for args in (["dmesg"], ["journalctl", "-k", "--no-pager", "-n", "2000"]):
        text = run(args)
        if text is None:
            continue
        hit = [line for line in text.splitlines()
               if re.search(rf"(Killed process |oom-kill.*pid=){int(pid)}\b", line)]
        return dict(status="VERIFIED_OOM_KILL" if hit else "NO_OOM_RECORD", source=args[0], lines=hit[-3:])
    return dict(status="UNVERIFIED", source=None, lines=[])


def classify_process_end(returncode, *, records=(), stderr_tail="", supervisor_reason=None, pid=None,
                         oom_probe=kernel_oom_evidence, split="p1_4_v1"):
    """One process end -> dict(cls, family, infrastructure, env_exit, detail...). See the module docstring.

    Precedence: supervisor watchdog reason (hang / wall timeout), then COMPLETE (exit 0 with an ``end`` record),
    then the child's last ``infra_error`` record, then the signal (refined by CUDA / OOM text in stderr), then
    the stderr text of a non-zero exit. ``env_exit`` is always the env's own ``classify_process_exit``; with split
    ``termination_reason_v1`` the env's category decides task vs infrastructure: an ``infra_error`` record the child
    classified with ``rl.termination.classify_exception`` (``env_category``) is mapped through its kind, and an
    uncaught strict-mode invalid env is ``TASK_STRICT_INVALID_ENV`` (not infrastructure)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    records = list(records)
    has_end = any(r.get("type") == "end" for r in records)
    infra = [r for r in records if r.get("type") == "infra_error"]
    env_exit = classify_process_exit(returncode, stderr_tail or "")
    out = dict(returncode=returncode, has_end_record=has_end, supervisor_reason=supervisor_reason, split=split,
               env_exit=env_exit)
    sig = None
    if returncode is not None and returncode < 0:
        sig = -returncode
    elif returncode is not None and returncode > 128 and returncode - 128 in (4, 6, 7, 8, 9, 11, 15):
        sig = returncode - 128
    if sig is not None:
        out["signal"] = signal_name(sig)

    def done(cls, **detail):
        if split == "termination_reason_v1" and cls == "INFRA_INVALID_STATE":
            cls = TASK_STRICT_INVALID_ENV          # the env's split: a strict-mode invalid env is a task failure
        out.update(cls=cls, family=INFRA_CLASSES.get(cls), infrastructure=cls in INFRA_CLASSES, **detail)
        out["agrees_with_env_exit"] = (env_exit["category"] == "infrastructure") == out["infrastructure"] \
            if cls != COMPLETE else env_exit["category"] == "ok"
        return out

    if supervisor_reason == "hang":
        return done("INFRA_HANG")
    if supervisor_reason == "wall_timeout":
        return done("INFRA_TIMEOUT")
    if returncode == 0 and has_end:
        return done(COMPLETE)
    if infra:
        last = infra[-1]
        detail = dict(exc_type=last.get("exc_type"), message=str(last.get("message"))[:2000],
                      env_category=last.get("env_category"), env_kind=last.get("env_kind"))
        if split == "termination_reason_v1" and last.get("env_category") == "infrastructure":
            kind = last.get("env_kind")
            cls = "INFRA_OOM_HOST" if (kind == "OOM" and last.get("exc_type") == "MemoryError") else \
                VECTOR_ENV_INFRA_KINDS.get(kind, "INFRA_CUDA_ERROR")
            return done(cls, **detail)
        if split == "termination_reason_v1" and last.get("env_category") == "task":
            return done(TASK_STRICT_INVALID_ENV, **detail)
        cls = classify_error_text(last.get("exc_type"), last.get("message")) or "INFRA_PYTHON_EXCEPTION"
        return done(cls, **detail)
    hint = classify_error_text(None, stderr_tail)
    if sig is not None:
        if hint in ("INFRA_OOM_DEVICE", "INFRA_CUDA_ERROR", "INFRA_OOM_HOST", "INFRA_DISK"):
            return done(hint, stderr_hint=True)
        if sig == signal.SIGKILL:
            evidence = oom_probe(pid) if (oom_probe and pid) else dict(status="UNVERIFIED")
            return done("INFRA_KILLED", kernel_oom=evidence,
                        cause="kernel OOM killer" if evidence.get("status") == "VERIFIED_OOM_KILL" else "UNVERIFIED")
        return done("INFRA_CRASH_SIGNAL")
    if returncode not in (0, None):
        if hint:
            return done(hint, stderr_hint=True)
        if "Traceback" in (stderr_tail or ""):
            return done("INFRA_PYTHON_EXCEPTION", stderr_hint=True)
    return done("INFRA_UNKNOWN_EXIT")


# ----------------------------------------------------------------------------- the info adapter

def _host(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class StepStatus:
    """Host view of one vector-env step (all ``[N]`` numpy arrays)."""

    source: str                       # "termination_reason" (termination_reason_v1) or "legacy_invalid_reason"
    terminated: np.ndarray
    truncated: np.ndarray
    termination_reason: np.ndarray    # bits of TERMINATION_REASONS (derived on the legacy path)
    invalid_reason: np.ndarray        # material bits (INVALID_REASONS), zeros if absent
    infra: np.ndarray                 # per-env infrastructure flag/code from the info, zeros if absent
    infra_key: str = None
    step_count: np.ndarray = None
    extra_keys: list = field(default_factory=list)


def step_status(info, terminated=None, truncated=None, *, num_envs):
    """Adapter over ``open_vector_env``'s ``info`` (see the module docstring); works on Torch or NumPy values."""
    N = int(num_envs)
    zeros = np.zeros(N, np.int64)
    invalid = _host(info.get("invalid_reason"))
    invalid = zeros.copy() if invalid is None else invalid.astype(np.int64).reshape(N)
    if "termination_reason" in info:
        source = "termination_reason"
        reason = _host(info["termination_reason"]).astype(np.int64).reshape(N)
    else:
        source = "legacy_invalid_reason"
        reason = material_termination_bits(invalid).astype(np.int64)
    infra_key = next((k for k in INFRA_INFO_KEYS if k in info), None)
    infra = zeros.copy() if infra_key is None else _host(info[infra_key]).astype(np.int64).reshape(N)
    if source == "termination_reason":
        infra = np.where((reason & TERM_INFRASTRUCTURE) != 0, 1, infra)   # the env's own infrastructure bit
    term = _host(terminated)
    term = ((reason & TERMINAL_BITS) != 0) | (infra != 0) if term is None else term.astype(bool).reshape(N)
    trunc = _host(truncated)
    trunc = np.zeros(N, bool) if trunc is None else trunc.astype(bool).reshape(N)
    step = _host(info.get("step_count"))
    return StepStatus(source=source, terminated=term, truncated=trunc, termination_reason=reason,
                      invalid_reason=invalid, infra=infra, infra_key=infra_key,
                      step_count=None if step is None else step.astype(np.int64).reshape(N),
                      extra_keys=sorted(k for k in info if k not in ("termination_reason", "invalid_reason")))


def classify_episode_end(*, cause, termination_reason=0, invalid_reason=0, infra=0, verdict=None,
                         numerical_invalid=DEFAULT_NUMERICAL_INVALID, split="p1_4_v1"):
    """Episode end class from the end ``cause`` ("full", "terminated", "forced", "run_end") and the reasons.

    ``terminated``, split ``termination_reason_v1`` (the env's own classification): bit 64 or an explicit infra
    flag -> ENV_INFRASTRUCTURE, everything else -> TERMINATED (task level, incl. MATERIAL_INVALID from non-finite
    state; the material detail stays in ``invalid_names``). Split ``p1_4_v1``: an infra flag or a numerical invalid
    bit (``numerical_invalid``) -> ENV_INVALID_STATE, otherwise TERMINATED.
    ``full``: the frozen evaluator verdict PASS -> TASK_SUCCESS, else TASK_FAILURE."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if cause == "forced":
        return TRUNCATED_BY_FORCED_RESET
    if cause == "run_end":
        return TRUNCATED_RUN_END
    if cause == "terminated":
        if split == "termination_reason_v1":
            if int(infra) != 0 or int(termination_reason) & TERM_INFRASTRUCTURE:
                return ENV_INFRASTRUCTURE
            return TERMINATED
        numerical = bits_of(numerical_invalid, INVALID_REASONS)
        if int(infra) != 0 or int(invalid_reason) & numerical:
            return ENV_INVALID_STATE
        return TERMINATED
    if cause == "full":
        return TASK_SUCCESS if verdict == "PASS" else TASK_FAILURE
    raise ValueError(f"unknown episode end cause {cause!r}")


# ----------------------------------------------------------------------------- ledger analysis helpers

def open_segments(records):
    """Per env, segments started (``start``/``reset`` records) without an ``episode`` end record."""
    started, ended = {}, set()
    for r in records:
        if r.get("type") in ("start_segments", "reset"):
            for seg in r.get("segments", []):
                started[(seg["env"], seg["segment"])] = dict(seg, global_step_start=r.get("global_step", 0))
        elif r.get("type") == "episode":
            ended.add((r["env"], r["segment"]))
    return [seg for key, seg in sorted(started.items()) if key not in ended]


def completed_quota(records, num_envs):
    """Quota episodes completed per env and the next schedule episode index per env."""
    counts = np.zeros(num_envs, np.int64)
    next_index = np.zeros(num_envs, np.int64)
    for r in records:
        if r.get("type") == "episode" and r.get("episode_index", -1) >= 0:
            if r.get("end_class") in QUOTA_CLASSES:
                counts[r["env"]] += 1
            next_index[r["env"]] = max(next_index[r["env"]], r["episode_index"] + 1)
    return counts, next_index


# ----------------------------------------------------------------------------- process / memory probes

def proc_status(pid="self"):
    """VmRSS / VmHWM (MiB), threads and open file descriptors of a process (Linux /proc); None if gone."""
    base = Path("/proc") / str(pid)
    try:
        text = (base / "status").read_text()
    except OSError:
        return None
    out = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            out[{"VmRSS": "rss_mib", "VmHWM": "hwm_mib"}[key]] = float(value.split()[0]) / 1024.
        elif key == "Threads":
            out["threads"] = int(value.split()[0])
    try:
        out["open_fds"] = len(os.listdir(base / "fd"))
    except OSError:
        out["open_fds"] = None
    return out


def nvidia_smi_sampler(gpu_index):
    """Per-PID VRAM + co-tenant sampler (``rl.telemetry.take_sample``: compute apps, ``-q -d PIDS``, pmon)."""
    from .telemetry import take_sample

    def sample(pids):
        return take_sample(gpu_index, set(pids))
    return sample


def free_vram_mib(gpu_index, runner=None):
    from .telemetry import _run, parse_query_gpu, GPU_FIELDS
    run = runner or _run
    text = run(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits",
                "-i", str(gpu_index)])
    rows = parse_query_gpu(text or "")
    if not rows:
        return None
    return rows[0]["memory_total_mib"] - rows[0]["memory_used_mib"]


# ----------------------------------------------------------------------------- supervisor

@dataclass
class SupervisorPolicy:
    """Watchdog, timeout and restart policy (``configs/rl_benchmark/long_rollout_v1.json`` ``timeouts`` and
    ``recovery``)."""

    startup_timeout_s: float = 1800.
    hang_min_s: float = 300.
    hang_factor_median_step: float = 20.
    max_wall_s: float = 0.            # 0: no wall limit
    term_grace_s: float = 30.
    poll_s: float = 1.
    sample_interval_s: float = 5.
    max_restarts: int = 1
    restart_on: tuple = ("INFRA_OOM_DEVICE", "INFRA_OOM_HOST", "INFRA_CUDA_ERROR", "INFRA_CRASH_SIGNAL",
                         "INFRA_KILLED", "INFRA_HANG")
    py_spy_on_hang: bool = True
    split: str = "p1_4_v1"                # classification split of the config (``classification.split``)

    @classmethod
    def from_config(cls, config, run=None, **override):
        t = dict(config.get("timeouts", {}), **(run or {}).get("timeouts", {}))     # a run may override
        r = dict(config.get("recovery", {}), **(run or {}).get("recovery", {}))
        values = dict(startup_timeout_s=t.get("startup_s"), hang_min_s=t.get("hang_min_s"),
                      hang_factor_median_step=t.get("hang_factor_median_step"),
                      term_grace_s=t.get("term_grace_s"), sample_interval_s=config.get("cadence", {}).get(
                          "supervisor_sample_interval_s"),
                      max_wall_s=(run or {}).get("max_wall_s"), max_restarts=r.get("max_restarts"),
                      restart_on=tuple(r["restart_on"]) if "restart_on" in r else None,
                      split=config.get("classification", {}).get("split"))
        values.update(override)
        return cls(**{k: v for k, v in values.items() if v is not None})


class Supervisor:
    """Run attempts of one rollout as child processes (see the module docstring).

    ``command(attempt, attempt_dir, resume_dirs)`` returns the child's argv; the child must write its ledger
    to ``attempt_dir / "ledger.jsonl"``. ``sampler(pids)`` returns a VRAM/co-tenant dict (or None to skip).
    Layout: ``run_dir/supervisor.jsonl``, ``run_dir/telemetry.jsonl``, ``run_dir/attempt_NNN/``."""

    def __init__(self, run_dir, command, *, policy=None, sampler=None, env=None, fault_inject=None,
                 fault_attempts=(1,), oom_probe=kernel_oom_evidence, meta=None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)     # never reuse a run directory
        self.command, self.policy = command, policy or SupervisorPolicy()
        self.sampler, self.env, self.oom_probe = sampler, dict(os.environ if env is None else env), oom_probe
        self.fault_inject, self.fault_attempts = fault_inject, tuple(fault_attempts)
        self.ledger = Ledger(self.run_dir / "supervisor.jsonl", schema=SUPERVISOR_SCHEMA)
        self.telemetry = Ledger(self.run_dir / "telemetry.jsonl", schema="rcs_rl_supervisor_telemetry_v1")
        self.meta = dict(meta or {})
        self.attempts = []

    # -- sampling thread
    def _sampler_loop(self, proc, stop, attempt):
        from .telemetry import process_tree
        while not stop.is_set():
            start = time.monotonic()
            pids = process_tree(proc.pid) if proc.poll() is None else {proc.pid}
            record = dict(attempt=attempt, pid=proc.pid, child=proc_status(proc.pid))
            if self.sampler is not None:
                try:
                    record["gpu"] = self.sampler(pids)
                except Exception as error:   # telemetry must never kill the run; record the failure
                    record["gpu_error"] = repr(error)
            self.telemetry.write("sample", **record)
            stop.wait(max(0., self.policy.sample_interval_s - (time.monotonic() - start)))

    def _kill(self, proc, reason, attempt_dir):
        info = dict(reason=reason, pid=proc.pid)
        if reason == "hang" and self.policy.py_spy_on_hang and shutil.which("py-spy"):
            try:
                dump = subprocess.run(["py-spy", "dump", "--pid", str(proc.pid)], capture_output=True, text=True,
                                      timeout=60)
                (attempt_dir / "py_spy_dump.txt").write_text(dump.stdout + dump.stderr)
                info["py_spy_dump"] = "py_spy_dump.txt"
            except (OSError, subprocess.TimeoutExpired) as error:
                info["py_spy_error"] = repr(error)
        else:
            info["py_spy_dump"] = None if reason != "hang" else "py-spy not installed"
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            info["sigterm"] = True
        except ProcessLookupError:
            pass
        try:
            proc.wait(self.policy.term_grace_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                info["sigkill"] = True
            except ProcessLookupError:
                pass
            proc.wait()
        self.ledger.write("watchdog_kill", **info)

    def _attempt(self, attempt, resume_dirs):
        policy = self.policy
        attempt_dir = self.run_dir / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(exist_ok=False)
        argv = [str(a) for a in self.command(attempt, attempt_dir, list(resume_dirs))]
        env = dict(self.env)
        env.pop("RCS_FAULT_INJECT", None)
        if self.fault_inject and attempt in self.fault_attempts:
            env["RCS_FAULT_INJECT"] = self.fault_inject
        stdout = (attempt_dir / "stdout.log").open("xb")
        stderr = (attempt_dir / "stderr.log").open("xb")
        t0 = time.time()
        proc = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        self.ledger.write("launch", attempt=attempt, attempt_dir=attempt_dir.name, argv=argv, pid=proc.pid,
                          resumed_from=[Path(d).name for d in resume_dirs],
                          fault_inject=env.get("RCS_FAULT_INJECT"))
        stop = threading.Event()
        sampler = threading.Thread(target=self._sampler_loop, args=(proc, stop, attempt), daemon=True)
        sampler.start()
        follower = LedgerFollower(attempt_dir / "ledger.jsonl")
        last_activity, step_seen, median_step, reason = time.time(), False, None, None
        last_size = -1
        while proc.poll() is None:
            for record in follower.poll():
                if record.get("type") == "heartbeat" and record.get("global_step") is not None:
                    step_seen = True          # build-phase heartbeats keep the startup limit
                    median_step = record.get("median_step_s") or median_step
            path = attempt_dir / "ledger.jsonl"
            size = path.stat().st_size if path.exists() else -1
            now = time.time()
            if size != last_size:
                last_size, last_activity = size, now
            limit = policy.startup_timeout_s if not step_seen else max(
                policy.hang_min_s, policy.hang_factor_median_step * (median_step or 0.))
            if now - last_activity > limit:
                reason = "hang"
                self.ledger.write("hang_detected", attempt=attempt, silent_s=now - last_activity, limit_s=limit,
                                  step_seen=step_seen, median_step_s=median_step)
                self._kill(proc, reason, attempt_dir)
                break
            if policy.max_wall_s and now - t0 > policy.max_wall_s:
                reason = "wall_timeout"
                self.ledger.write("wall_timeout", attempt=attempt, wall_s=now - t0, limit_s=policy.max_wall_s)
                self._kill(proc, reason, attempt_dir)
                break
            time.sleep(policy.poll_s)
        returncode = proc.wait()
        stop.set()
        sampler.join()
        stdout.close()
        stderr.close()
        records, report = read_ledger(attempt_dir / "ledger.jsonl")
        tail = (attempt_dir / "stderr.log").read_bytes()[-8192:].decode(errors="replace")
        result = classify_process_end(returncode, records=records, stderr_tail=tail, supervisor_reason=reason,
                                      pid=proc.pid, oom_probe=self.oom_probe, split=policy.split)
        incomplete = [] if result["cls"] == COMPLETE else open_segments(records)
        for seg in incomplete:
            self.ledger.write("episode_incomplete", attempt=attempt, end_class=INFRA_INCOMPLETE,
                              infra_cls=result["cls"], **seg)
        summary = dict(attempt=attempt, attempt_dir=attempt_dir.name, pid=proc.pid, wall_s=time.time() - t0,
                       classification=result, ledger_report=report, incomplete_segments=len(incomplete),
                       stderr_tail=tail[-2000:], last_records=[r.get("type") for r in records[-5:]])
        self.ledger.write("exit", **summary)
        return summary, attempt_dir

    def run(self):
        self.ledger.write("supervisor_start", policy=self.policy.__dict__, pid=os.getpid(), meta=self.meta,
                          fault_inject=self.fault_inject, fault_attempts=list(self.fault_attempts))
        resume, attempt, status = [], 1, None
        while True:
            summary, attempt_dir = self._attempt(attempt, resume)
            self.attempts.append(summary)
            cls = summary["classification"]["cls"]
            if cls == COMPLETE:
                status = COMPLETE if attempt == 1 else "COMPLETE_AFTER_RESTART"
                break
            restarts = attempt - 1
            if cls in self.policy.restart_on and restarts < self.policy.max_restarts:
                resume = resume + [attempt_dir]
                self.ledger.write("restart", from_attempt=attempt, to_attempt=attempt + 1, cause=cls,
                                  resumed_from=[Path(d).name for d in resume])
                attempt += 1
                continue
            status = f"FAILED_{cls}"
            break
        tallies = {}
        for a in self.attempts:
            c = a["classification"]["cls"]
            if c in INFRA_CLASSES:
                tallies[c] = tallies.get(c, 0) + 1
        final = dict(status=status, attempts=len(self.attempts), infra_failures=tallies,
                     infra_failure_families={f: sum(n for c, n in tallies.items() if INFRA_CLASSES[c] == f)
                                             for f in INFRA_FAMILIES if any(INFRA_CLASSES[c] == f for c in tallies)},
                     classes=[a["classification"]["cls"] for a in self.attempts])
        self.ledger.write("final", **final)
        self.ledger.close()
        self.telemetry.close()
        return final


__all__ = ["Ledger", "LedgerFollower", "read_ledger", "classify_error_text", "classify_process_end",
           "classify_episode_end", "step_status", "StepStatus", "Supervisor", "SupervisorPolicy", "splitmix64",
           "episode_seed", "open_segments", "completed_quota", "proc_status", "nvidia_smi_sampler",
           "free_vram_mib", "INFRA_CLASSES", "EPISODE_CLASSES", "QUOTA_CLASSES", "SPLITS", "TASK_STRICT_INVALID_ENV",
           "ENV_INFRASTRUCTURE", "TERMINATION_VERSION"]
