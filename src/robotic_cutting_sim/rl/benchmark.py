"""Benchmark rows for the three measurement boundaries (RL_READY_PLAN section 2, decision D-03).

A row describes one measured (or NOT_RUN) workload at one boundary. Timings of different boundaries are
never merged into one speedup number; ``reduced_fidelity`` rows are reported separately. ``source`` says
where the numbers come from (``measured`` in this run, ``wp17_evidence`` for re-labelled prior evidence)
and ``notes`` holds the exact scope of the timing (which is required to read the row correctly).
"""
from dataclasses import asdict, dataclass, field, fields
import json
import math

BOUNDARIES = ("single_env_native", "single_env_headless", "batched_headless")
WARM_COLD = ("warm", "cold", "n/a")
SHARING = ("SHARED_GPU", "EXCLUSIVE_GPU", "UNKNOWN")
VERDICTS = ("PASS", "FAIL", "BLOCKED", "NOT_RUN", "REPORTED_NOT_SCORED")


@dataclass
class BenchmarkRow:
    boundary: str
    workload_id: str
    num_envs: int | None
    particles_per_env: int | None
    control_hz: float | None
    material_dt_s: float | None
    substeps_per_control_step: int | None
    cameras: str
    robot_dof: int | None
    contacts: str
    recording: str
    warm_cold: str
    env_steps_per_s: float | None
    aggregate_sim_time_per_wall: float | None
    vram_peak_mib: float | None
    reset_mean_s: float | None
    reset_p95_s: float | None
    physics_verdict: str
    reduced_fidelity: bool
    gpu_name: str | None
    gpu_index: int | None
    gpu_driver: str | None
    gpu_sharing: str
    co_tenant: dict = field(default_factory=dict)
    source_commit: str | None = None
    source: str = "measured"
    notes: str = ""

    def __post_init__(self):
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"boundary must be one of {BOUNDARIES}")
        if self.warm_cold not in WARM_COLD:
            raise ValueError(f"warm_cold must be one of {WARM_COLD}")
        if self.gpu_sharing not in SHARING:
            raise ValueError(f"gpu_sharing must be one of {SHARING}")
        if self.physics_verdict not in VERDICTS:
            raise ValueError(f"physics_verdict must be one of {VERDICTS}")
        if not isinstance(self.reduced_fidelity, bool):
            raise ValueError("reduced_fidelity must be a bool")
        for name in ("env_steps_per_s", "aggregate_sim_time_per_wall", "vram_peak_mib", "reset_mean_s",
                     "reset_p95_s", "control_hz", "material_dt_s"):
            value = getattr(self, name)
            if value is not None and not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be a finite nonnegative number or None")
        if self.physics_verdict == "NOT_RUN" and self.env_steps_per_s is not None:
            raise ValueError("A NOT_RUN row carries no throughput")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        names = {f.name for f in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"Unknown benchmark row fields {sorted(unknown)}")
        return cls(**data)


def rows_to_json(rows):
    return json.dumps([row.to_dict() for row in rows], indent=1, sort_keys=True, allow_nan=False)


def rows_from_json(text):
    return [BenchmarkRow.from_dict(item) for item in json.loads(text)]


MAIN_COLUMNS = ("N", "particles/env", "control Hz", "material dt", "cameras", "env-steps/s",
                "aggregate sim-time/wall", "VRAM peak", "reset p95", "physics verdict")
EXTRA_COLUMNS = ("boundary", "workload", "substeps", "robot DOF", "contacts", "recording", "warm/cold",
                 "reset mean", "reduced fidelity", "GPU", "sharing", "co-tenant", "source", "commit")


def _fmt(value, spec="{:.4g}", unit=""):
    if value is None:
        return "n/a"
    return spec.format(value) + unit


def _co_tenant(stats):
    if not stats:
        return "n/a"
    parts = []
    if stats.get("foreign_processes_max") is not None:
        parts.append(f"procs {stats['foreign_processes_max']}")
    if stats.get("foreign_sm_pct_max") is not None:
        parts.append(f"SM max {stats['foreign_sm_pct_max']:.0f}%")
    if stats.get("foreign_memory_mib_max") is not None:
        parts.append(f"mem max {stats['foreign_memory_mib_max']:.0f} MiB")
    if stats.get("baseline_total_util_pct_max") is not None:
        parts.append(f"baseline util {stats['baseline_total_util_pct_max']:.0f}%")
    return ", ".join(parts) or "n/a"


def row_cells(row):
    return [
        _fmt(row.num_envs, "{}"), _fmt(row.particles_per_env, "{}"), _fmt(row.control_hz, "{:g}"),
        _fmt(row.material_dt_s, "{:g}", " s"), row.cameras, _fmt(row.env_steps_per_s, "{:.3f}"),
        _fmt(row.aggregate_sim_time_per_wall, "{:.4f}"), _fmt(row.vram_peak_mib, "{:.0f}", " MiB"),
        _fmt(row.reset_p95_s, "{:.3f}", " s"), row.physics_verdict,
        row.boundary, row.workload_id, _fmt(row.substeps_per_control_step, "{}"), _fmt(row.robot_dof, "{}"),
        row.contacts, row.recording, row.warm_cold, _fmt(row.reset_mean_s, "{:.3f}", " s"),
        "yes" if row.reduced_fidelity else "no",
        "n/a" if row.gpu_name is None else f"{row.gpu_name} (#{row.gpu_index}, {row.gpu_driver})",
        row.gpu_sharing, _co_tenant(row.co_tenant), row.source, (row.source_commit or "n/a")[:12]]


def markdown_table(rows):
    header = (*MAIN_COLUMNS, *EXTRA_COLUMNS)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("|", "/") for cell in row_cells(row)) + " |")
    return "\n".join(lines) + "\n"


def co_tenant_stats(summary):
    """Compact co-tenant statistics of a telemetry summary for a row."""
    if not summary or summary.get("valid_samples", 0) == 0:
        return {}

    def pick(key, stat):
        value = summary.get(key)
        return None if value is None else value.get(stat)
    return dict(samples=summary["valid_samples"], foreign_processes_max=summary.get("foreign_process_count_max"),
                foreign_compute_pids=summary.get("foreign_compute_pids"),
                foreign_graphics_pids=summary.get("foreign_graphics_pids"),
                foreign_sm_pct_max=pick("foreign_sm_pct", "max"), foreign_sm_pct_mean=pick("foreign_sm_pct", "mean"),
                foreign_memory_mib_max=pick("foreign_memory_mib", "max"),
                foreign_memory_mib_mean=pick("foreign_memory_mib", "mean"),
                total_util_pct_max=pick("total_util_pct", "max"), total_util_pct_mean=pick("total_util_pct", "mean"),
                baseline_total_util_pct_max=pick("baseline_total_util_pct", "max"),
                sharing_reasons=summary.get("sharing_reasons"))


def percentile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    position = (len(values) - 1) * q / 100.
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)
