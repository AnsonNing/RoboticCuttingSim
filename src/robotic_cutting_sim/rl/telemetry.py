"""Background GPU telemetry for one GPU index (decision D-07): ~1 Hz ``nvidia-smi`` samples.

Each sample reads ``--query-gpu`` (total utilisation, memory used/total) and ``--query-compute-apps``
(per-PID compute memory) and, when available, ``-q -d PIDS`` (compute *and* graphics processes; graphics
co-tenants such as offscreen renderers do not appear in the compute list) and ``pmon -c 1`` (per-process
SM utilisation, which separates co-tenant utilisation from ours). Display servers are reported but never
count as co-tenants. Process names are never stored (only PIDs and memory).

Sharing label: ``SHARED_GPU`` when, during the window, any foreign compute process or foreign non-display
graphics process is present, or foreign utilisation (per-process SM %, or total utilisation in the
``baseline`` phase before our work starts) exceeds ``util_threshold_pct``; otherwise ``EXCLUSIVE_GPU``.
``UNKNOWN`` when no sample was taken.
"""
import os
from pathlib import Path
import subprocess
import threading
import time

DISPLAY_NAMES = ("Xorg", "Xwayland", "gnome-shell")
GPU_FIELDS = ("index", "uuid", "name", "driver_version", "utilization.gpu", "memory.used", "memory.total")
APP_FIELDS = ("pid", "gpu_uuid", "used_memory")


def _number(text):
    text = str(text).strip()
    if text in ("", "-", "N/A", "[N/A]", "[Not Supported]"):
        return None
    try:
        return float(text.split()[0])
    except ValueError:
        return None


def parse_query_gpu(text):
    """Rows of ``nvidia-smi --query-gpu=<GPU_FIELDS> --format=csv,noheader,nounits``."""
    rows = []
    for line in text.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(GPU_FIELDS):
            continue
        rows.append(dict(index=int(parts[0]), uuid=parts[1], name=parts[2], driver=parts[3],
                         util_pct=_number(parts[4]), memory_used_mib=_number(parts[5]),
                         memory_total_mib=_number(parts[6])))
    return rows


def parse_compute_apps(text):
    """Rows of ``nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits``."""
    rows = []
    for line in text.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(APP_FIELDS) or not parts[0].isdigit():
            continue
        rows.append(dict(pid=int(parts[0]), gpu_uuid=parts[1], memory_mib=_number(parts[2])))
    return rows


def parse_pids_query(text):
    """Processes of ``nvidia-smi -q -d PIDS -i <index>``: pid, type (C, G, C+G), display flag, memory."""
    rows, current = [], None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "Process ID":
            current = dict(pid=int(value), type=None, display=False, memory_mib=None)
            rows.append(current)
        elif current is not None and key == "Type":
            current["type"] = value
        elif current is not None and key == "Name":
            current["display"] = Path(value).name in DISPLAY_NAMES
        elif current is not None and key == "Used GPU Memory":
            current["memory_mib"] = _number(value)
    return rows


def parse_pmon(text):
    """Rows of ``nvidia-smi pmon -c 1 -s u -i <index>``: pid -> SM utilisation percent (None if '-')."""
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4 or not parts[1].isdigit():
            continue
        rows.append(dict(gpu=int(parts[0]), pid=int(parts[1]), type=parts[2], sm_pct=_number(parts[3])))
    return rows


def process_tree(root_pid):
    """``root_pid`` and all of its descendants (Linux /proc)."""
    pids, frontier = {int(root_pid)}, [int(root_pid)]
    while frontier:
        pid = frontier.pop()
        for children in Path(f"/proc/{pid}/task").glob("*/children"):
            try:
                found = [int(value) for value in children.read_text().split()]
            except OSError:
                continue
            for child in found:
                if child not in pids:
                    pids.add(child)
                    frontier.append(child)
    return pids


def _run(args, timeout=10.):
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def take_sample(gpu_index, own_pids, *, pmon=True, pids_query=True, runner=_run):
    """One telemetry sample (dict) of GPU ``gpu_index``; ``own_pids`` is a set of our process ids."""
    gpu_text = runner(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits",
                       "-i", str(gpu_index)])
    gpu = (parse_query_gpu(gpu_text) or [None])[0] if gpu_text else None
    sample = dict(t=time.time(), gpu=gpu, ok=gpu is not None)
    if gpu is None:
        return sample
    apps = [row for row in parse_compute_apps(runner(["nvidia-smi", f"--query-compute-apps={','.join(APP_FIELDS)}",
                                                      "--format=csv,noheader,nounits"]) or "")
            if row["gpu_uuid"] == gpu["uuid"]]
    processes = parse_pids_query(runner(["nvidia-smi", "-q", "-d", "PIDS", "-i", str(gpu_index)]) or "") \
        if pids_query else []
    sm = parse_pmon(runner(["nvidia-smi", "pmon", "-c", "1", "-s", "u", "-i", str(gpu_index)]) or "") \
        if pmon else []
    own = set(own_pids)
    display = {row["pid"] for row in processes if row["display"]}
    foreign_compute = {row["pid"]: row["memory_mib"] or 0. for row in apps if row["pid"] not in own}
    foreign_graphics = {row["pid"]: row["memory_mib"] or 0. for row in processes
                        if row["pid"] not in own and row["pid"] not in display and row["pid"] not in foreign_compute}
    sample.update(
        own_memory_mib=sum(row["memory_mib"] or 0. for row in apps if row["pid"] in own),
        foreign_compute=foreign_compute, foreign_graphics=foreign_graphics,
        foreign_memory_mib=sum(foreign_compute.values()) + sum(foreign_graphics.values()),
        display_pids=sorted(display),
        own_sm_pct=sum(row["sm_pct"] or 0. for row in sm if row["pid"] in own) if sm else None,
        foreign_sm_pct=sum(row["sm_pct"] or 0. for row in sm if row["pid"] not in own and row["pid"] not in display)
        if sm else None,
        pmon_available=bool(sm) if pmon else False)
    return sample


def summarize(samples, *, util_threshold_pct=5.):
    """Peak own VRAM, foreign utilisation/memory statistics and the sharing label of a sample list."""
    good = [s for s in samples if s.get("ok")]
    if not good:
        return dict(samples=len(samples), valid_samples=0, gpu_sharing="UNKNOWN")

    def stats(values):
        values = [float(v) for v in values if v is not None]
        return dict(max=max(values), mean=sum(values) / len(values), n=len(values)) if values else None

    gpu = good[0]["gpu"]
    run = [s for s in good if s.get("phase") != "baseline"]
    baseline = [s for s in good if s.get("phase") == "baseline"]
    compute_pids = sorted({pid for s in good for pid in s.get("foreign_compute", {})})
    graphics_pids = sorted({pid for s in good for pid in s.get("foreign_graphics", {})})
    foreign_sm = stats(s.get("foreign_sm_pct") for s in good)
    baseline_util = stats(s["gpu"]["util_pct"] for s in baseline)
    reasons = []
    if compute_pids:
        reasons.append("foreign_compute_process")
    if graphics_pids:
        reasons.append("foreign_graphics_process")
    if foreign_sm and foreign_sm["max"] > util_threshold_pct:
        reasons.append("foreign_sm_utilisation")
    if baseline_util and baseline_util["max"] > util_threshold_pct:
        reasons.append("baseline_total_utilisation")
    return dict(
        samples=len(samples), valid_samples=len(good), baseline_samples=len(baseline),
        gpu_index=gpu["index"], gpu_name=gpu["name"], gpu_driver=gpu["driver"], gpu_uuid=gpu["uuid"],
        memory_total_mib=gpu["memory_total_mib"],
        own_vram_peak_mib=max((s.get("own_memory_mib") or 0.) for s in good),
        total_memory_used_peak_mib=max(s["gpu"]["memory_used_mib"] or 0. for s in good),
        total_util_pct=stats(s["gpu"]["util_pct"] for s in run or good),
        baseline_total_util_pct=baseline_util,
        own_sm_pct=stats(s.get("own_sm_pct") for s in good),
        foreign_sm_pct=foreign_sm,
        foreign_memory_mib=stats(s.get("foreign_memory_mib") for s in good),
        foreign_compute_pids=compute_pids, foreign_graphics_pids=graphics_pids,
        foreign_process_count_max=max(len(s.get("foreign_compute", {})) + len(s.get("foreign_graphics", {}))
                                      for s in good),
        util_threshold_pct=util_threshold_pct, sharing_reasons=reasons,
        gpu_sharing="SHARED_GPU" if reasons else "EXCLUSIVE_GPU")


class GpuTelemetry:
    """Background sampler. ``with GpuTelemetry(1) as t: t.set_phase('run'); ...``; then ``t.summary()``.

    ``gpu_index`` is the physical ``nvidia-smi`` index (not the CUDA ordinal inside the process).
    """

    def __init__(self, gpu_index, *, interval_s=1., own_pid=None, util_threshold_pct=5., pmon=True,
                 pids_query=True, runner=_run):
        self.gpu_index, self.interval_s = int(gpu_index), float(interval_s)
        self.own_pid = os.getpid() if own_pid is None else int(own_pid)
        self.util_threshold_pct, self.pmon, self.pids_query, self.runner = util_threshold_pct, pmon, pids_query, runner
        self.samples, self.phase = [], "run"
        self._stop, self._thread, self._lock = threading.Event(), None, threading.Lock()

    def set_phase(self, phase):
        self.phase = str(phase)

    def sample(self):
        sample = take_sample(self.gpu_index, process_tree(self.own_pid), pmon=self.pmon,
                             pids_query=self.pids_query, runner=self.runner)
        sample["phase"] = self.phase
        with self._lock:
            self.samples.append(sample)
        return sample

    def _loop(self):
        while not self._stop.is_set():
            start = time.monotonic()
            self.sample()
            self._stop.wait(max(0., self.interval_s - (time.monotonic() - start)))

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Telemetry already started")
        self._thread = threading.Thread(target=self._loop, name="gpu-telemetry", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
        return self.summary()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def summary(self, phases=None):
        with self._lock:
            samples = [s for s in self.samples if phases is None or s.get("phase") in phases]
        return summarize(samples, util_threshold_pct=self.util_threshold_pct)
