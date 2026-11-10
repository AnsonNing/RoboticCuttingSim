#!/usr/bin/env python3
"""perf_acceptance_v1 — RCS's own performance-acceptance harness (P1-6 R0a, decision D-11).

One process = one side of one run. Workload A (``run_vector_milestones.py``): headless replay of the WP-17 Native
TRIM recordings (seeds 512 / 15) through ``VectorCuttingEnv.step_measured_edge``; 1 mm particles, 4 mm grid,
0.1 ms material dt, 200 substeps per 50 Hz control step, tlmpm_v7_2 law 8, handle + two measured finger pads, no
cameras, no recording, no robot DOF, ``reduced_fidelity`` (ideal edge tracking). The variant (``--variant-json``,
``{}`` = default) goes to ``open_vector_env(**variant)`` unchanged; reserved or unknown keys raise
``UNSUPPORTED_OPTION``.

    PYTHONPATH=<checkout>/src .deps/isaacsim-venv/bin/python scripts/rl_ready/run_perf_acceptance.py \\
        --variant-json V.json --mode throughput|episode|samestate|profile --num-envs N --seeds 512[,15] \\
        --warmup-steps W --steps K [--device cuda:0] [--gpu-uuid UUID | --gpu-index I] [--wp17 DIR] --out DIR

Modes (W + K <= 1200 control steps; the timed window is the K steps after the W warm-up steps):

* ``throughput``: one untimed and 10 timed full resets, W + K replay steps; final codes per env (work check).
* ``episode``: W + K replay steps with the in-process L2 monitor (candidate-reported), per-step contact counts and
  the Native sample steps dumped (codes exact; positions as int8 micrometre deltas to the Native sample with exact
  escapes); exact final snapshot per env; in-process 2-frame ``evaluate()`` (candidate-reported).
* ``samestate``: per seed, the recorded Native state of each ``SAMESTATE_SAMPLES`` sample written into the batch
  and the contact stage run alone twice (impulse, stats); then a W + K step replay from reset (contacts, final).
* ``profile``: as throughput, the K timed steps inside ``cudaProfilerStart/Stop`` (for an external nsys).

Seeds: one batch per round (``perf_acceptance.plan_rounds``), each released before the next is built; at most one
live device ``wp.HashGrid`` in the process at any time (Warp 1.17 static-descriptor hazard) — checked before every
build. Marks ``RCS_PERF_MARK {"event", "global_step", "num_envs"}`` are printed after a device synchronize;
``global_step`` counts vector steps over the whole process.

Output (only under ``--out``, which must not exist): ``result.json`` (``rcs_perf_result_v1``), ``dumps/*.npz`` (no
pickles), ``telemetry_samples.jsonl``. Exit 0: the run completed (whatever its physics). Exit 3: ``result.json``
with ``status: ERROR`` (an error attributable to the code under test: unsupported option, invalid state, missing
attribute). Exit 1 without ``result.json`` (``error.json`` only): an infrastructure failure (CUDA, OOM, I/O; the
caller retries once). Exit 2: usage, or ``--out`` exists (nothing written).
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

T_SCRIPT = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from robotic_cutting_sim.rl import perf_acceptance as PA  # noqa: E402

RESET_REPEATS = {"throughput": 10, "profile": 10, "episode": 3, "samestate": 3}
CANDIDATE_ERRORS = ("UnsupportedBatchConfiguration", "UnsupportedVectorConfiguration", "TypeError", "ValueError",
                    "FloatingPointError", "InvalidBatchEnvs", "AttributeError", "KeyError", "AssertionError",
                    "IndexError", "NotImplementedError", "HarnessRefusal")
WORKLOAD = dict(id="A", name="WP-17 TRIM replay (seeded_trim s512/s15)", control_hz=50, cameras="none",
                recording="none", robot_dof=0, contacts="handle + two measured finger pads (handle_and_measured_pads_v1)",
                material="seeded_trim tlmpm_v7_2 law 8, 1 mm particles / 4 mm grid / 0.1 ms / 200 substeps, K=2, S=8",
                reduced_fidelity=True, reduced_fidelity_reasons=["ideal_edge_tracking_no_robot_dynamics",
                                                                 "reaction_not_fed_back"])


class HarnessRefusal(RuntimeError):
    def __init__(self, code, detail):
        super().__init__(f"{code}: {detail}")
        self.code = code


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant-json", required=True)
    p.add_argument("--mode", choices=PA.MODES, required=True)
    p.add_argument("--num-envs", type=int, required=True)
    p.add_argument("--seeds", required=True, help="comma separated, e.g. 512,15")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpu-uuid")
    g.add_argument("--gpu-index", type=int)
    p.add_argument("--wp17", default=None, help="WP-17 record tree: DIR/<record>/seed_<s>/ (default artifacts/wp17)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    try:
        args.seed_list = [int(s) for s in str(args.seeds).split(",") if s.strip()]
        PA.plan_rounds(args.mode, args.seed_list, args.num_envs)
    except ValueError as error:
        p.error(f"--seeds/--num-envs: {error}")
    if args.warmup_steps < 0 or args.steps < 1 or args.warmup_steps + args.steps > PA.EPISODE_STEPS:
        p.error(f"need warmup_steps >= 0, steps >= 1, warmup_steps + steps <= {PA.EPISODE_STEPS}")
    return args


def git(*args):
    try:
        out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def environment(args):
    import robotic_cutting_sim
    package_root = Path(robotic_cutting_sim.__file__).resolve().parent
    return dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                checkout=str(ROOT), package_root=str(package_root),
                package_root_inside_checkout=str(package_root).startswith(str(ROOT) + os.sep),
                harness=str(Path(__file__).resolve().relative_to(ROOT)), harness_version=PA.HARNESS_VERSION,
                harness_sha256=PA.sha256_file(Path(__file__).resolve()),
                shared_module_sha256=PA.sha256_file(Path(PA.__file__).resolve()),
                python=sys.version.split()[0], interpreter=sys.executable, argv=sys.argv,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                warp_cache_path=os.environ.get("WARP_CACHE_PATH"), device=args.device, kit_started=False)


def wp17_record(root, seeds):
    """The ``<record>`` directory of ``root`` that holds every seed (``root/<record>/seed_<s>/``)."""
    root = Path(root)
    records = sorted(d for d in root.iterdir() if d.is_dir() and all((d / f"seed_{s}").is_dir() for s in seeds))
    if not records:
        raise FileNotFoundError(f"NO_WP17_RECORD: {root}: no record with seed_{seeds} (infrastructure)")
    preferred = [d for d in records if d.name == "native_optimized_001"]
    return (preferred or records)[0]


def write_json(path, data, exclusive=True):
    with Path(path).open("x" if exclusive else "w") as stream:
        json.dump(data, stream, indent=1, sort_keys=True, allow_nan=True,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def save_npz(out, name, arrays):
    path = out / "dumps" / name
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return dict(path=f"dumps/{name}", sha256=PA.sha256_file(path), bytes=path.stat().st_size)


# ----------------------------------------------------------------------------- GPU side

class Gpu:
    """Torch/Warp handles, the sync, the HashGrid census and the telemetry of the process."""

    def __init__(self, args):
        import torch
        import warp as wp
        self.torch, self.wp, self.args = torch, wp, args
        wp.init()
        self.device = args.device
        self.cuda = str(args.device).startswith("cuda")
        self.telemetry, self.gpu_index = None, physical_gpu_index(args)
        if self.cuda and self.gpu_index is not None:
            from robotic_cutting_sim.rl.telemetry import GpuTelemetry
            self.telemetry = GpuTelemetry(self.gpu_index, interval_s=1.)
            self.telemetry.set_phase("baseline")
            self.telemetry.start()
            time.sleep(2.)
            self.telemetry.set_phase("setup")

    def sync(self):
        if self.cuda:
            self.torch.cuda.synchronize()

    def live_device_hashgrids(self):
        import warnings
        gc.collect()
        with warnings.catch_warnings():   # isinstance() on deprecated torch module attributes warns
            warnings.simplefilter("ignore")
            return sum(1 for o in gc.get_objects() if isinstance(o, self.wp.HashGrid) and getattr(o, "id", None)
                       and getattr(getattr(o, "device", None), "is_cuda", False))

    def allocator_peak_mib(self):
        torch, wp = self.torch, self.wp
        if not self.cuda:
            return None
        high = wp.get_mempool_used_mem_high(self.device) if wp.is_mempool_enabled(self.device) else 0
        return (int(high) + int(torch.cuda.max_memory_reserved())) / 2 ** 20

    def gpu_identity(self):
        if not self.cuda:
            return dict(device="cpu")
        props = self.torch.cuda.get_device_properties(self.torch.cuda.current_device())
        return dict(name=props.name, cuda_ordinal=self.torch.cuda.current_device(), nvidia_smi_index=self.gpu_index,
                    uuid=self.args.gpu_uuid, total_mib=props.total_memory / 2 ** 20)


def physical_gpu_index(args):
    """nvidia-smi index of the GPU this process uses (``--gpu-index``, the ``--gpu-uuid``, or a single
    ``CUDA_VISIBLE_DEVICES`` entry); None when unknown (then no telemetry: ``UNKNOWN`` sharing)."""
    if args.gpu_index is not None:
        return int(args.gpu_index)
    uuid = args.gpu_uuid
    visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if uuid is None and visible and "," not in visible:
        if visible.isdigit():
            return int(visible)
        uuid = visible
    if uuid is None:
        return None
    try:
        text = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[1] == uuid and parts[0].isdigit():
            return int(parts[0])
    return None


class Marks:
    def __init__(self, gpu):
        self.gpu, self.global_step, self.rows = gpu, 0, []

    def __call__(self, event, num_envs, step=None):
        self.gpu.sync()
        if step is not None:
            self.global_step = step
        line = PA.format_mark(event, self.global_step, num_envs)
        print(line, flush=True)
        self.rows.append(dict(event=event, global_step=self.global_step, num_envs=num_envs,
                              t_perf_s=time.perf_counter() - T_SCRIPT))
        return time.perf_counter()


def run_round(args, gpu, marks, r, env_seeds, variant, out, RV, P3):
    torch, wp = gpu.torch, gpu.wp
    from robotic_cutting_sim.rl import open_vector_env
    from robotic_cutting_sim.rl.policies_batched import MeasuredTrajectoryDriver
    from robotic_cutting_sim.rl.replay import digit_counts
    N, mode = len(env_seeds), args.mode
    W, K = args.warmup_steps, args.steps
    order = sorted(set(env_seeds))
    live = gpu.live_device_hashgrids() if gpu.cuda else 0
    if live:
        raise HarnessRefusal("MULTIPLE_DEVICE_HASHGRIDS", f"{live} live device HashGrid(s) before building round {r}")
    inputs = RV.load_inputs(env_seeds)
    natives = {s: RV.native_final_codes(s) for s in order} if mode in ("episode", "samestate") else {}
    kwargs = dict(task=RV.family_for(env_seeds, inputs), device=args.device, strict=True,
                  **({"parity_mode": True} if not gpu.cuda else {}), **variant)
    if gpu.cuda:
        torch.cuda.reset_peak_memory_stats()
    res = dict(round=r, env_seeds=list(env_seeds), num_envs=N)
    dumps = []
    t_build = time.perf_counter()
    with open_vector_env(N, **kwargs) as env:
        res["build_wall_s"] = time.perf_counter() - t_build
        res["device_hashgrids_after_build"] = gpu.live_device_hashgrids() if gpu.cuda else 0
        res["payload_check"] = RV.check_payloads(env, inputs)
        pk = env.material.packed
        res["particles_per_env"] = [int(x) for x in pk.particle_counts]
        res["material_dt_s"] = float(pk.uniform["dt"])
        res["substeps"] = int(pk.uniform["substeps"])
        res["control_hz"] = 1. / float(env.control_dt)
        s = env.material.solver
        res["particle_contact_receipt"] = (s.particle_contact_receipt() if hasattr(s, "particle_contact_receipt")
                                           else None)
        res["receipt"] = env.receipt()
        driver = MeasuredTrajectoryDriver([inputs[x]["trajectory"] for x in order],
                                          [order.index(x) for x in env_seeds], device=env.torch_device)
        env.reset(seeds=[42] * N)    # untimed first reset (lazy first-call work); then the timed resets
        resets = []
        for _ in range(RESET_REPEATS[mode]):
            gpu.sync()
            t0 = time.perf_counter()
            env.reset(seeds=[42] * N)
            gpu.sync()
            resets.append((time.perf_counter() - t0) * 1e3)
        res["reset_ms"] = resets

        def step():
            pose, cut, pads = driver.inputs(env._step_count)
            env.step_measured_edge(pose, cut, pad_boxes=pads)

        if mode == "samestate":
            res["samestate"], arrays = samestate_stage(env, gpu, natives[env_seeds[0]][1], P3)
            res["reset_ms"] += timed_reset(env, gpu, N)
        else:
            arrays = {}
        monitor = None
        if mode == "episode":
            RV._grouped_query_audit = RV._define_kernels()
            monitor = P3.L2MonitorActiveGrid(env)
            monitor.audit(0)
        sample_steps = {x: np.rint(np.asarray(natives[x][1]["time_s"]) / PA.CONTROL_DT_S).astype(int)
                        for x in natives}
        samples = {e: [] for e in range(N)}
        contacts = np.zeros((W + K, N), np.int64) if mode in ("episode", "samestate") else None
        walls = np.zeros(W + K)
        start = marks.global_step
        if gpu.telemetry:
            gpu.telemetry.set_phase("run")
        marks("warm_begin", N)
        t_window = None
        profiling = False
        for t in range(W + K):
            if t == W:
                t_window = marks("warm_end", N, start + W)
                if mode == "profile":
                    torch.cuda.cudart().cudaProfilerStart()
                    profiling = True
            t0 = time.perf_counter()
            step()
            walls[t] = time.perf_counter() - t0
            if monitor is not None:
                monitor.update()
            if contacts is not None:
                contacts[t] = P3.stats_counts(env)
            if mode == "episode":
                n_step = t + 1
                for e, seed in enumerate(env_seeds):
                    hits = np.flatnonzero(sample_steps[seed] == n_step)
                    if hits.size:
                        k = int(hits[0])
                        snap = env.snapshot_world(e)
                        samples[e].append((n_step, k, np.asarray(snap["codes"]).copy(),
                                           PA.encode_positions(natives[seed][1]["positions"][k], snap["positions"])))
                if n_step % 300 == 0:
                    monitor.audit(n_step)
            marks.global_step = start + t + 1
        gpu.sync()
        t_end = time.perf_counter()
        if profiling:
            torch.cuda.cudart().cudaProfilerStop()
        marks("episode_end", N, start + W + K)
        if gpu.telemetry:
            gpu.telemetry.set_phase("post")
        res["timing"] = dict(window_steps=K, window_s=t_end - t_window, env_steps_per_s=K * N / (t_end - t_window),
                             step_wall_median_s=float(np.median(walls[W:])), step_wall_p95_s=float(
                                 np.percentile(walls[W:], 95)), first_step_s=float(walls[0]),
                             warmup_steps=W, scope="perf_counter from the warm_end mark (after a device synchronize) "
                             "to the device synchronize before the episode_end mark; env.step_measured_edge incl. "
                             "its strict int32 readback" + ("; L2 monitor, contact counts and Native samples inside"
                                                          if mode == "episode" else
                                                          "; contact counts inside" if mode == "samestate" else ""))
        res["allocator_peak_mib"] = gpu.allocator_peak_mib()
        res["invalid_reason"] = env.material.invalid_reason.numpy().tolist()
        res["guard_violations"] = P3.guard_violations(env)
        finals = [env.snapshot_world(e) for e in range(N)]
        res["final_step_count"] = [int(f["step_count"]) for f in finals]
        res["final_labelled"] = [digit_counts(f["codes"], 2) for f in finals]
        tag = f"r{r}"
        if mode in ("throughput", "profile"):
            dumps.append(dict(kind="final_codes", round=r, env_seeds=list(env_seeds), **save_npz(
                out, f"{tag}_final_codes.npz", {f"codes_e{e}": np.asarray(f["codes"], np.int32)
                                                for e, f in enumerate(finals)})))
        elif mode == "samestate":
            arrays.update({f"final_codes_e{e}": np.asarray(f["codes"], np.int32) for e, f in enumerate(finals)})
            arrays.update({f"final_positions_e{e}": np.asarray(f["positions"], np.float64)
                           for e, f in enumerate(finals)})
            arrays["contacts"] = contacts
            dumps.append(dict(kind="samestate", round=r, seed=int(env_seeds[0]), env_seeds=list(env_seeds),
                              samples=list(PA.SAMESTATE_SAMPLES), steps=W + K,
                              **save_npz(out, f"{tag}_samestate_s{env_seeds[0]}.npz", arrays)))
        else:
            res["l2_candidate_reported"] = monitor.result()
            ev2 = env.evaluate(list(range(N)))
            res["evaluate_2_frames_candidate_reported"] = RV.eval_summary(ev2)
            for e, seed in enumerate(env_seeds):
                f = finals[e]
                rows = samples[e]
                enc = [x[3] for x in rows]
                arrays = dict(
                    seed=np.int64(seed), env=np.int64(e), steps=np.int64(W + K),
                    sample_step=np.asarray([x[0] for x in rows], np.int64),
                    sample_index=np.asarray([x[1] for x in rows], np.int64),
                    sample_codes=(np.stack([x[2] for x in rows]).astype(np.int32) if rows
                                  else np.zeros((0, len(f["codes"])), np.int32)),
                    sample_dq=(np.stack([x["dq"] for x in enc]) if rows
                               else np.zeros((0, len(f["codes"]), 3), np.int8)),
                    sample_escape_index=np.concatenate(
                        [x["escape_index"] + i * x["dq"].size for i, x in enumerate(enc)]) if rows
                    else np.zeros(0, np.int64),
                    sample_escape_delta=np.concatenate([x["escape_delta"] for x in enc]) if rows
                    else np.zeros(0, np.float64),
                    final_positions=np.asarray(f["positions"], np.float64),
                    final_velocities=np.asarray(f["velocities"], np.float32),
                    final_codes=np.asarray(f["codes"], np.int32), final_step_count=np.int64(f["step_count"]),
                    contacts=contacts[:, e].copy(), invalid_reason=np.int64(res["invalid_reason"][e]))
                dumps.append(dict(kind="episode", round=r, env=e, seed=int(seed), samples=len(rows),
                                  **save_npz(out, f"{tag}_e{e}_s{seed}.npz", arrays)))
        res["dumps"] = dumps
        if mode == "profile":
            res["profile_step_walls_s"] = walls[W:].tolist()
    return res


def timed_reset(env, gpu, N):
    gpu.sync()
    t0 = time.perf_counter()
    env.reset(seeds=[42] * N)
    gpu.sync()
    return [(time.perf_counter() - t0) * 1e3]


def samestate_stage(env, gpu, native, P3):
    """Recorded Native states written into every env; the contact stage alone twice (``run_p1_3a`` helpers)."""
    N = env.num_envs
    out, arrays = dict(samples=list(PA.SAMESTATE_SAMPLES), rerun={}), {}
    for k in PA.SAMESTATE_SAMPLES:
        env.reset(seeds=[42] * N)
        P3.load_state(env, native, k)
        gpu.sync()
        stages = []
        for _ in range(2):
            with env._stream():
                P3.contact_stage(env)
            gpu.sync()
            stages.append(P3.read_stage(env))
        out["rerun"][str(k)] = P3.compare_stage(stages[0], stages[1])
        for tag, st in zip("ab", stages):
            arrays[f"impulse_{tag}_k{k}"] = np.asarray(st["impulse"], np.float32)
            arrays[f"stats_sum_{tag}_k{k}"] = np.asarray(st["stats_sum"])
            arrays[f"stats_max_{tag}_k{k}"] = np.asarray(st["stats_max"])
    return out, arrays


def row_for(args, res, gpu_info, sharing, telemetry_run, own_vram_mib):
    """The plan-column row of one round (``benchmark.MAIN_COLUMNS`` + extras)."""
    from robotic_cutting_sim.rl.benchmark import co_tenant_stats
    t = res["timing"]
    particles = res["particles_per_env"]
    vram, source = (own_vram_mib, "nvidia_smi_process") if own_vram_mib else (res["allocator_peak_mib"],
                                                                               "allocator_peak")
    return dict(num_envs=res["num_envs"], particles_per_env=int(round(float(np.mean(particles)))),
                particles_per_env_list=particles, control_hz=res["control_hz"], material_dt_s=res["material_dt_s"],
                cameras="none", env_steps_per_s=t["env_steps_per_s"],
                aggregate_sim_time_per_wall=t["env_steps_per_s"] / res["control_hz"],
                vram_peak_mib=vram, vram_source=source, allocator_peak_mib=res["allocator_peak_mib"],
                reset_p95_ms=PA.percentile(res["reset_ms"], 95), reset_mean_ms=float(np.mean(res["reset_ms"])),
                reset_count=len(res["reset_ms"]), physics_verdict="REPORTED_NOT_SCORED",
                physics_note="the judge recomputes physics from the dumps; in-process values are candidate-reported",
                substeps=res["substeps"], robot_dof=0, contacts=WORKLOAD["contacts"], recording="none",
                warm="warm" if args.warmup_steps >= 1 else "cold", warmup_steps=args.warmup_steps,
                timed_steps=args.steps, mode=args.mode, seeds=res["env_seeds"], round=res["round"],
                gpu=gpu_info, gpu_sharing=sharing, co_tenant=co_tenant_stats(telemetry_run),
                workload_id="A", boundary="batched_headless", reduced_fidelity=True)


def run(args, out, variant):
    gpu = Gpu(args)
    sys.path.insert(0, str(SCRIPTS))
    import run_vector_milestones as RV  # noqa: E402  (helpers; its OUT_ROOT is never used)
    import run_p1_3a as P3  # noqa: E402  (helpers; its OUT_ROOT is never used)
    wp17_root = Path(args.wp17) if args.wp17 else ROOT / "artifacts" / "wp17"
    RV.WP17 = wp17_record(wp17_root, args.seed_list)
    import torch
    import warp
    versions = dict(warp=warp.__version__, torch=torch.__version__, numpy=np.__version__)
    marks = Marks(gpu)
    rounds = []
    for r, env_seeds in enumerate(PA.plan_rounds(args.mode, args.seed_list, args.num_envs)):
        rounds.append(run_round(args, gpu, marks, r, env_seeds, variant, out, RV, P3))
        gc.collect()   # the round's batch (and its grid) is unreachable once run_round returned
        if gpu.cuda:
            gpu.wp.synchronize_device(gpu.device)
        rounds[-1]["device_hashgrids_after_release"] = gpu.live_device_hashgrids() if gpu.cuda else 0
    summary = run_phase = None
    samples = []
    if gpu.telemetry:
        summary = gpu.telemetry.stop()
        run_phase = gpu.telemetry.summary(phases=("run",))
        samples = gpu.telemetry.samples
    with (out / "telemetry_samples.jsonl").open("x") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, default=str) + "\n")
    gpu_info = dict(gpu.gpu_identity(), **({k: summary.get(k) for k in ("gpu_index", "gpu_name", "gpu_driver",
                                                                         "gpu_uuid", "memory_total_mib")}
                                           if summary and summary.get("valid_samples") else {}))
    sharing = (run_phase or {}).get("gpu_sharing") or "UNKNOWN"
    own = (summary or {}).get("own_vram_peak_mib") or None
    rows = [row_for(args, res, gpu_info, sharing, run_phase, own) for res in rounds]
    return dict(rounds=rounds, rows=rows[-1:], round_rows=rows, marks=marks.rows, versions=versions,
                gpu=gpu_info, gpu_sharing=sharing, telemetry=summary, telemetry_run_phase=run_phase,
                wp17_record=str(RV.WP17), dumps=[d for res in rounds for d in res.pop("dumps")],
                hashgrids_after=gpu.live_device_hashgrids() if gpu.cuda else 0)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        print(f"run_perf_acceptance: --out {out} exists (never overwritten)", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    (out / "dumps").mkdir()
    base = dict(schema=PA.SCHEMA_RESULT, harness=PA.HARNESS_VERSION, mode=args.mode, num_envs=args.num_envs,
                seeds=args.seed_list, warmup_steps=args.warmup_steps, steps=args.steps,
                variant_json=str(args.variant_json), workload=WORKLOAD,
                timing_note="in-process timing; the caller's outer clock on the marks is the reference")
    status, code, error, data = "ERROR", None, None, {}
    try:
        base["environment"] = environment(args)
        raw = Path(args.variant_json).read_text()
        base["variant_sha256"] = PA.sha256_bytes(raw.encode())
        variant = json.loads(raw)
        base["variant"] = variant
        variant = PA.check_variant(variant)
        data = run(args, out, variant)
        status = "COMPLETE"
    except Exception as exc:   # preserved and classified, never hidden
        error = traceback.format_exc()
        print(error, file=sys.stderr)
        code = getattr(exc, "code", None) or type(exc).__name__
        if not any(c.__name__ in CANDIDATE_ERRORS for c in type(exc).__mro__) or "environment" not in base:
            write_json(out / "error.json", dict(base, status="INFRA_ERROR", error=error, error_code=code,
                                               wall_s=time.perf_counter() - T_SCRIPT))
            return 1
    result = dict(base, status=status, error=error, error_code=code, wall_s=time.perf_counter() - T_SCRIPT, **data)
    write_json(out / "result.json", result)
    print(json.dumps(dict(result=str(out / "result.json"), status=status, error_code=code,
                          gpu_sharing=data.get("gpu_sharing"), wall_s=result["wall_s"])), flush=True)
    return 0 if status == "COMPLETE" else 3


if __name__ == "__main__":
    sys.exit(main())
