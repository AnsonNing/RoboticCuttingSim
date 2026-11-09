"""Workload B stage S1 GPU probes (ADR 0022 §9), robot-only: G0 smoke, T2 host syncs, G1 isolation, G2 timing.

RCS GPU rules (decision D-07): one job per GPU under the flock, explicit device, >= 4 GB left free, rows labelled
SHARED_GPU / EXCLUSIVE_GPU from sampled co-tenant use (``robotic_cutting_sim.rl.telemetry``). Example (physical GPU 0)::

    CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src \
        flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock \
        .deps/isaaclab-venv/bin/python scripts/rl_ready/probe_lab_batched_gpu.py --probe g0 --num-envs 1 \
        --gpu-index 0 --out artifacts/rl_ready/p1_1b/g0_b1_n1_001

One Isaac Lab build per process; the output directory must not exist. Probes:

* ``g0``  N worlds: cold build/reset/first step (graph capture inside), then ``--steps`` warm steps with a constant
          0.5 N external edge force in every env. Graph safety: the wrench kernel (state-force callback) must have run
          in all 10 substeps of every step (``applied_impulse_z == 10 * 2 ms * 0.5 N``) and, in B2, the contact
          callback in all 10 (``contact_integral[:, 9] == 10``). Per-step states are saved; ``--no-graph`` runs the
          same steps eagerly for the graph-vs-eager comparison (``--probe compare``).
* ``t2``  ``--steps`` warm steps under ``torch.cuda.set_sync_debug_mode("warn")`` with ``wp.array.numpy`` and
          ``wp.synchronize*`` counted by call site. PASS: the only syncs are the strict-mode validity read
          (one int32 per step, ``check_valid``).
* ``g1``  N=4, ``--deterministic`` (Newton ``run_to_run``), ``--variant base|perturb|reset`` (env 2 perturbed;
          env 1 reset at step 30); per-step states saved for ``--probe compare``.
* ``g2``  ``--steps`` steps with a staggered masked reset every ``--reset-every`` steps; strict mode (one int32
          read per step, so each step ends synchronized); env-steps/s, aggregate sim-time/wall, reset p95, own VRAM
          peak and slope.
* ``compare``  bitwise / max-difference comparison of saved ``states.npz`` runs (no GPU work).

Robot-only: no material (``robot_only_no_material_coupling``); B1 also ``rigid_tool_mount_no_grasp``. Every row
records N, particles/env 0, control 50 Hz, 10 x 2 ms rigid substeps, no material dt, cameras none, recording none,
warm/cold, GPU name/index/driver, sharing label and commit.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np

MIN_FREE_MB = 4096
JOB_ESTIMATE_MB = 2560
EXTERNAL_FORCE_N = .5


def _gpu_row(index):
    out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=name,driver_version,memory.total,memory.used,"
                          "utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    name, driver, total, used, util = [x.strip() for x in out.stdout.strip().split(",")]
    return {"name": name, "index": int(index), "driver": driver, "memory_total_mb": float(total),
            "memory_used_mb": float(used), "utilization_pct": float(util)}


def _commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "scripts"],
                               capture_output=True, text=True).stdout.strip()
        import hashlib
        sources = sorted(Path("src/robotic_cutting_sim/batched").glob("*.py"))+[
            Path("src/robotic_cutting_sim/rl/lab_batched_env.py"), Path(__file__)]
        digest = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
        return {"sha": sha, "tracked_src_or_scripts_modified": bool(dirty), "workload_b_sources_sha256": digest}
    except Exception:
        return None


def _task(seed):
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.seeded_trim import generate_scene, goal_config
    return compile_task(generate_scene(seed), goal_config(), seed=42)


def _actions(n, step, device, perturb=None):
    import torch
    a = torch.zeros((n, 8), device=device)
    a[:, 0] = float(.6*np.sin(.35*step))
    a[:, 1] = float(.4*np.cos(.245*step))
    a[:, 2] = -.5 if step < 6 else .3
    a[:, 5] = float(.2*np.sin(.175*step))
    a[:, 6] = 1.
    if perturb is not None:
        a[perturb, :3] = torch.tensor([-1., 1., -1.], device=device)
    return a


class ReferenceDriver:
    """Scripted ``edge_delta_v1`` controller tracking the task's frozen reference (``task.reference_action``), on the
    device: per env, the target is the reference at the env's own episode step + 1 (restarts after a reset); the
    action is the clipped edge-frame delta from the commanded pose, armed with digit 0 where the reference cuts.
    A driver for timing/engagement rows, not a policy and not the WP-17 controller."""

    def __init__(self, task, env, device):
        import torch
        T = int(round(float(task.duration_s)/float(task.control_dt_s)))
        rows = [task.reference_action((k+1)*float(task.control_dt_s)) for k in range(T)]
        self.target = torch.tensor(np.stack([r["target"] for r in rows]), dtype=torch.float64, device=device)
        self.cut = torch.tensor([bool(r.get("cut_enabled", False)) for r in rows], device=device)
        K = int(task.max_cuts)
        self.selector = torch.tensor([(int(r.get("cut_index", 0))+.5)/K*2.-1. for r in rows], dtype=torch.float64,
                                     device=device)
        limits = task.action_limits()
        self.max_t, self.max_r = float(limits.max_translation_m), float(limits.max_rotation_rad)
        self.env, self.T = env, T

    def __call__(self):
        import torch
        from robotic_cutting_sim.rl.actions import _as_matrix, _as_rotvec, _compose, _inv, _normalize
        env = self.env
        k = torch.clamp(env.episode_length_buf.to(torch.long), max=self.T-1)
        target, cut = self.target[k], self.cut[k]
        p, q = env.commanded[:, :3], _normalize(env.commanded[:, 3:])
        local = torch.einsum("nji,nj->ni", _as_matrix(q), target[:, :3]-p)
        rot = _as_rotvec(_compose(_inv(q), _normalize(target[:, 3:])))
        a = torch.zeros((len(k), 8), dtype=torch.float64, device=k.device)
        a[:, :3] = torch.clamp(local/self.max_t, -1., 1.)
        a[:, 3:6] = torch.clamp(rot/self.max_r, -1., 1.)
        a[:, 6] = torch.where(cut, torch.ones_like(a[:, 6]), -torch.ones_like(a[:, 6]))
        a[:, 7] = self.selector[k]
        return a.to(torch.float32), cut


def _site(stack):
    """First frame outside warp/torch/stdlib: where a sync was requested."""
    for frame in reversed(stack[:-1]):
        name = frame.filename
        if "/warp/" in name or "/torch/" in name or "/python3.12/" in name and "site-packages" not in name:
            continue
        return f"{Path(name).name}:{frame.lineno}:{frame.name}"
    return "unknown"


def compare(args):
    runs = {p: np.load(Path(p)/"states.npz") for p in args.inputs}
    ref_path = args.inputs[0]
    ref = runs[ref_path]
    cs, bw = ref["coord_start"], ref["body_world"]
    N = len(cs)-1
    out = {"reference": ref_path, "runs": {}}
    for path, d in runs.items():
        if path == ref_path:
            continue
        steps = min(len(d["joint_q"]), len(ref["joint_q"]))
        per = {}
        for e in range(N):
            q = d["joint_q"][:steps, cs[e]:cs[e+1]]
            q0 = ref["joint_q"][:steps, cs[e]:cs[e+1]]
            b = d["body_q"][:steps][:, bw == e]
            b0 = ref["body_q"][:steps][:, bw == e]
            per[e] = {"bitwise": bool(np.array_equal(q, q0) and np.array_equal(b, b0)),
                      "max_abs_joint_q": float(np.abs(q-q0).max()), "max_abs_body_q": float(np.abs(b-b0).max()),
                      "first_differing_step": int(np.argmax(np.any(q != q0, 1))) if not np.array_equal(q, q0) else None}
        out["runs"][path] = {"steps": steps, "worlds": per}
    Path(args.out).mkdir(parents=True, exist_ok=False)
    (Path(args.out)/"compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", required=True, choices=("g0", "t2", "g1", "g2", "compare"))
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--tool-mount", default="rigid_mount_v1")
    parser.add_argument("--gpu-index", type=int, default=0, help="physical nvidia-smi index (telemetry)")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--reset-every", type=int, default=0)
    parser.add_argument("--variant", default="base", choices=("base", "perturb", "reset"))
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--material", default=None, help="tlmpm_v7_2: BatchedCuttingMaterial(isaac_lab_newton)")
    parser.add_argument("--driver", default="scripted", choices=("scripted", "reference"))
    parser.add_argument("--scene-seed", type=int, default=512)
    parser.add_argument("--inputs", nargs="*", default=())
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.probe == "compare":
        return compare(args)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        raise SystemExit("GPU probe: set CUDA_VISIBLE_DEVICES to the one GPU this job holds the flock for")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    gpu = _gpu_row(args.gpu_index)
    free = gpu["memory_total_mb"]-gpu["memory_used_mb"]
    if free-JOB_ESTIMATE_MB < MIN_FREE_MB:
        (out/"row.json").write_text(json.dumps({"status": "BLOCKED", "reason": f"{free:.0f} MB free", "gpu": gpu}))
        raise SystemExit(f"BLOCKED: GPU {args.gpu_index} has {free:.0f} MB free; D-07 needs >= {MIN_FREE_MB} MB left")
    from robotic_cutting_sim.rl.telemetry import GpuTelemetry
    telemetry = GpuTelemetry(args.gpu_index)
    telemetry.set_phase("baseline")
    for _ in range(3):
        telemetry.sample()
        time.sleep(.3)
    telemetry.set_phase("build")
    telemetry.start()
    t_start = time.perf_counter()
    import torch
    import warp as wp
    from isaaclab.app.sim_launcher import launch_simulation
    from robotic_cutting_sim.batched import lab_scene
    from robotic_cutting_sim.rl.lab_batched_env import IsaacLabBatchedCuttingEnv
    t_import = time.perf_counter()
    task = _task(args.scene_seed)
    options = lab_scene.BatchedLabOptions(num_envs=args.num_envs, tool_mount=args.tool_mount, device="cuda:0",
                                          deterministic_mode="run_to_run" if args.deterministic else "not_guaranteed",
                                          use_cuda_graph=not args.no_graph, strict=True, material=args.material)
    cfg = lab_scene.make_batched_lab_cfg(options, task)
    fidelity = options.fidelity()
    fidelity["reasons"] = [("robot_only_no_material" if r == "robot_only_no_material_coupling" else r)
                           for r in fidelity["reasons"]]
    row = {"probe": args.probe, "variant": args.variant, "N": args.num_envs, "tool_mount": args.tool_mount,
           "particles_per_env": 0, "control_hz": 50, "driver": args.driver, "rigid_substeps_per_control_step": 10, "rigid_dt_s": .002,
           "material_dt_s": 1e-4 if args.material else None,
           "material": args.material or "none (robot-only)", "cameras": "none", "recording": "none",
           "use_cuda_graph": not args.no_graph, "deterministic_mode": options.deterministic_mode,
           "strict_mode": True, "gpu": gpu, "commit": _commit(), "workload": "B (robot-only)", **fidelity,
           "boundary": "batched_headless (Isaac Lab Newton/MJWarp kit-less, one process)"}
    device = "cuda:0"
    with launch_simulation(cfg, {"headless": True, "visualizer": [], "device": device}):
        from isaaclab_newton.physics import NewtonManager
        env = IsaacLabBatchedCuttingEnv(cfg, options, task)
        torch.cuda.synchronize()
        t_build = time.perf_counter()
        env.reset()
        torch.cuda.synchronize()
        t_reset = time.perf_counter()
        if args.probe == "g0":
            env.frames.external_wrench.assign(np.tile([0., 0., EXTERNAL_FORCE_N, 0., 0., 0.], (args.num_envs, 1)))
        env.step(_actions(args.num_envs, 0, device))
        torch.cuda.synchronize()
        t_first = time.perf_counter()
        row["cold"] = {"imports_s": t_import-t_start, "build_s": t_build-t_import, "reset_s": t_reset-t_build,
                       "first_step_s": t_first-t_reset, "process_to_first_step_s": t_first-t_start,
                       "warp_kernel_cache": "user cache (not cleared)",
                       "grasp_settle_included_in_reset": args.tool_mount == "physical_grasp_v1"}
        row["graph_captured"] = NewtonManager._graph is not None
        row["receipt"] = env.receipt()
        if args.material:
            pk = env.coupling.packed
            row["particles_per_env"] = [int(x) for x in pk.particle_counts]
            row["material_substeps_per_control_step"] = int(env.coupling.solver.substeps)
            row["material_graph_recaptured_newton"] = env.material_graph_recaptured
        driver = ReferenceDriver(task, env, device) if args.driver == "reference" else None
        wrench_max = torch.zeros(args.num_envs, dtype=torch.float64, device=device)
        armed_steps = torch.zeros(args.num_envs, dtype=torch.int64, device=device)
        if env.grasp_initialization:
            row["grasp_initialization"] = {k: v for k, v in env.grasp_initialization.items() if k != "accumulators"}
        telemetry.set_phase("warm")
        states = {"joint_q": [], "body_q": [], "edge": [], "status": []}
        walls, reset_walls, graph_checks = [], [], []

        def record():
            state = NewtonManager.get_state_0()
            states["joint_q"].append(state.joint_q.numpy().copy())
            states["body_q"].append(state.body_q.numpy().copy())
            states["edge"].append(env.frames.measured_edge.numpy().copy())
            states["status"].append(env.action_status.cpu().numpy().copy())

        if args.probe == "t2":
            counts = collections.Counter()
            originals = {"numpy": wp.array.numpy}

            def counted_numpy(self):
                counts["wp.array.numpy@"+_site(traceback.extract_stack())] += 1
                return originals["numpy"](self)
            wp.array.numpy = counted_numpy
            for name in ("synchronize", "synchronize_device", "synchronize_stream"):
                originals[name] = getattr(wp, name)

                def counted(*a, _name=name, **k):
                    counts[f"wp.{_name}@"+_site(traceback.extract_stack())] += 1
                    return originals[_name](*a, **k)
                setattr(wp, name, counted)
            import warnings
            torch.cuda.set_sync_debug_mode("warn")
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    t0 = time.perf_counter()
                    for step in range(1, args.steps+1):
                        env.step(_actions(args.num_envs, step, device))
                    wall = time.perf_counter()-t0
            finally:
                torch.cuda.set_sync_debug_mode(0)
                wp.array.numpy = originals.pop("numpy")
                for name, fn in originals.items():
                    setattr(wp, name, fn)
            torch_syncs = collections.Counter()
            for w in caught:
                if "synchroniz" in str(w.message):
                    torch_syncs[f"{Path(w.filename).name}:{w.lineno}"] += 1
            allowed = {k: v for k, v in counts.items() if "check_valid" in k}
            other = {k: v for k, v in counts.items() if "check_valid" not in k}
            row["t2"] = {"steps": args.steps, "warp_reads_and_syncs": dict(counts), "torch_syncs": dict(torch_syncs),
                         "strict_reads": sum(allowed.values()), "expected_strict_reads": args.steps,
                         "other_syncs": {**other, **{"torch@"+k: v for k, v in torch_syncs.items()}},
                         "wall_s": wall,
                         "status": "PASS" if not other and not torch_syncs and sum(allowed.values()) == args.steps
                         else "FAIL"}
        else:
            for step in range(1, args.steps+1):
                if args.probe == "g1" and args.variant == "reset" and step == 30:
                    mask = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
                    mask[1] = True
                    env.reset_envs(mask)
                if args.reset_every and step % args.reset_every == 0:
                    mask = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
                    mask[(step//args.reset_every) % args.num_envs] = True
                    t0 = time.perf_counter()
                    env.reset_envs(mask)
                    torch.cuda.synchronize()
                    reset_walls.append(time.perf_counter()-t0)
                perturb = 2 if (args.probe == "g1" and args.variant == "perturb") else None
                t0 = time.perf_counter()
                if driver is not None:
                    action, cut = driver()
                    armed_steps += cut.to(torch.int64)
                else:
                    action = _actions(args.num_envs, step, device, perturb)
                env.step(action)
                if args.material:
                    wrench_max = torch.maximum(wrench_max, wp.to_torch(env.coupling.pending_wrench)[:, :3].norm(dim=1))
                torch.cuda.synchronize()
                walls.append(time.perf_counter()-t0)
                if args.probe == "g0":
                    impulse = env.frames.applied_impulse.numpy()[:, 2]
                    contacts = env.contact_integral.numpy()[:, 9] if args.tool_mount == "physical_grasp_v1" else None
                    graph_checks.append({"impulse_z": impulse.tolist(),
                                         "contact_substeps": None if contacts is None else contacts.tolist()})
                if args.probe in ("g0", "g1"):
                    record()
            total = float(np.sum(walls))
            row["warm"] = {"steps": args.steps, "wall_s_total": total, "step_wall_s_mean": float(np.mean(walls)),
                           "step_wall_s_p50": float(np.percentile(walls, 50)),
                           "step_wall_s_p95": float(np.percentile(walls, 95)),
                           "env_steps_per_s": args.num_envs*args.steps/total,
                           "aggregate_sim_time_per_wall": args.num_envs*args.steps*.02/total,
                           "resets": len(reset_walls),
                           "reset_p95_s": float(np.percentile(reset_walls, 95)) if reset_walls else None,
                           "torch_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "timing": "per-step wall with a device sync after each step (strict mode already reads "
                                     "one int32 per step)"}
            if args.material or driver is not None:
                row["engagement"] = {"max_material_force_n": wrench_max.cpu().tolist(),
                                     "reference_cut_steps": armed_steps.cpu().tolist(),
                                     "material_invalid_reason": env.coupling.invalid_reason.numpy().tolist()
                                     if args.material else None,
                                     "material_receipt": env.coupling.receipt() if args.material else None}
            if graph_checks:
                expected = 10*.002*EXTERNAL_FORCE_N
                worst = max(abs(v-expected) for c in graph_checks for v in c["impulse_z"])
                contact_ok = all(c["contact_substeps"] is None or all(x == 10 for x in c["contact_substeps"])
                                 for c in graph_checks)
                row["g0_graph_safety"] = {"expected_impulse_z_per_step": expected, "max_abs_deviation": worst,
                                          "wrench_kernel_every_substep": worst < 1e-6, "contact_callback_every_substep":
                                          contact_ok, "per_step": graph_checks[:5],
                                          "status": "PASS" if worst < 1e-6 and contact_ok and
                                          (row["graph_captured"] or args.no_graph) else "FAIL"}
        if states["joint_q"]:
            np.savez_compressed(out/"states.npz", **{k: np.asarray(v) for k, v in states.items()},
                                body_world=env.model.body_world.numpy(), coord_start=env.coord_start.numpy())
    summary = telemetry.stop()
    warm = [s for s in telemetry.samples if s.get("phase") == "warm" and s.get("ok")]
    slope = None
    if len(warm) >= 3:
        t = np.asarray([s["t"] for s in warm])
        m = np.asarray([s.get("own_memory_mib") or 0. for s in warm])
        slope = float(np.polyfit(t-t[0], m, 1)[0])
    row["telemetry"] = summary
    row["vram_own_peak_mib"] = summary.get("own_vram_peak_mib")
    row["vram_own_slope_mib_per_s_warm"] = slope
    row["gpu_sharing"] = summary.get("gpu_sharing")
    row["wall_s_total_process"] = time.perf_counter()-t_start
    (out/"row.json").write_text(json.dumps(row, indent=1, default=str))
    print(json.dumps({k: row.get(k) for k in ("probe", "variant", "N", "tool_mount", "graph_captured", "cold", "warm",
                                              "t2", "g0_graph_safety", "gpu_sharing", "vram_own_peak_mib")},
                     default=str, indent=1))


if __name__ == "__main__":
    main()
