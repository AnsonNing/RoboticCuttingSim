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


WP17_ATTEMPT = "artifacts/wp17/native_optimized_001/seed_{seed}/box/env/recording/attempt_000"


class RecordedDriver:
    """``recorded_command_policy_v1`` (``rl.policies_batched``, imported unchanged): the COMMANDED edge targets of
    the WP-17 Native episode of each env's scene seed, converted to ``edge_delta_v1`` actions open loop from the
    env's own initial commanded pose (``env.home_commanded``), replayed by the env's episode step."""

    def __init__(self, env, seeds, device):
        import torch
        from robotic_cutting_sim.rl.policies_batched import (RecordedCommandPolicyV1, load_recorded_commands,
                                                              recorded_command_actions)
        home = env.home_commanded.cpu().numpy()
        tables, targets, cuts, self.reports = [], [], [], []
        for e, (seed, task) in enumerate(zip(seeds, env.tasks)):
            rec = load_recorded_commands(WP17_ATTEMPT.format(seed=seed))
            if rec["config_sha256"] != task.config_sha256:
                raise ValueError(f"WP-17 seed {seed} was recorded for another task config")
            actions, _, report = recorded_command_actions(rec["targets"], rec["cut_enabled"], rec["cut_index"],
                                                          home[e], task.action_limits(), task.T_world_workpiece,
                                                          task.max_cuts)
            tables.append(actions)
            targets.append(rec["targets"])
            cuts.append(rec["cut_enabled"])
            self.reports.append(dict(env=e, scene_seed=seed, actions_sha256=rec["actions_sha256"],
                                     reset_edge_native=rec["reset_edge"].tolist(), initial_commanded=home[e].tolist(),
                                     **report))
        self.policy = RecordedCommandPolicyV1(tables, np.arange(len(tables)), device=device)
        self.target = torch.as_tensor(np.stack(targets), dtype=torch.float64, device=device)   # [N, T, 7]
        self.cut = torch.as_tensor(np.stack(cuts), device=device)
        self.env, self.T = env, self.policy.steps

    def __call__(self):
        import torch
        step = self.env.episode_length_buf.to(torch.int64)
        action = self.policy(None, {"step_count": step})
        k = torch.clamp(step, max=self.T-1)
        n = torch.arange(len(k), device=k.device)
        return action, self.cut[n, k] & (step < self.T)

    def current_target(self):
        import torch
        k = torch.clamp(self.env.episode_length_buf.to(torch.int64), max=self.T-1)
        return self.target[torch.arange(len(k), device=k.device), k]


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
    parser.add_argument("--probe", required=True, choices=("g0", "t2", "g1", "g2", "g3", "compare"))
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--tool-mount", default="rigid_mount_v1")
    parser.add_argument("--gpu-index", type=int, default=0, help="physical nvidia-smi index (telemetry)")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--reset-every", type=int, default=0)
    parser.add_argument("--variant", default="base", choices=("base", "perturb", "reset"))
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--material", default=None, help="tlmpm_v7_2: BatchedCuttingMaterial(isaac_lab_newton)")
    parser.add_argument("--driver", default="scripted", choices=("scripted", "reference", "recorded", "hold"))
    parser.add_argument("--scene-seed", type=int, default=512)
    parser.add_argument("--scene-seeds", type=int, nargs="*", default=None, help="per-env scene seeds (cycled)")
    parser.add_argument("--evaluate", action="store_true", help="frozen evaluator per env at the end")
    parser.add_argument("--record-inputs", action="store_true", help="save the material input sequence (G5)")
    parser.add_argument("--restore-env", type=int, default=None, help="g3: env restored at --restore-step")
    parser.add_argument("--restore-step", type=int, default=10)
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
    seeds = [args.scene_seed] if not args.scene_seeds else list(args.scene_seeds)
    seeds = [seeds[e % len(seeds)] for e in range(args.num_envs)]
    compiled = {seed: _task(seed) for seed in dict.fromkeys(seeds)}
    task = compiled[seeds[0]]
    env_tasks = [compiled[seed] for seed in seeds]
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
        env = IsaacLabBatchedCuttingEnv(cfg, options, task, tasks=env_tasks)
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
        row["scene_seeds"] = seeds
        driver = (ReferenceDriver(task, env, device) if args.driver == "reference" else
                  RecordedDriver(env, seeds, device) if args.driver == "recorded" else None)
        if args.driver == "recorded":
            row["recorded_policy"] = {"policy": "recorded_command_policy_v1", "per_env": driver.reports}
        inputs = {"edge": [], "cut": [], "box_centers": [], "box_halves": []}
        grasp_rows = {"slip": [], "angle": [], "status": [], "bilateral": []}
        wrench_max = torch.zeros(args.num_envs, dtype=torch.float64, device=device)
        trace = {"target": [], "edge": [], "commanded": [], "status": [], "force": []}
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
                if args.probe == "g3" and step == args.restore_step and args.restore_env is not None:
                    mask = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
                    mask[args.restore_env] = True
                    env.reset_envs(mask)
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
                    trace["target"].append(driver.current_target().clone() if args.driver == "recorded" else
                                           driver.target[torch.clamp(env.episode_length_buf.to(torch.long),
                                                                     max=driver.T-1)].clone())
                elif args.driver == "hold":
                    action = torch.zeros((args.num_envs, 8), device=device)
                    action[:, 6] = -1.
                else:
                    action = _actions(args.num_envs, step, device, perturb)
                env.step(action)
                if args.material:
                    wrench_max = torch.maximum(wrench_max, wp.to_torch(env.coupling.pending_wrench)[:, :3].norm(dim=1))
                if args.record_inputs:
                    inputs["edge"].append(wp.to_torch(env.coupling.measured_pose).clone())
                    inputs["cut"].append(wp.to_torch(env.coupling.cut_input).clone())
                    inputs["box_centers"].append(wp.to_torch(env.coupling.box_centers).clone())
                    inputs["box_halves"].append(wp.to_torch(env.coupling.box_halves).clone())
                if args.tool_mount == "physical_grasp_v1":
                    grasp_rows["slip"].append(wp.to_torch(env.grasp_slip).clone())
                    grasp_rows["angle"].append(wp.to_torch(env.grasp_angle).clone())
                    grasp_rows["status"].append(wp.to_torch(env.grasp_status).clone())
                    ci = wp.to_torch(env.contact_integral)
                    grasp_rows["bilateral"].append(ci[:, 8]/torch.clamp(ci[:, 9], min=1.))
                if driver is not None:
                    trace["edge"].append(wp.to_torch(env.frames.measured_edge).clone())
                    trace["commanded"].append(env.commanded.clone())
                    trace["status"].append(env.action_status.clone())
                    if args.material:
                        trace["force"].append(wp.to_torch(env.coupling.pending_wrench).clone())
                torch.cuda.synchronize()
                walls.append(time.perf_counter()-t0)
                if args.probe == "g0":
                    impulse = env.frames.applied_impulse.numpy()[:, 2]
                    contacts = env.contact_integral.numpy()[:, 9] if args.tool_mount == "physical_grasp_v1" else None
                    graph_checks.append({"impulse_z": impulse.tolist(),
                                         "contact_substeps": None if contacts is None else contacts.tolist()})
                if args.probe in ("g0", "g1", "g3"):
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
        if args.evaluate and args.material:
            telemetry.set_phase("evaluate")
            evaluations = env.evaluate()
            row["evaluation"] = [{k: v for k, v in r.items() if k != "result"} for r in evaluations]
            final_codes = {f"env{e}": env.snapshot_world(e)["codes"] for e in range(args.num_envs)}
            pk = env.coupling.packed
            x_all = env.coupling.solver.x.numpy()
            final_codes.update({f"x_local_env{e}": x_all[int(pk.particle_offset[e]):int(pk.particle_offset[e+1])]
                                for e in range(args.num_envs)})
            for r in row["evaluation"]:
                codes = final_codes[f"env{r['env']}"]
                r["labelled_by_digit"] = {str(d): int(np.count_nonzero((codes >> (2*d)) & 3))
                                          for d in range(int(task.max_cuts))}
            np.savez_compressed(out/"final_codes.npz", **final_codes)
        if args.material:
            row["termination_reason"] = env.termination_reason.cpu().tolist()
            row["reward_fractions"] = {k: v.cpu().tolist() for k, v in env.compute_task_reward().items()}
        if inputs["edge"]:
            np.savez_compressed(out/"material_inputs.npz", **{k: torch.stack(v).cpu().numpy() for k, v in inputs.items()},
                                initial_edge=env.material_initial_edge, initial_boxes=env.material_initial_boxes,
                                scene_seeds=np.asarray(seeds), seed=int(options.seed))
        if grasp_rows["slip"]:
            g = {k: torch.stack(v).cpu().numpy() for k, v in grasp_rows.items()}
            np.savez_compressed(out/"grasp_trace.npz", **g)
            row["grasp_trace_summary"] = {"max_slip_m": g["slip"].max(0).tolist(), "max_angle_rad": g["angle"].max(0).tolist(),
                                          "final_status": g["status"][-1].tolist(),
                                          "min_bilateral_fraction": g["bilateral"].min(0).tolist()}
            if env.snapshot is not None:
                row["grasp_snapshot"] = env.snapshot.receipt()
                row["grasp_snapshot"]["relative_R"] = env.snapshot.relative_R.numpy().tolist()
                row["grasp_snapshot"]["relative_p"] = env.snapshot.relative_p.numpy().tolist()
                row["grasp_snapshot"]["accumulators"] = (env.grasp_initialization or {}).get("accumulators")
        if trace["edge"]:
            np.savez_compressed(out/"trace.npz", **{k: torch.stack(v).cpu().numpy() for k, v in trace.items() if v})
        if states["joint_q"]:
            np.savez_compressed(out/"states.npz", **{k: np.asarray(v) for k, v in states.items()},
                                body_world=env.model.body_world.numpy(), coord_start=env.coord_start.numpy(),
                                hand=env.tables["hand"], tool=env.tables["tool"])
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
