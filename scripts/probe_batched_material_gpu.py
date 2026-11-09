"""GPU smoke of the batched v7_2 material (ADR 0021 §5 L2, P1-1 step 5).

Builds ``BatchedCuttingMaterial`` for N copies of a full-resolution TRIM scene (seeded_trim, 1 mm particles,
4 mm grid, 0.1 ms material dt, 200 substeps per 20 ms control step) on one CUDA device with real graph capture,
drives a few control steps of a scripted edge motion into the first protrusion with the device pose
interpolation (``pose_interp``), and checks the L2 structural invariants per env after every control step:
mass bitwise constant, codes monotone per digit (no rewrites), finite state, positive minimum det F,
slot_overflow 0, §4 range audits and an instrumented grouped hash query (every neighbour in the querying env).
A frozen single ``LiveCuttingMaterial`` (GPU, its own graph) is driven with the same poses for a descriptive
comparison only (GPU atomics are not bitwise reproducible; this is not L3).

Usage (shared GPU; see RL_READY_PLAN D-07):
  flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock env CUDA_VISIBLE_DEVICES=1 \
      TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .deps/isaacsim-venv/bin/python \
      scripts/probe_batched_material_gpu.py --out artifacts/rl_ready/p1_1/gpu_smoke_001 --envs 1 4 --steps 5
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

SPACING = .001
SEED = 512
START_X = .0365   # 2 mm before the seed-512 protrusion at the reference cutting height (t = 4 s)
DX = .001


@wp.kernel
def _grouped_query_audit(grid: wp.uint64, x: wp.array(dtype=wp.vec3), env_of_particle: wp.array(dtype=int),
                         radius: float, foreign: wp.array(dtype=int), visited: wp.array(dtype=int)):
    p = wp.tid()
    e = env_of_particle[p]
    query = wp.hash_grid_query(grid, x[p], radius, e)
    for q in query:
        wp.atomic_add(visited, 0, 1)
        if env_of_particle[q] != e:
            wp.atomic_add(foreign, 0, 1)


def gpu_snapshot():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total,"
                              "utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             check=True).stdout.strip().splitlines()
        rows = [dict(zip(("index", "name", "driver", "memory_used_mib", "memory_total_mib", "util_pct"),
                         [v.strip() for v in line.split(",")])) for line in out]
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        return {"gpus": rows, "compute_apps": apps.splitlines()}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def trim_material(device):
    from robotic_cutting_sim.coupled_robot import CouplingConfig
    from robotic_cutting_sim.live_material import LiveCuttingMaterial
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.seeded_trim import generate_scene, goal_config
    task = compile_task(generate_scene(SEED), goal_config(), seed=42, particle_spacing_m=SPACING)
    config = CouplingConfig(control_dt=task.control_dt_s, material_dt=1e-4, dx=task.grid_dx_m, device=device)
    material = LiveCuttingMaterial(config, task=task, max_cuts=task.max_cuts, cutting_law=task.cutting_law,
                                   solver="tlmpm_v7_2")
    pose = np.asarray(task.reference_edge_pose_world(4.0), float)
    pose[0] = START_X
    return material, pose


def monotone_codes(old, new, max_cuts):
    """Per digit: an assigned digit never changes (0 -> d only)."""
    violations = 0
    for k in range(max_cuts):
        a, b = (old >> (2*k)) & 3, (new >> (2*k)) & 3
        violations += int(np.count_nonzero((a != 0) & (a != b)))
    return violations


def run_batch(n_env, steps, device, out):
    from robotic_cutting_sim.batched import layout
    from robotic_cutting_sim.batched.material_batch import BatchedCuttingMaterial
    t0 = time.perf_counter()
    specs = []
    for _ in range(n_env):
        material, pose = trim_material("cpu")
        specs.append(layout.EnvSpec(material, pose, 42))
    start_pose = pose
    batch = BatchedCuttingMaterial(specs, device=device, strict=True)
    wp.synchronize_device(device)
    build_s = time.perf_counter()-t0
    s, pk = batch.solver, batch.packed
    mass0 = s.mass.numpy().copy()
    code_prev = s.code.numpy().copy()
    radius = 2.4*s.radius
    foreign = wp.zeros(1, dtype=int, device=device)
    visited = wp.zeros(1, dtype=int, device=device)
    rows, timings = [], []
    poses = np.repeat(start_pose[None], n_env, 0)
    for step in range(steps):
        poses = poses.copy()
        poses[:, 0] += DX
        wp.synchronize_device(device)
        t = time.perf_counter()
        batch.advance(poses, np.zeros(n_env, np.int32))          # device pose_interp + graph + wrench + validity
        wp.synchronize_device(device)
        timings.append(time.perf_counter()-t)
        code = s.code.numpy()
        mass = s.mass.numpy()
        diag = batch.diagnostics_read()
        audit = layout.range_audit(pk, s.nodes.numpy(), s.link_offsets.numpy(), s.link_neighbours.numpy(),
                                   s.pending_q.numpy(), s.env_of_particle.numpy())
        foreign.zero_()
        visited.zero_()
        wp.launch(_grouped_query_audit, s.total_particles, [s.hash.id, s.x, s.env_of_particle, radius, foreign,
                                                             visited], device=device)
        po = pk.particle_offset
        per_env = []
        for e in range(n_env):
            sl = slice(int(po[e]), int(po[e+1]))
            per_env.append({"env": e, "labelled": int(np.count_nonzero(code[sl])),
                            "code_monotone_violations": monotone_codes(code_prev[sl], code[sl], s.max_cuts),
                            "mass_bitwise_constant": bool(np.array_equal(mass[sl], mass0[sl])),
                            "finite": not diag[e]["nonfinite"], "minimum_detF": diag[e]["minimum_detF"],
                            "max_speed_m_s": diag[e]["maximum_material_speed_m_s"],
                            "slot_overflow": int(s.slot_overflow.numpy()[e]),
                            "slot_max_occupied": int(s.slot_max_occupied.numpy()[e]),
                            "invalid_reason": int(batch.invalid_reason.numpy()[e]),
                            "pending_wrench": batch.pending_wrench.numpy()[e].tolist(),
                            "link_diagnostics": s.link_diagnostics.numpy()[2*e:2*e+2].tolist()})
        code_prev = code.copy()
        rows.append({"step": step, "wall_s": timings[-1], "range_audit": audit,
                     "grouped_query": {"visited": int(visited.numpy()[0]), "foreign": int(foreign.numpy()[0])},
                     "envs": per_env})
    final = {e: batch.snapshot(e) for e in range(n_env)}
    l2 = all(all(r["mass_bitwise_constant"] and r["finite"] and r["code_monotone_violations"] == 0
                 and r["slot_overflow"] == 0 and r["minimum_detF"] > 0 and r["invalid_reason"] == 0
                 for r in row["envs"]) and set(row["range_audit"].values()) == {0}
             and row["grouped_query"]["foreign"] == 0 and row["grouped_query"]["visited"] > 0 for row in rows)
    copies = {}
    if n_env > 1:
        x0 = final[0]["x"]
        copies = {"max_position_difference_to_env0_m": [float(np.abs(final[e]["x"]-x0).max()) for e in range(n_env)],
                  "labelled_counts": [int(np.count_nonzero(final[e]["code"])) for e in range(n_env)],
                  "codes_identical_to_env0": [bool(np.array_equal(final[e]["code"], final[0]["code"]))
                                              for e in range(n_env)]}
    result = {"num_envs": n_env, "particles_per_env": pk.particle_counts.tolist(),
              "total_particles": pk.total_particles, "total_nodes": pk.total_nodes, "build_wall_s": build_s,
              "graph": batch.capture_receipt, "memory_bytes": batch.memory_bytes(),
              "memory_bytes_per_env": batch.memory_bytes()/n_env,
              "control_step_wall_s": timings, "warm_control_step_wall_s_median": float(np.median(timings[1:]))
              if len(timings) > 1 else None, "steps": rows, "copies": copies, "l2_pass": bool(l2),
              "receipt": batch.receipt()}
    (out/f"batch_n{n_env}.json").write_text(json.dumps(result, indent=1, default=str)+"\n")
    return result, start_pose, final


def run_single(steps, device, start_pose, out):
    material, pose = trim_material(device)
    t0 = time.perf_counter()
    material.reset(pose, seed=42)
    build_s = time.perf_counter()-t0
    timings = []
    current = pose.copy()
    for step in range(steps):
        current = current.copy()
        current[0] += DX
        t = time.perf_counter()
        material.advance(current, (step+1)*material.config.control_dt, cut_index=0)
        timings.append(time.perf_counter()-t)
    snap = material.debug_state()
    result = {"build_wall_s": build_s, "control_step_wall_s": timings,
              "warm_control_step_wall_s_median": float(np.median(timings[1:])) if steps > 1 else None,
              "labelled": int(np.count_nonzero(snap["codes"])), "memory_bytes": material.bed.memory_bytes()}
    (out/"single.json").write_text(json.dumps(result, indent=1)+"\n")
    return result, snap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)     # never overwrite an output directory
    wp.init()
    meta = {"command": " ".join([sys.executable]+sys.argv), "cwd": os.getcwd(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "python": platform.python_version(),
            "warp": wp.config.version, "numpy": np.__version__, "device": args.device,
            "device_name": wp.get_device(args.device).name,
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "workload": {"scene": f"seeded_trim id seed {SEED}", "particle_spacing_m": SPACING, "grid_dx_m": .004,
                         "material_dt_s": 1e-4, "substeps_per_control_step": 200, "control_hz": 50,
                         "motion": f"+x {DX} m per control step from x={START_X}, cut digit 0",
                         "cameras": "none", "recording": "none", "obstacle_boxes": "handle only (1)"},
            "gpu_before": gpu_snapshot()}
    (out/"meta.json").write_text(json.dumps(meta, indent=1)+"\n")
    summary = {"batches": {}}
    single, single_snap = run_single(args.steps, args.device, None, out)
    summary["single"] = {k: single[k] for k in ("warm_control_step_wall_s_median", "labelled", "memory_bytes")}
    for n_env in args.envs:
        result, _, final = run_batch(n_env, args.steps, args.device, out)
        summary["batches"][n_env] = {
            "l2_pass": result["l2_pass"], "warm_control_step_wall_s_median": result["warm_control_step_wall_s_median"],
            "memory_bytes_per_env": result["memory_bytes_per_env"],
            "allocated_during_capture_bytes": (result["graph"] or {}).get("allocated_during_capture_bytes"),
            "labelled_final": [int(np.count_nonzero(final[e]["code"])) for e in final],
            "descriptive_vs_single_gpu": {
                "max_position_difference_m": [float(np.abs(final[e]["x"]-single_snap["positions"]).max())
                                              for e in final],
                "labelled_difference": [int(np.count_nonzero(final[e]["code"]))-single["labelled"] for e in final]},
            "copies": result["copies"]}
    summary["gpu_after"] = gpu_snapshot()
    (out/"summary.json").write_text(json.dumps(summary, indent=1, default=str)+"\n")
    print(json.dumps(summary, indent=1, default=str))
    return 0 if all(v["l2_pass"] for v in summary["batches"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
