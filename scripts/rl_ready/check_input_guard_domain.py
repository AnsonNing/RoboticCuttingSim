"""P1-2b: replay the recorded legitimate material inputs through the material input guard's checks (no physics).

Inputs (read-only): the WP-17 Native measured edge / pad-box trajectories of seeds 512 and 15 (workload A replay
drive of P1-0/P1-2) and the P1-1b S3 workload B material input sequences (``material_inputs.npz`` of
``s3_b1_n2_001`` and ``s3_b2_n2_001``: edge, cut, pad boxes per step, recorded by the Lab run). Every step of every
sequence is checked with ``MaterialInputGuard.check_pose`` / ``check_boxes`` against the full-resolution (1 mm)
domain of its scene; the result is the number of steps that would trip each check (expected 0 for every
legitimate input) and the margins left to each threshold. Runs on the CPU (Warp CPU, Torch CPU):

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rl_ready/check_input_guard_domain.py \
        --out artifacts/rl_ready/p1_2b/guard_domain_check_001
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _sequence_report(guard, e, pose, last, boxes=None):
    """Per-step bits of one env's sequence (T steps); ``last`` is the previous accepted pose of every step."""
    import torch
    from robotic_cutting_sim.rl.material_guard import GUARD_REASONS
    N = guard.num_envs
    T = len(pose)
    bits = np.zeros(T, np.int64)
    jump, angle = np.zeros(T), np.zeros(T)
    for t in range(T):
        p = torch.zeros((N, 7), dtype=torch.float64)
        q = torch.zeros((N, 7), dtype=torch.float64)
        p[:, 6] = q[:, 6] = 1.
        p[e] = torch.as_tensor(pose[t])
        q[e] = torch.as_tensor(last[t])
        b = int(guard.check_pose(p, q)[e])
        if boxes is not None:
            c = torch.full((N, boxes.shape[1], 3), .01, dtype=torch.float32)
            h = torch.full((N, boxes.shape[1], 3), .01, dtype=torch.float32)
            c[e] = torch.as_tensor(boxes[t][:, 0])
            h[e] = torch.as_tensor(boxes[t][:, 1])
            b |= int(guard.check_boxes(c, h)[e])
        bits[t] = b
        jump[t] = np.linalg.norm(pose[t][:3] - last[t][:3])
        qa, qb = pose[t][3:] / np.linalg.norm(pose[t][3:]), last[t][3:] / np.linalg.norm(last[t][3:])
        angle[t] = 2 * np.arccos(min(1., abs(float(qa @ qb))))
    R = guard.R_world_local[e].numpy()
    local = (pose[:, :3] - guard.t_world_local[e].numpy()) @ R
    lo, hi = guard.lo[e].numpy(), guard.hi[e].numpy()
    out = dict(steps=T, tripped_steps=int(np.count_nonzero(bits)),
               reasons={name: int(np.count_nonzero(bits & bit)) for bit, name in GUARD_REASONS.items()},
               max_translation_step_m=float(jump.max()), max_rotation_step_rad=float(angle.max()),
               min_distance_to_domain_boundary_m=float(min((local - lo).min(), (hi - local).min())))
    if boxes is not None:
        out["max_pad_center_m"] = float(np.linalg.norm(boxes[:, :, 0], axis=-1).max())
        out["max_pad_half_extent_m"] = float(boxes[:, :, 1].max())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    from run_vector_milestones import family_for, load_inputs
    from robotic_cutting_sim.rl import open_vector_env
    from robotic_cutting_sim.rl.material_guard import MaterialInputGuard, MaterialInputGuardConfig
    seeds = (512, 15)
    inputs = load_inputs(seeds)
    fam = family_for(seeds, inputs)
    report = dict(config=MaterialInputGuardConfig().__dict__, workload_a_replay={}, workload_b={})
    with open_vector_env(2, task=fam, device="cpu", parity_mode=True) as env:
        guard = env.input_guard
        report["domain_local_frame"] = guard.receipt()["domain_local_frame"]
        for e, seed in enumerate(seeds):
            traj = inputs[seed]["trajectory"]
            pose = np.asarray(traj.edge_pose_world_xyzw, float)
            last = np.vstack([np.asarray(traj.reset_edge_pose_world_xyzw, float)[None], pose[:-1]])
            report["workload_a_replay"][str(seed)] = _sequence_report(guard, e, pose, last,
                                                                      np.asarray(traj.pad_boxes_edge, np.float32))

        class _Shim:   # the same packed grid/frames with workload B's HOME edge as the initial measured pose
            def __init__(self, material, initial):
                self.packed = material.packed
                self.initial_measured_pose = type("A", (), {"numpy": lambda s: initial})()
        for run in ("s3_b1_n2_001", "s3_b2_n2_001"):
            path = ROOT / "artifacts/rl_ready/p1_1b" / run / "material_inputs.npz"
            if not path.exists():
                report["workload_b"][run] = "NOT_RUN: input file missing"
                continue
            d = np.load(path)
            run_seeds = [int(s) for s in d["scene_seeds"]]
            if run_seeds != list(seeds):
                report["workload_b"][run] = f"NOT_RUN: scene seeds {run_seeds}"
                continue
            shim_guard = MaterialInputGuard(_Shim(env.material, np.asarray(d["initial_edge"], float)), 2, "cpu")
            edges = np.asarray(d["edge"], float)
            T = edges.shape[0]
            boxes = np.stack([np.asarray(d["box_centers"], np.float32).reshape(T, 2, -1, 3),
                              np.asarray(d["box_halves"], np.float32).reshape(T, 2, -1, 3)], 3)   # [T, N, B, 2, 3]
            per_env = {}
            for e, seed in enumerate(seeds):
                last = np.vstack([np.asarray(d["initial_edge"], float)[e][None], edges[:-1, e]])
                per_env[f"env{e}_seed{seed}"] = _sequence_report(shim_guard, e, edges[:, e], last, boxes[:, e])
            report["workload_b"][run] = dict(domain_local_frame=shim_guard.receipt()["domain_local_frame"], **per_env)
    report["wall_s"] = time.perf_counter() - start
    rows = [v for v in report["workload_a_replay"].values()] + [
        v for r in report["workload_b"].values() if isinstance(r, dict) for k, v in r.items() if k.startswith("env")]
    report["status"] = "PASS" if rows and all(r["tripped_steps"] == 0 for r in rows) else "FAIL"
    (out / "guard_domain_check.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "workload_a_replay", "workload_b", "wall_s")}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
