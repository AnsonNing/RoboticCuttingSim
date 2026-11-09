"""G4 (ADR 0022 §8): batched N=1 B2 against the single ``IsaacLabCuttingEnv`` on the same registered ruler task.

Both runs: ``feedback=False`` (no material wrench on the robot), the identical ``joint_position_rad`` arm sequence
(no IK), 100 control steps, physical grasp. Not an acceptance gate for coupling; the tolerance is fixed before
running as 3 x the single env's own 3-run spread (per quantity, max over the steps).

    --mode single   the frozen single env (``isaac_lab_env.make_lab_cfg``, task ``TRIM-01-R``, physical grasp,
                    cameras off, ``action_mode="joint_position_rad"``); writes its settled arm q and the trace
    --mode batched  ``IsaacLabBatchedCuttingEnv`` N=1 B2, robot-only, HOME = the single run's settled arm q
                    (``--initial-from``), the same joint sequence through ``step_joint_targets`` (G4 harness)

Sequence: q_k = q_settled + A * sin(2 pi k / 50), A = (0, .05, 0, .05, 0, .05, 0) rad.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src flock <gpu0.lock> .deps/isaaclab-venv/bin/python \
        scripts/rl_ready/probe_lab_batched_g4.py --mode single --out artifacts/rl_ready/p1_1b/g4_single_001
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

TASK_ID = "TRIM-01-R"
STATIC_SCENE = "/home/ningan/RoboticCuttingSim/artifacts/research_handoff_001/workcell/workcell.usda"
AMPLITUDE = np.array([0., .05, 0., .05, 0., .05, 0.])
STEPS = 100


def sequence(q0, steps=STEPS):
    k = np.arange(1, steps+1)[:, None]
    return np.asarray(q0, float)[None]+AMPLITUDE[None]*np.sin(2*np.pi*k/50.)


def mode_single(args, out):
    import torch
    from isaaclab.app.sim_launcher import launch_simulation
    from robotic_cutting_sim.isaac_lab_env import IsaacLabCuttingEnv, LabCuttingConfig, make_lab_cfg
    options = LabCuttingConfig(device="cuda:0", material_device="cuda:0", camera=False, task_id=TASK_ID,
                               tool_attachment_mode="physical_grasp", feedback=False,
                               action_mode="joint_position_rad", static_scene=STATIC_SCENE, seed=42)
    cfg = make_lab_cfg(options)
    rows = []
    with launch_simulation(cfg, {"headless": True, "visualizer": [], "device": "cuda:0"}):
        env = IsaacLabCuttingEnv(cfg, options)
        env.reset(seed=42)
        settled = env.measured()
        q0 = np.asarray(settled["q"][0], float)
        for q in sequence(q0):
            env.step(torch.tensor(q[None], dtype=torch.float32, device="cuda:0"))
            m = env.measured()
            rows.append(dict(arm_q=np.asarray(m["q"][0], float), hand=np.asarray(m["hand_pose"], float),
                             tool=np.asarray(m["tool_body_pose_xyzw"], float), finger=np.asarray(m["finger_q"][0], float)))
        info = dict(grasp_initialization=env.grasp_initialization, settled_q=q0.tolist(),
                    settled_finger=np.asarray(settled["finger_q"][0], float).tolist())
    return rows, info


def mode_batched(args, out):
    import torch
    from isaaclab.app.sim_launcher import launch_simulation
    from robotic_cutting_sim.batched import lab_scene
    from robotic_cutting_sim.rl.lab_batched_env import IsaacLabBatchedCuttingEnv
    from robotic_cutting_sim.task_scenarios import configure_task
    from robotic_cutting_sim.tasks import make_task
    q0 = np.asarray(json.loads((Path(args.initial_from)/"g4.json").read_text())["info"]["settled_q"], float)
    task = configure_task(make_task(TASK_ID, seed=42), scenario=None, material_profile_id=None)   # as make_lab_cfg
    options = lab_scene.BatchedLabOptions(num_envs=1, tool_mount="physical_grasp_v1", device="cuda:0",
                                          feedback=False, initial_joint_positions=tuple(q0), seed=42)
    cfg = lab_scene.make_batched_lab_cfg(options, task)
    rows = []
    with launch_simulation(cfg, {"headless": True, "visualizer": [], "device": "cuda:0"}):
        from isaaclab_newton.physics import NewtonManager
        env = IsaacLabBatchedCuttingEnv(cfg, options, task)
        env.reset()
        names = list(env.robots[0].joint_names)
        arm = [names.index(f"panda_joint{i}") for i in range(1, 8)]
        fingers = [names.index("panda_finger_joint1"), names.index("panda_finger_joint2")]
        for q in sequence(q0):
            env.step_joint_targets(torch.tensor(q[None], dtype=torch.float32, device="cuda:0"))
            body = NewtonManager.get_state_0().body_q.numpy()
            jp = env.robots[0].data.joint_pos.torch.cpu().numpy()[0]
            rows.append(dict(arm_q=jp[arm].astype(float), hand=body[env.tables["hand"][0]].astype(float),
                             tool=body[env.tables["tool"][0]].astype(float), finger=jp[fingers].astype(float)))
        info = dict(grasp_initialization={k: v for k, v in (env.grasp_initialization or {}).items()
                                          if k != "accumulators"}, initial_q=q0.tolist())
    return rows, info


def compare(args, out):
    """Tolerance = 3 x the single runs' spread (max over steps of the max pairwise difference), per quantity."""
    from scipy.spatial.transform import Rotation
    singles = [np.load(Path(p)/"g4_trace.npz") for p in args.singles]
    batched = [np.load(Path(p)/"g4_trace.npz") for p in args.batched]

    def position_diff(a, b, key):
        return np.linalg.norm(a[key][:, :3]-b[key][:, :3], axis=1)

    def angle_diff(a, b, key):
        return (Rotation.from_quat(a[key][:, 3:]).inv()*Rotation.from_quat(b[key][:, 3:])).magnitude()

    metrics = {"arm_q_rad": lambda a, b: np.abs(a["arm_q"]-b["arm_q"]).max(1),
               "hand_position_m": lambda a, b: position_diff(a, b, "hand"),
               "tool_position_m": lambda a, b: position_diff(a, b, "tool"),
               "tool_rotation_rad": lambda a, b: angle_diff(a, b, "tool"),
               "finger_q_m": lambda a, b: np.abs(a["finger"]-b["finger"]).max(1)}
    result = {}
    for name, fn in metrics.items():
        spread = np.zeros(len(singles[0]["arm_q"]))
        for i in range(len(singles)):
            for j in range(i+1, len(singles)):
                spread = np.maximum(spread, fn(singles[i], singles[j]))
        tolerance = 3.*float(spread.max())
        curves = [np.max([fn(b, s) for s in singles], axis=0) for b in batched]
        worst = float(max(c.max() for c in curves))
        result[name] = dict(single_spread_max=float(spread.max()), tolerance=tolerance, batched_vs_single_max=worst,
                            within_tolerance=worst <= tolerance,
                            divergence_curve_every_10_steps=[float(v) for v in curves[0][9::10]])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=("single", "batched", "compare"))
    parser.add_argument("--initial-from", default=None, help="batched: a single run directory (settled q)")
    parser.add_argument("--singles", nargs="*", default=())
    parser.add_argument("--batched", nargs="*", default=())
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    if args.mode == "compare":
        result = compare(args, out)
        (out/"g4_compare.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        return
    rows, info = (mode_single if args.mode == "single" else mode_batched)(args, out)
    np.savez_compressed(out/"g4_trace.npz", **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
    (out/"g4.json").write_text(json.dumps({"mode": args.mode, "task_id": TASK_ID, "steps": STEPS,
                                           "amplitude_rad": AMPLITUDE.tolist(), "feedback": False, "info": info,
                                           "wall_s": time.perf_counter()-start}, indent=1, default=str))
    print(json.dumps({"mode": args.mode, "info": info}, default=str)[:2000])


if __name__ == "__main__":
    main()
