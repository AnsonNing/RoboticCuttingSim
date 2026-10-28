#!/usr/bin/env python3
"""Full-episode controller validation on the ideal-tracking kinematic backend.

For each task x controller: run a complete episode through CuttingTaskEnv (the
same API as the live runtimes), check every action against the step limits and
the executed edge, then score with the task's frozen oracle (env.evaluate()).
Results are pipeline evidence only (runtime_source = kinematic_ideal_tracking):
no robot, IK, grasp or camera is involved. One JSON row per run.

  python scripts/run_kinematic_validation.py --out artifacts/kinematic_validation_001.json
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

CUT_TASKS = ("CUT-01-R", "CUT-02", "CUT-01")
CONTROLLERS = ("reference", "push_policy", "saw_policy", "oblique20_policy", "yaw30_policy", "gym_saw_policy", "agent_api")


def _plan(task, controller):
    """Waypoints for a scripted controller, or None when it does not apply to the task."""
    from robotic_cutting_sim.policies import skim, through_cut
    if task.goal_kind == "flatten":
        options = {"push_policy": {}, "saw_policy": {"saw_amplitude_m": .006}, "gym_saw_policy": {"saw_amplitude_m": .006}}
        return skim(task, **options[controller]) if controller in options else None
    options = {"push_policy": {}, "saw_policy": {"saw_amplitude_m": .01, "saw_period_s": .6},
               "gym_saw_policy": {"saw_amplitude_m": .01, "saw_period_s": .6},
               "oblique20_policy": {"tilt_deg": 20.}, "yaw30_policy": {"yaw_deg": 30.}}
    return through_cut(task, cut_speed_m_s=.01, **options[controller]) if controller in options else None


def _agent_session(env):
    from scipy.spatial.transform import Rotation
    from robotic_cutting_sim.agent_api import CuttingAgentAPI
    agent = CuttingAgentAPI(env)
    calls = [("describe_task", agent.describe_task())]
    calls.append(("reset", agent.reset()))
    task = env.task
    rpy = Rotation.from_quat(task.reference_edge_pose_world(0.)[3:]).as_euler("xyz", degrees=True)
    if task.goal_kind == "sever":
        top = task.geometry.base_height_m
        calls.append(("move_edge_to", agent.move_edge_to([0., 0., top+.008], rpy)))
        calls.append(("cut_along", agent.cut_along("advance", top+.012, speed_m_s=.01, saw_amplitude_m=.008)))
        calls.append(("move_edge_by", agent.move_edge_by([0, 0, .04])))
    else:
        from robotic_cutting_sim.policies import skim
        plan = skim(task)  # same pass a scripted policy uses, issued as agent tool calls
        start, end = np.asarray(plan[1].pose[:3]), np.asarray(plan[2].pose[:3])
        calls.append(("move_edge_to", agent.move_edge_to(start+[0, 0, .03], rpy)))
        calls.append(("move_edge_to", agent.move_edge_to(start, rpy)))
        calls.append(("cut_along", agent.cut_along(list(end-start), float(np.linalg.norm(end-start)), speed_m_s=.025)))
        calls.append(("move_edge_by", agent.move_edge_by([0, 0, .03])))
    calls.append(("finish_episode", agent.finish_episode()))
    return [{"call": name, "result": result} for name, result in calls]


def run_one(job):
    task_id, controller, spacing, material_dt, device = job
    from robotic_cutting_sim.control import DELTA_MODE, ActionLimits
    from robotic_cutting_sim.kinematic_env import make_kinematic_env
    from robotic_cutting_sim.policies import WaypointPolicy
    from robotic_cutting_sim.reward import CuttingReward
    row = {"task_id": task_id, "controller": controller, "particle_spacing_m": spacing,
           "material_dt_s": material_dt, "device": device, "runtime_source": "kinematic_ideal_tracking"}
    start = time.time()
    try:
        limits = ActionLimits()
        # Policies send world-frame deltas; the Gym adapter's actions use the env default frame.
        env = make_kinematic_env(task_id, particle_spacing_m=spacing, material_dt=material_dt, device=device,
                                 action_limits=limits, ik_failure="reject", delta_frame="world")
        env.reward = CuttingReward(env.task)
        if controller not in ("reference", "agent_api") and _plan(env.task, controller) is None:
            return {**row, "status": "NOT_APPLICABLE"}
        adapter = None
        if controller.startswith("gym_"):
            from robotic_cutting_sim.gym_env import CuttingGymEnv
            adapter = CuttingGymEnv(env)  # its constructor performs the reset
            observation = env.observe()
        else:
            observation, _ = env.reset()
        actions, max_translation, max_rotation, tracking, rewards = 0, 0., 0., 0., 0.
        if controller == "agent_api":
            row["agent_calls"] = _agent_session(env)  # the session resets the env itself
        else:
            policy = None if controller == "reference" else WaypointPolicy(env.task, _plan(env.task, controller), limits)
            while env.time_s < env.task.duration_s-1e-9:
                if policy is None:
                    action = None
                else:
                    action = policy(observation)
                if adapter is not None and action is not None and not policy.done:
                    normalized = np.r_[action["target"][:3]/limits.max_translation_m,
                                       action["target"][3:]/limits.max_rotation_rad]
                    adapter.cut_enabled = action["cut_enabled"]
                    _, reward, terminated, truncated, info = adapter.step(normalized)
                    observation = env.observe()
                else:
                    observation, reward, terminated, truncated, info = env.step(action)
                rewards += reward
                actions += 1
                report = info.get("action_report") or {}
                max_translation = max(max_translation, report.get("applied_translation_m", 0.))
                max_rotation = max(max_rotation, report.get("applied_rotation_rad", 0.))
                diagnostics = env.backend.last_diagnostics
                tracking = max(tracking, float(np.linalg.norm(diagnostics["commanded_edge_pose"][:3]
                                                             - diagnostics["measured_edge_pose"][:3])))
                if terminated:
                    row["terminated"] = info.get("termination_reason")
                    break
            row.update({"steps": actions, "policy_completed_waypoints": None if policy is None else policy.done,
                        "max_applied_translation_m": max_translation, "max_applied_rotation_rad": max_rotation,
                        "max_tracking_error_m": tracking, "episode_reward": rewards})
        result = env.evaluate()
        metrics = result.get("metrics", {})
        row.update({"status": result.get("status"), "success": result.get("success"),
                    "failed_checks": sorted(k for k, v in result.get("checks", {}).items() if not v),
                    "missing": result.get("missing"),
                    "metrics": {k: metrics[k] for k in ("final_components", "substantial_parts",
                                "two_main_parts_mass_fraction", "maximum_reference_midpoint_plane_distance_m",
                                "removed_reference_above_plane_mass_fraction", "surface") if k in metrics},
                    "material_receipt": {k: env.backend.material.parameter_receipt.get(k)
                                         for k in ("solver", "cutting_law", "max_cuts")}})
        try:
            research = env.evaluate(profile="research")
            row["evaluation_v2_research"] = {"assessment": research.get("assessment"),
                "cutting_outcome_status": research.get("cutting_outcome", {}).get("status"),
                "surface_quality_status": research.get("surface_quality", {}).get("status")}
        except Exception as error:  # recorded, never hidden
            row["evaluation_v2_research"] = {"error": f"{type(error).__name__}: {error}"}
    except Exception as error:  # preserved as a failed row, never hidden
        row.update({"status": "ERROR", "error": f"{type(error).__name__}: {error}"})
    row["wall_s"] = round(time.time()-start, 2)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=list(CUT_TASKS)+["TRIM-01-R"])
    parser.add_argument("--controllers", nargs="+", default=list(CONTROLLERS))
    parser.add_argument("--spacing", type=float, default=.004)
    parser.add_argument("--material-dt", type=float, default=.0005)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    jobs = [(t, c, args.spacing, args.material_dt, args.device) for t in args.tasks for c in args.controllers]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_one, jobs))
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"command": " ".join(sys.argv), "commit": commit, "rows": rows}, indent=1,
                              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))+"\n")
    for row in rows:
        print(row["task_id"], row["controller"], row.get("status"), row.get("success"), row.get("failed_checks"),
              row.get("error", ""), row.get("wall_s"))


if __name__ == "__main__":
    main()
