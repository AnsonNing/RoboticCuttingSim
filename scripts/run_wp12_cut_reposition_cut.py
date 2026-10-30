"""WP-12 Phase 6 part A: one kinematic episode of cut -> reposition -> cut (opt-in step field planner_offset_local_m).

Goal (declared here, before the run): on ``configs/wp12/scenes/bar_flat_board.json`` sever at local x1, push the +x
piece by d along +x with a cut-disabled kerf_face blade push, then sever the pushed piece at reference x2 =
node(midpoint(x1, L/2)) with ``planner_offset_local_m = [d, 0, 0]`` (the knife goes to x2 + d: open loop, declared
push distance). goal_eval_v2a scores every step on its declared reference-space plane. The push displacement is
measured from the recorded (privileged, evaluation-only) particle history and compared with the declared d.

Kinematic ideal-tracking runtime (no robot dynamics, grasp or cameras). The default resolution (particle spacing
4 mm, material dt 0.5 ms, Warp CPU) is DEVELOPMENT ONLY, not an acceptance resolution.

    TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_cut_reposition_cut.py \
        --out artifacts/wp12/phase6_goalfeat_001
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from robotic_cutting_sim import goal_planner  # noqa: E402
from robotic_cutting_sim import scene_config as sc  # noqa: E402
from robotic_cutting_sim.policies import WaypointPolicy  # noqa: E402


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def build_goal(scene, x1_fraction, push, duration, dx):
    L = scene["workpiece"]["base_box_m"]["length_m"]
    x1 = round(x1_fraction*L/dx)*dx
    x2 = round(((x1+L/2)/2)/dx)*dx
    f1 = [(x1+L/2)/L, (L/2-x1)/L]
    f2 = [(x2+L/2)/L, (L/2-x2)/L]

    def sever(x, goal_id, fractions, **extra):
        return {"type": "sever", "goal_id": goal_id, "planes": [{"origin_local_m": [x, 0., 0.], "normal_local": [1., 0., 0.]}],
                "expected_pieces": 2, "minimum_part_mass_fraction": round(.5*min(fractions), 6), **extra}

    push_step = {"type": "move_piece", "goal_id": "push", "direction_local": [1., 0., 0.], "distance_m": push,
                 "selector": {"side_of_plane": {"origin_local_m": [x1, 0., 0.], "normal_local": [1., 0., 0.], "side": "positive"}}}
    goal = {"schema": "rcs_goal_v1", "goal_id": "cut_reposition_cut", "type": "sequence",
            "description": f"Sever at x = {x1*1e3:.1f} mm, kerf_face push of the +x piece by {push*1e3:.1f} mm, sever the pushed "
                           f"piece at reference x = {x2*1e3:.1f} mm planned at {(x2+push)*1e3:.1f} mm (planner_offset_local_m).",
            "episode": {"duration_s": duration, "control_dt_s": .02},
            "planner": {"controller_profile": "path_v2", "cut_extent": "plane_section"},
            "evaluator_version": "goal_eval_v2a",
            "steps": [sever(x1, "sever1", f1), push_step, sever(x2, "sever2", f2, planner_offset_local_m=[push, 0., 0.])]}
    return goal, x1, x2


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scene", type=Path, default=ROOT/"configs/wp12/scenes/bar_flat_board.json")
    p.add_argument("--x1-fraction", type=float, default=0.)
    p.add_argument("--push-m", type=float, default=.02)
    p.add_argument("--particle-spacing-m", type=float, default=.004)
    p.add_argument("--material-dt", type=float, default=.0005)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--history-stride", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=None, help="smoke only: stop early (the result is labelled partial)")
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; never overwrite experiment output")
    args.out.mkdir(parents=True)
    started = time.monotonic()
    scene = sc.resolve_scene(args.scene)
    dx = scene["solver"]["grid_dx_m"]
    probe_goal, x1, x2 = build_goal(scene, args.x1_fraction, args.push_m, 120., dx)
    probe = sc.compile_task(scene, probe_goal, particle_spacing_m=args.particle_spacing_m)
    options = goal_planner.planner_options(probe.goal, None)
    limits = goal_planner.action_limits(options, probe.scene)
    waypoints = goal_planner.plan(probe.scene, probe.goal, probe.tool_profile, options)
    plan_end = goal_planner.ideal_duration_s(probe, waypoints, limits, options)
    duration = math.ceil((plan_end+1.)/.5)*.5   # WP-11 Phase E duration rule
    goal, x1, x2 = build_goal(scene, args.x1_fraction, args.push_m, duration, dx)
    task = sc.compile_task(scene, goal, particle_spacing_m=args.particle_spacing_m)
    options = goal_planner.planner_options(task.goal, None)
    limits = goal_planner.action_limits(options, task.scene)
    waypoints, report = goal_planner.plan_with_report(task.scene, task.goal, task.tool_profile, options)
    command = " ".join([f"TMPDIR={__import__('os').environ.get('TMPDIR', '')}", ".venv/bin/python", *sys.argv])
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    result = {"schema": "rcs_wp12_phase6_goalfeat_run_v1", "label": "DEV kinematic ideal-tracking, coarse resolution "
              "(development only; not an acceptance resolution, not a live result)",
              "command": command, "git_head": head, "git_dirty_files": dirty, "python": platform.python_version(),
              "started": datetime.now().astimezone().isoformat(), "task_id": task.task_id, "config_sha256": task.config_sha256,
              "scene_path": str(args.scene), "goal": task.goal, "planner_options": options, "plan_report": report,
              "plan_end_s_ideal": plan_end, "duration_s": duration, "seed": args.seed, "device": args.device,
              "particle_spacing_m": args.particle_spacing_m, "material_dt_s": args.material_dt, "grid_dx_m": task.grid_dx_m,
              "x1_local_m": x1, "x2_reference_local_m": x2, "x2_planned_local_m": x2+args.push_m,
              "declared_push_m": args.push_m, "status": "RUNNING"}
    (args.out/"config.json").write_text(json.dumps(jsonable({k: result[k] for k in result if k != "status"}), indent=1))
    env = None
    try:
        env = sc.make_configured_kinematic_env(task, seed=args.seed, device=args.device, material_dt=args.material_dt,
                                               delta_frame="world", ik_failure="reject", history_stride=args.history_stride,
                                               action_limits=limits)
        observation, _ = env.reset(seed=args.seed)
        setup_wall = time.monotonic()-started
        policy = WaypointPolicy(env.task, waypoints, limits, **goal_planner.policy_options(options))
        steps = int(round(task.duration_s/task.control_dt_s))
        if args.max_steps is not None:
            steps = min(steps, args.max_steps)
            result["partial"] = True
        phases, loop_start = [], time.monotonic()
        clamped = 0
        for step in range(steps):
            action = policy(observation)
            observation, _, terminated, truncated, info = env.step(action)
            clamped += bool((info.get("action_report") or {}).get("workspace_clamped", False))
            if not phases or phases[-1]["phase"] != policy.phase:
                phases.append({"phase": policy.phase, "time_s": float(env.time_s), "step": step+1})
            if step % 100 == 0:
                print(f"[P6A] step {step+1}/{steps} t={env.time_s:.2f}s phase={policy.phase} "
                      f"wall={time.monotonic()-loop_start:.0f}s", flush=True)
            if terminated or truncated:
                break
        episode_wall = time.monotonic()-loop_start
        eval_start = time.monotonic()
        evaluation = env.evaluate()
        evaluation_wall = time.monotonic()-eval_start
        # Push displacement from the privileged recorded history (evaluation-only, never an observation).
        material = env.backend.material
        history = env._history
        T_inv = np.linalg.inv(task.T_world_workpiece)
        to_local = lambda pts: pts@T_inv[:3, :3].T+T_inv[:3, 3]
        ref = to_local(np.asarray(material.world_points, float))
        masses = np.asarray(material.masses, float)
        spacing = args.particle_spacing_m
        right = ref[:, 0] > x1+spacing
        left = ref[:, 0] < x1-spacing
        times = np.asarray([s["time_s"] for s in history], float)

        def phase_time(prefix):
            rows = [r for r in phases if r["phase"].startswith(prefix)]
            return rows[0]["time_s"] if rows else None

        def displacement(t, mask):
            k = int(np.argmin(np.abs(times-t)))
            local = to_local(np.asarray(history[k]["positions"], float))
            d = local[mask]-ref[mask]
            return {"frame_time_s": float(times[k]), "mass_weighted_mean_m": (masses[mask]@d/masses[mask].sum()).tolist(),
                    "x_p05_m": float(np.percentile(d[:, 0], 5)), "x_p50_m": float(np.median(d[:, 0])),
                    "x_p95_m": float(np.percentile(d[:, 0], 95))}

        t_cut1 = phase_time("APPROACH:cut1")
        measured = {"definition": "local-frame displacement of the reference particles with x > x1 + spacing (pushed side) and "
                                  "x < x1 - spacing (other side) from the recorded history frame nearest the phase time",
                    "at_start_of_APPROACH_cut1": None if t_cut1 is None else
                    {"pushed_side": displacement(t_cut1, right), "other_side": displacement(t_cut1, left)},
                    "at_end": {"pushed_side": displacement(times[-1], right), "other_side": displacement(times[-1], left)}}
        if t_cut1 is not None:
            dx_pushed = measured["at_start_of_APPROACH_cut1"]["pushed_side"]["mass_weighted_mean_m"][0]
            measured["pushed_minus_declared_x_m"] = dx_pushed-args.push_m
            measured["implied_second_cut_reference_x_m"] = x2+args.push_m-dx_pushed
        outcome = evaluation.get("outcome", {})
        result.update(status="COMPLETE", policy_done=bool(policy.done), steps_run=step+1, workspace_clamped_steps=clamped,
                      phases=phases, evaluation=evaluation, verdict=outcome.get("status"),
                      step_outcomes=[{"index": r["index"], "goal_id": r["goal_id"], "type": r["type"],
                                      "status": r["outcome"]["status"],
                                      "failed_checks": [k for k, v in r["outcome"].get("checks", {}).items() if v is False]}
                                     for r in outcome.get("steps", [])],
                      push_measurement=measured,
                      runtime={"setup_wall_s": setup_wall, "episode_wall_s": episode_wall, "evaluation_wall_s": evaluation_wall,
                               "total_wall_s": time.monotonic()-started, "simulated_s": float(env.time_s)},
                      particles=int(material.bed.n), finished=datetime.now().astimezone().isoformat())
    except Exception as error:   # preserved as a result
        result.update(status="ABORTED", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                      runtime={"total_wall_s": time.monotonic()-started})
    finally:
        if env is not None:
            env.close()
    (args.out/"result.json").write_text(json.dumps(jsonable(result), indent=1))
    print(json.dumps(jsonable({k: result.get(k) for k in ("status", "verdict", "step_outcomes", "push_measurement", "runtime",
                                                          "task_id", "error")}), indent=1))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
