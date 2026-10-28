#!/usr/bin/env python3
"""Render one kinematic-backend episode (particles + knife) to MP4.

A visual check of controller output on the ideal-tracking backend: particles
are colored by current mechanical piece, the ruler blade and handle are drawn
at the commanded (= executed) edge pose. This is NOT a robot or camera video;
runtime_source is kinematic_ideal_tracking and is printed on every frame.

  python scripts/render_kinematic_episode.py --task CUT-01-R --controller saw_policy --out artifacts/saw.mp4
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))


def record(task_id, controller, spacing, material_dt, every):
    from run_kinematic_validation import _agent_session, _plan
    from robotic_cutting_sim.control import ActionLimits
    from robotic_cutting_sim.kinematic_env import make_kinematic_env
    from robotic_cutting_sim.policies import WaypointPolicy
    env = make_kinematic_env(task_id, particle_spacing_m=spacing, material_dt=material_dt,
                             action_limits=ActionLimits(), ik_failure="reject", delta_frame="world")
    frames, step = [], env.step

    def capture(cut):
        state = env.backend.material.world_state()
        frames.append({"time_s": env.time_s, "positions": state["positions"].copy(), "codes": state["codes"].copy(),
                       "edge": env.backend.edge.copy(), "cut": cut})

    def recording_step(action=None):
        result = step(action)
        if int(round(env.time_s/env.task.control_dt_s)) % every == 0:
            capture(bool(action is None or action.get("cut_enabled", True)))
        return result

    env.step = recording_step
    observation, _ = env.reset()
    capture(False)
    if controller == "agent_api":
        _agent_session(env)
    else:
        policy = None if controller == "reference" else WaypointPolicy(env.task, _plan(env.task, controller), ActionLimits())
        while env.time_s < env.task.duration_s-1e-9:
            observation, *_ = env.step(None if policy is None else policy(observation))
    result = env.evaluate()
    return env, frames, {"status": result.get("status"), "success": result.get("success")}


def render(env, frames, outcome, controller, out, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from robotic_cutting_sim.cut_topology import reference_components
    from robotic_cutting_sim.physical_tool import _convex_triangles
    from robotic_cutting_sim.scene_frames import pose_matrix, transform_points
    from robotic_cutting_sim.tool_profile import make_profile_tool_spec

    material, task = env.backend.material, env.task
    spec = make_profile_tool_spec(task.tool_profile) if task.tool_profile.bevel != "legacy_wedge" else None
    blade_vertices, blade_faces = task.tool_profile.blade_mesh_edge()
    cache = {}
    figure = plt.figure(figsize=(8, 6), dpi=110)
    palette = np.array([[.16, .45, .75], [.93, .55, .15], [.35, .65, .3], [.6, .6, .6]])
    points = np.concatenate([f["positions"] for f in frames])
    low, high = points.min(0)-.02, points.max(0)+.02
    high[2] = max(high[2], .09)

    def draw(axis, index):
        frame = frames[index]
        key = frame["codes"].tobytes()
        if key not in cache:
            cache[key] = reference_components(material.points, material.lattice, frame["codes"],
                                              cuts=material.material_config.cuts)[1]
        component = cache[key]
        order = np.argsort(np.bincount(component))[::-1]
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        colors = palette[np.minimum(rank[component], 3)]
        p = frame["positions"]
        axis.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors, s=6, depthshade=False)
        T = pose_matrix(frame["edge"])
        blade = transform_points(T, blade_vertices)
        axis.add_collection3d(Poly3DCollection(blade[blade_faces], facecolor=(.85, .1, .1, .45) if frame["cut"]
                                               else (.5, .5, .5, .45), edgecolor="k", linewidths=.3))
        if spec is not None:
            T_world_tool = T@np.linalg.inv(spec.T_tool_edge)
            handle = transform_points(T_world_tool, spec.shapes[0].vertices_tool_m)
            axis.add_collection3d(Poly3DCollection(handle[_convex_triangles(handle)], facecolor=(.2, .2, .2, .25),
                                                   edgecolor=(.2, .2, .2, .3), linewidths=.2))
        axis.set_xlim(low[0], high[0]); axis.set_ylim(low[1], high[1]); axis.set_zlim(low[2], high[2])
        axis.set_box_aspect(high-low)
        axis.view_init(elev=22, azim=-60)
        axis.set_xlabel("x (m)"); axis.set_ylabel("y (m)"); axis.set_zlabel("z (m)")
        pieces = int(component.max()+1)
        title = (f"{task.task_id} | {controller} | t = {frame['time_s']:.2f} s | pieces = {pieces}"
                 f" | cutting {'ON' if frame['cut'] else 'off'}")
        if index == len(frames)-1:
            title += f"\nfinal oracle: {outcome['status']}"
        axis.set_title(title, fontsize=9)
        figure.text(.01, .01, "kinematic_ideal_tracking: no robot / camera; edge pose = command", fontsize=7)

    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    with writer.saving(figure, str(out), dpi=110):
        for index in range(len(frames)):
            figure.clf()
            draw(figure.add_subplot(projection="3d"), index)
            writer.grab_frame()
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="CUT-01-R")
    parser.add_argument("--controller", default="saw_policy")
    parser.add_argument("--spacing", type=float, default=.004)
    parser.add_argument("--material-dt", type=float, default=.0005)
    parser.add_argument("--every", type=int, default=4, help="record every N control steps")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    env, frames, outcome = record(args.task, args.controller, args.spacing, args.material_dt, args.every)
    render(env, frames, outcome, args.controller, out, args.fps)
    out.with_suffix(".json").write_text(json.dumps({"command": " ".join(sys.argv), "frames": len(frames),
                                                    "outcome": outcome, "runtime_source": "kinematic_ideal_tracking"},
                                                   indent=1)+"\n")
    print(out, outcome)


if __name__ == "__main__":
    main()
