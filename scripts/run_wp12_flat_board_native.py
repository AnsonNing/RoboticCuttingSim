#!/usr/bin/env python3
"""WP-12 Phase 2: Native Isaac Sim checks of the blade_master_flat_board_v1 workcell (one fresh process).

Builds the compiled flat-board task in the existing Native runtime
(``LiveIsaacSimCutting``, physical grasp of ``ruler_knife_v2_y``, HOME reset),
then, in this order:

1. ``start`` render: global D455 RGB, both wrist D405 RGB and the presentation
   overview/close-up RGB (presentation only, never an observation) -> PNG;
2. gravity settle: ``--settle-s`` of coupled steps with the knife held at HOME
   and cutting disabled; workpiece particle drift, lowest particle vs the board
   top, speed and min det F (privileged material state, debug only);
3. knife-board probe (rigid-only controls, material no longer advanced): from
   HOME to a vertical-cut pose above the board at local x = ``--probe-x`` (away
   from the workpiece), then a slow descent of the commanded edge to
   ``--press-depth`` below the board top, a hold and a lift. Per control step:
   the knife's lowest point (blade hull vertices on the measured tool body
   pose) vs the board top, native tool-board contact force, grasp slip (hand-to-
   tool relative pose vs the probe start) and joint tracking error;
4. end render: by default none, because capture() after rigid-only controls fails the runner's wrist-camera
   extrinsic check (first run workcell_003); the knife-over-board render is taken after a coupled (cutting
   disabled) approach instead. ``--end-render`` (opt-in, WP-12 Phase 2b) passes
   ``refresh_camera_extrinsics=True`` to every rigid probe step and renders ``after_probe`` (RGB PNG, depth
   validity and the runner's own extrinsic checks).

The runner itself is unchanged except the opt-in flat-board table/appearance
authoring (``isaac_sim_runner._author_flat_board_appearance``).

  .deps/isaacsim-venv/bin/python scripts/run_wp12_flat_board_native.py --out artifacts/wp12/workcell_NNN --gpu 0
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False)+"\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_png(path, rgb):
    from PIL import Image
    Image.fromarray(np.asarray(rgb, np.uint8)).save(path)


def pose_matrix(pose):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rotation.from_quat(pose[3:]).as_matrix(), pose[:3]
    return T


def relative(hand, tool):
    H = np.linalg.inv(pose_matrix(hand))@pose_matrix(tool)
    return H


def slip(H0, H):
    D = np.linalg.inv(H0)@H
    return float(np.linalg.norm(D[:3, 3])), float(Rotation.from_matrix(D[:3, :3]).magnitude())


def blade_vertices_tool(spec):
    rows = [np.asarray(shape.vertices_tool_m, float) for shape in spec.shapes if shape.role == "blade"]
    if not rows:
        raise RuntimeError("Tool spec has no blade shape")
    return np.vstack(rows)


def lowest_world(points_tool, tool_body_pose):
    T = pose_matrix(np.asarray(tool_body_pose, float))
    world = points_tool@T[:3, :3].T+T[:3, 3]
    i = int(np.argmin(world[:, 2]))
    return float(world[i, 2]), world[i].tolist()


def slerp_path(a, b, max_translation, max_rotation_deg):
    a, b = np.asarray(a, float), np.asarray(b, float)
    distance = float(np.linalg.norm(b[:3]-a[:3]))
    angle = float(np.degrees((Rotation.from_quat(a[3:]).inv()*Rotation.from_quat(b[3:])).magnitude()))
    n = max(1, int(np.ceil(distance/max_translation)), int(np.ceil(angle/max_rotation_deg)))
    from scipy.spatial.transform import Slerp
    s = Slerp([0., 1.], Rotation.from_quat([a[3:], b[3:]]))
    return [np.r_[a[:3]+(b[:3]-a[:3])*u, s(u).as_quat()] for u in np.arange(1, n+1)/n]


def capture_png(env, output, tag, images, depth_stats=None):
    frames, _ = env.capture()
    for name, frame in frames.items():
        if depth_stats is not None:
            valid = np.asarray(frame["valid_mask"], bool)
            depth = np.asarray(frame["depth_m"], float)
            depth_stats[name] = {"valid_fraction": float(valid.mean()),
                                 "depth_valid_min_m": float(depth[valid].min()) if valid.any() else None,
                                 "depth_valid_median_m": float(np.median(depth[valid])) if valid.any() else None,
                                 "T_world_camera_translation_m": np.asarray(frame.get("T_world_camera", np.eye(4)))[:3, 3].tolist()
                                 if frame.get("T_world_camera") is not None else None}
        path = output/f"{tag}_{name}_rgb.png"
        save_png(path, frame["rgb"])
        images.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
    for kind in ("overview_frame", "closeup_frame"):
        frame = getattr(env, kind, None)
        if frame is not None:
            path = output/f"{tag}_presentation_{kind.split('_')[0]}_rgb.png"
            save_png(path, frame["rgb"])
            images.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
    return sorted(frames)


def material_summary(env, task):
    state = env.material.debug_state()
    positions = np.asarray(state["positions"], float)
    out = {"time_s": float(env.material.time_s), "particles": int(len(positions)),
           "centroid_local_m": positions.mean(0).tolist(), "min_z_local_m": float(positions[:, 2].min()),
           "particles_below_board_top": int((positions[:, 2] < 0).sum())}
    if "velocities" in state:
        v = np.asarray(state["velocities"], float)
        out["max_speed_m_s"] = float(np.linalg.norm(v, axis=1).max())
        out["mean_speed_m_s"] = float(np.linalg.norm(v, axis=1).mean())
    if "F" in state:
        F = np.asarray(state["F"], float).reshape(-1, 3, 3)
        out["min_det_F"] = float(np.linalg.det(F).min())
    return out, positions


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scene", default="configs/wp12/scenes/bar_flat_board.json")
    p.add_argument("--goal", default="configs/wp11/goals/sever_mid.json")
    p.add_argument("--settle-s", type=float, default=2.0)
    p.add_argument("--probe-x", type=float, default=0.12, help="workpiece-local x of the knife probe (board, away from the bar)")
    p.add_argument("--probe-start-z", type=float, default=0.010)
    p.add_argument("--press-depth", type=float, default=0.002, help="commanded edge depth below the board top")
    p.add_argument("--descent-speed", type=float, default=0.002)
    p.add_argument("--hold-s", type=float, default=1.0)
    p.add_argument("--skip-settle", action="store_true")
    p.add_argument("--end-render", action="store_true",
                   help="opt-in: refresh wrist-camera extrinsics in every rigid probe step and render after the probe")
    a = p.parse_args()
    output = a.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from robotic_cutting_sim.recording import manifest
    from robotic_cutting_sim.scene_config import compile_task, configured_task_payload
    sources = [Path(__file__)]+[ROOT/"src/robotic_cutting_sim"/n for n in (
        "isaac_sim_runner.py", "scene_config.py", "flat_board_workcell.py", "native_fixture.py", "native_tool.py",
        "live_material.py", "tlmpm_v5.py", "physical_contacts.py")]+[ROOT/"configs/wp12_workcells_v1.json", ROOT/a.scene, ROOT/a.goal]
    hashes = {str(s.relative_to(ROOT)): sha256(s) for s in sources}
    record = manifest({**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, "argv": sys.argv},
                      "wp12_flat_board_native")
    record["source_sha256"] = hashes
    write_json(output/"manifest.json", record)
    result = {"status": "RUNNING", "seed": a.seed, "gpu": a.gpu, "images": []}
    start_wall = time.monotonic()
    env = None
    try:
        task = compile_task(str(ROOT/a.scene), str(ROOT/a.goal), seed=a.seed)
        geometry = task.workcell_geometry()
        if geometry is None:
            raise ValueError("The scene is not a flat-board workcell scene")
        board_top_world = geometry["board_top_world_z_m"]
        result.update(task_id=task.task_id, config_sha256=task.config_sha256, workcell_geometry=geometry)
        from robotic_cutting_sim.physical_tool import ToolGraspConfig
        from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig, LiveIsaacSimCutting, launch_app
        config = IsaacSimCuttingConfig(task_id=task.task_id, seed=a.seed, gpu=a.gpu, device=f"cuda:{a.gpu}",
            camera=True, camera_layout="wrist_d405_global_d455", camera_mode="native_single_view", native_noise="off",
            overview_camera=True, tool_attachment_mode="physical_grasp", tool_init_mode="pregrasped", grasp=ToolGraspConfig(),
            max_cuts=task.max_cuts, configured_task=configured_task_payload(task))
        write_json(output/"runtime_config.json", asdict(config))
        app = launch_app(config, output)
        env = LiveIsaacSimCutting(app, config, output)
        t0 = time.monotonic()
        env.reset(seed=a.seed)
        result["reset_wall_s"] = time.monotonic()-t0
        result["pregrasp"] = {k: v for k, v in (getattr(env, "grasp_reset_receipt", None) or {}).items()
                              if k in ("status", "reason", "max_slip_m", "max_rotation_rad", "finger_forces_n")}
        if (getattr(env, "grasp_reset_receipt", None) or {}).get("status") != "PASS_STABLE_GRASP":
            raise RuntimeError("Physical pregrasp failed: "+str(getattr(env, "grasp_reset_receipt", None)))
        result["fixture_plan"] = env.fixture_plan
        result["home_edge_pose_world"] = np.asarray(env.home_edge_pose).tolist()
        result["start_camera_depth"] = {}
        result["start_cameras"] = capture_png(env, output, "start", result["images"], result["start_camera_depth"])
        write_json(output/"result.json", result)

        # ---- 2. gravity settle (coupled, cutting disabled, knife at HOME)
        home_pose = np.asarray(env.home_edge_pose, float)
        settle = {"steps": 0}
        if not a.skip_settle:
            m0, p0 = material_summary(env, task)
            series = []
            t0 = time.monotonic()
            for i in range(int(round(a.settle_s/task.control_dt_s))):
                env.step({"mode": "cartesian_edge_pose_world", "target": home_pose, "cut_enabled": False}, cutting=False)
                if i % 10 == 9:
                    m, pos = material_summary(env, task)
                    series.append({"time_s": m["time_s"], "centroid_drift_m": float(np.linalg.norm(pos.mean(0)-p0.mean(0))),
                                   "max_particle_displacement_m": float(np.linalg.norm(pos-p0, axis=1).max()),
                                   "min_z_local_m": m["min_z_local_m"], "max_speed_m_s": m.get("max_speed_m_s"),
                                   "min_det_F": m.get("min_det_F")})
            m1, p1 = material_summary(env, task)
            d = p1-p0
            settle = {"steps": int(round(a.settle_s/task.control_dt_s)), "wall_s": time.monotonic()-t0, "initial": m0, "final": m1,
                      "centroid_drift_m": d.mean(0).tolist(), "centroid_drift_norm_m": float(np.linalg.norm(d.mean(0))),
                      "horizontal_drift_norm_m": float(np.linalg.norm(d.mean(0)[:2])),
                      "max_particle_displacement_m": float(np.linalg.norm(d, axis=1).max()),
                      "bottom_layer_z_change_m": float(p1[p0[:, 2] < p0[:, 2].min()+1e-6, 2].mean()-p0[:, 2].min()),
                      "series": series, "labels_changed": bool(np.any(np.asarray(env.material.debug_state()["codes"]) != 0))}
            result["settle"] = settle
            result["after_settle_cameras"] = capture_png(env, output, "after_settle", result["images"])
            write_json(output/"result.json", result)

        # ---- 3. knife-board probe (rigid-only controls)
        env.set_gripper("hold")
        env.set_external_tool_wrench(np.zeros(6))
        labels = list(task.reference_phase_labels)
        i_pre = next(i for i, label in enumerate(labels) if label.startswith("PRE_CUT"))
        knots = np.asarray(task.edge_position_knots_local_txyz, float)
        cut_orientation = np.asarray(task.reference_edge_pose_world(float(knots[i_pre+1, 0])), float)[3:]
        T = task.T_world_workpiece

        def world(local_xyz):
            return T[:3, :3]@np.asarray(local_xyz, float)+T[:3, 3]
        above = np.r_[world([a.probe_x, 0., a.probe_start_z+0.02]), cut_orientation]
        blade = blade_vertices_tool(env.tool_spec)
        # Coupled approach (cutting disabled; the runner's bounded Cartesian controller and action workspace) so the
        # cameras stay registered for a b.png-like render with the knife over the board; the press itself is rigid-only.
        approach = {"steps": 0}
        if not a.skip_settle:
            t0 = time.monotonic()
            for i in range(400):
                env.step({"mode": "cartesian_edge_pose_world", "target": above, "cut_enabled": False}, cutting=False)
                tool = np.asarray(env.proprioception()["tool_pose"], float)
                angle = float((Rotation.from_quat(tool[3:]).inv()*Rotation.from_quat(above[3:])).magnitude())
                if np.linalg.norm(tool[:3]-above[:3]) < 5e-4 and angle < np.deg2rad(.5):
                    break
            approach = {"steps": i+1, "wall_s": time.monotonic()-t0, "final_position_error_m": float(np.linalg.norm(tool[:3]-above[:3])),
                        "final_rotation_error_rad": angle, "material_labels_changed": bool(np.any(np.asarray(env.material.debug_state()["codes"]) != 0))}
            result["coupled_approach"] = approach
            result["knife_over_board_cameras"] = capture_png(env, output, "knife_over_board", result["images"])
            write_json(output/"result.json", result)
        rows = []
        state0 = env.proprioception()
        H0 = relative(state0["hand_pose"], state0["tool_body_pose_xyzw"])
        finger_roles = ("left_finger", "right_finger", "palm")

        def probe(pose, phase, commanded_local_z=None):
            q, info = env.solve_edge_ik(pose)
            state, _ = (env.rigid_probe_step(np.asarray(q, float), refresh_camera_extrinsics=True) if a.end_render
                        else env.rigid_probe_step(np.asarray(q, float)))
            trace = env.grasp_trace[-env.config.robot_substeps:]
            fixture_n, table_n, separations, sensor_fixture = [], [], [], []
            for row in trace:
                contacts = row["contacts"]
                tensor = contacts.get("tensor_forces", {})
                by_role = tensor.get("force_by_role_world_on_tool_n", {}) if isinstance(tensor, dict) else {}
                fixture_n.append(float(np.linalg.norm(by_role.get("fixture", [0., 0., 0.]))))
                table_n.append(float(np.linalg.norm(by_role.get("table", [0., 0., 0.]))))
                sensor_fixture.append(float(contacts["roles"].get("fixture", {}).get("force_magnitude_sum_n", 0.)))
                for record in tensor.get("records", []) if isinstance(tensor, dict) else []:
                    if record.get("role") == "fixture" and record.get("separation_m") is not None:
                        separations.append(float(record["separation_m"]))
            finger_force = np.mean([np.sum([np.asarray(row["contacts"]["tensor_forces"]["force_by_role_world_on_tool_n"][k], float)
                                            for k in finger_roles], axis=0) for row in trace], axis=0)
            low_z, low_point = lowest_world(blade, state["tool_body_pose_xyzw"])
            s_t, s_r = slip(H0, relative(state["hand_pose"], state["tool_body_pose_xyzw"]))
            row = {"phase": phase, "commanded_edge_local_z_m": commanded_local_z,
                   "commanded_edge_world_z_m": float(pose[2]),
                   "measured_edge_world_z_m": float(state["tool_pose"][2]),
                   "knife_lowest_world_z_m": low_z, "knife_lowest_point_world_m": low_point,
                   "knife_lowest_minus_board_top_m": low_z-board_top_world,
                   "fixture_tensor_force_n_max": max(fixture_n), "fixture_sensor_force_sum_n_max": max(sensor_fixture),
                   "table_tensor_force_n_max": max(table_n),
                   "min_fixture_contact_separation_m": min(separations) if separations else None,
                   "fixture_sensor_contact_pairs": int(sum(1 for row in trace for r in row["contacts"]["contacts"] if r["role"] == "fixture")),
                   "finger_force_on_tool_world_n": finger_force.tolist(),
                   "grasp_slip_translation_m": s_t, "grasp_slip_rotation_rad": s_r,
                   "joint_tracking_error_max_rad": float(np.abs(np.asarray(state["q"])[0]-np.asarray(q)).max()),
                   "ik_position_error_m": float(info.get("position_error_m", float("nan"))),
                   "bilateral_loaded_tool_fraction": float(np.mean([r["contacts"]["bilateral_loaded_tool_contact"] for r in trace]))}
            rows.append(row)
            return row
        t0 = time.monotonic()
        if approach["steps"] == 0:
            for pose in slerp_path(home_pose, above, .001, 1.):
                probe(pose, "transfer")
        for _ in range(25):
            probe(above, "settle_above")
        dz = a.descent_speed*task.control_dt_s
        z = a.probe_start_z
        while z > -a.press_depth+1e-12:
            z = max(-a.press_depth, z-dz)
            probe(np.r_[world([a.probe_x, 0., z]), cut_orientation], "descent", z)
        for _ in range(int(round(a.hold_s/task.control_dt_s))):
            probe(np.r_[world([a.probe_x, 0., z]), cut_orientation], "hold", z)
        while z < a.probe_start_z:
            z = min(a.probe_start_z, z+5*dz)
            probe(np.r_[world([a.probe_x, 0., z]), cut_orientation], "lift", z)
        with (output/"knife_board_probe.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, default=json_default, allow_nan=False)+"\n")
        # Board reaction on the tool, inferred quasi-statically from the measured finger forces (native tensor view):
        # F_board = -(F_fingers - F_fingers,free hover); the native tool contact sensor lists the tool-board pairs but
        # its tensor view is filtered to the fingers/palm only, so the board force is not read directly.
        free = np.mean([r["finger_force_on_tool_world_n"] for r in rows if r["phase"] == "settle_above"][5:], axis=0)
        for r in rows:
            r["inferred_board_force_on_tool_world_n"] = (-(np.asarray(r["finger_force_on_tool_world_n"])-free)).tolist()
        touch = [r for r in rows if r["phase"] in ("descent", "hold")]
        contact_rows = [r for r in touch if r["fixture_tensor_force_n_max"] > 1e-6 or r["fixture_sensor_force_sum_n_max"] > 1e-6]
        first_contact = contact_rows[0] if contact_rows else None
        result["knife_board_probe"] = {
            "probe_x_local_m": a.probe_x, "press_depth_commanded_m": a.press_depth, "descent_speed_m_s": a.descent_speed,
            "board_top_world_z_m": board_top_world, "steps": len(rows), "wall_s": time.monotonic()-t0,
            "first_contact_commanded_edge_local_z_m": None if first_contact is None else first_contact["commanded_edge_local_z_m"],
            "first_contact_knife_lowest_minus_board_top_m": None if first_contact is None else first_contact["knife_lowest_minus_board_top_m"],
            "min_knife_lowest_minus_board_top_m": float(min(r["knife_lowest_minus_board_top_m"] for r in rows)),
            "max_knife_penetration_into_board_m": float(max(0., -min(r["knife_lowest_minus_board_top_m"] for r in rows))),
            "min_fixture_contact_separation_m": min((r["min_fixture_contact_separation_m"] for r in rows
                                                     if r["min_fixture_contact_separation_m"] is not None), default=None),
            "max_fixture_contact_force_n": float(max(r["fixture_tensor_force_n_max"] for r in rows)),
            "hold_mean_fixture_contact_force_n": float(np.mean([r["fixture_tensor_force_n_max"] for r in rows if r["phase"] == "hold"])),
            "max_table_contact_force_n": float(max(r["table_tensor_force_n_max"] for r in rows)),
            "free_hover_finger_force_on_tool_world_n": free.tolist(), "tool_weight_n": float(env.tool_spec.mass_kg*9.81),
            "hold_mean_inferred_board_force_on_tool_world_n": np.mean([r["inferred_board_force_on_tool_world_n"] for r in rows
                                                                       if r["phase"] == "hold"], axis=0).tolist(),
            "max_inferred_board_normal_force_n": float(max(r["inferred_board_force_on_tool_world_n"][2] for r in touch)),
            "fixture_sensor_contact_pairs_total": int(sum(r["fixture_sensor_contact_pairs"] for r in rows)),
            "force_scope": "max_fixture_contact_force_n reads the native tensor view, which is filtered to fingers/palm (so 0 by "
                           "construction); the board reaction is the quasi-static inference above.",
            "max_grasp_slip_translation_m": float(max(r["grasp_slip_translation_m"] for r in rows)),
            "max_grasp_slip_rotation_rad": float(max(r["grasp_slip_rotation_rad"] for r in rows)),
            "final_grasp_slip_translation_m": rows[-1]["grasp_slip_translation_m"],
            "final_grasp_slip_rotation_rad": rows[-1]["grasp_slip_rotation_rad"],
            "max_joint_tracking_error_rad": float(max(r["joint_tracking_error_max_rad"] for r in touch)),
            "min_bilateral_loaded_tool_fraction": float(min(r["bilateral_loaded_tool_fraction"] for r in rows)),
            "trace": "knife_board_probe.jsonl",
            "scope": "Rigid-only PhysX controls (material not advanced). Commanded edge targets go below the rigid board top on "
                     "purpose; the measured knife must stay on the board. Lowest point = blade hull vertices on the measured "
                     "tool body pose (PhysX tool meshes: contact offset 0.2 mm, rest offset 0)."}
        if a.end_render:
            write_json(output/"result.json", result)
            depth = {}
            result["end_cameras"] = capture_png(env, output, "after_probe", result["images"], depth)
            result["end_camera_depth"] = depth
            result["start_vs_end_camera_translation_m"] = {
                name: float(np.linalg.norm(np.asarray(depth[name]["T_world_camera_translation_m"])
                                           - np.asarray(result["start_camera_depth"][name]["T_world_camera_translation_m"])))
                for name in depth if name in result["start_camera_depth"]}
            result["end_camera_runtime_extrinsic_checks"] = getattr(env, "camera_runtime_extrinsic_checks", None)
            result["end_cameras_scope"] = ("rigid_probe_step(refresh_camera_extrinsics=True) recomposed the wrist extrinsics "
                                           "from the measured hands after every rigid step; capture() ran its own checks")
        else:
            result["end_cameras"] = ("NOT_RUN: the runner refreshes wrist-camera extrinsics only in coupled steps, so capture() after "
                                     "rigid-only controls fails its own extrinsic check (workcell_003)")
        result["status"] = "EXECUTED"
    except BaseException as error:
        result.update(status="FAIL", error=repr(error))
        (output/"failure.txt").write_text(traceback.format_exc())
    finally:
        result["wall_s"] = time.monotonic()-start_wall
        result["source_bytes_unchanged"] = {k: sha256(ROOT/k) == v for k, v in hashes.items()}
        write_json(output/"result.json", result)
        print(result["status"], output/"result.json", flush=True)
        if env is not None:
            env.close()
    return 0 if result["status"] == "EXECUTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
