#!/usr/bin/env python3
"""Measure real Panda/TLMPM partitioned coupling, cutting and force feedback."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from robotic_cutting_sim.coupled_robot import CoupledCutting, CouplingConfig, reference_pose
from robotic_cutting_sim.benchmark import BenchmarkSpec
from robotic_cutting_sim.cut_topology import mechanical_components
from robotic_cutting_sim.recording import manifest, new_output, write_json
from robotic_cutting_sim.robot import wedge_mesh


def components(bed, codes):
    return mechanical_components(bed.nodes.numpy(), bed.weights.numpy(), codes, 1,
                                 gradients=bed.gradients.numpy())


def run(output, config, seed, duration):
    scene = CoupledCutting(config)
    start = time.perf_counter()
    proprio, setup = scene.reset(seed)
    record = manifest({"coupling": asdict(config), "seed": seed, "duration_s": duration}, "experimental_panda_tlmpm")
    root = Path(__file__).resolve().parents[1]
    sources = [root / "src/robotic_cutting_sim" / name for name in ("coupled_robot.py", "robot.py", "tlmpm.py", "cut_topology.py")]
    record["implementation_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (output / "implementation").mkdir()
    for source in sources:
        shutil.copyfile(source, output / "implementation" / source.name)
    write_json(output / "manifest.json", record)
    write_json(output / "setup.json", setup)
    initial_components, _ = components(scene.bed, np.zeros(len(scene.points), np.int32))
    steps = round(duration / config.control_dt)
    states, diagnostics, body_poses = [proprio], [], [scene.robot.state.body_q.numpy().copy()]
    material = [scene.debug_material_state()]
    snapshot_times = [0.]
    trace, failed = [], None
    for step in range(steps):
        proprio, info = scene.step()
        states.append(proprio)
        diagnostics.append(info)
        body_poses.append(scene.robot.state.body_q.numpy().copy())
        q, v = scene.bed.x.numpy(), scene.bed.v.numpy()
        determinant = np.linalg.det(scene.bed.F.numpy())
        finite = bool(np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(determinant).all())
        row = {"time_s": scene.time_s, "max_speed_m_s": float(np.linalg.norm(v, axis=1).max()),
               "min_detF": float(determinant.min()), "max_detF": float(determinant.max()),
               "min_z_m": float(q[:, 2].min()),
               "tool_position_error_m": float(np.linalg.norm(proprio["tool_pose"][:3] - info["desired_tool_pose"][:3])),
               "material_force_norm_N": float(np.linalg.norm(info["next_material_wrench_world"][:3])),
               "applied_force_norm_N": float(np.linalg.norm(info["applied_previous_material_wrench_world"][:3])),
               "tool_contacts": int(info["material_contact_statistics"][1]),
               "material_contacts": int(info["material_contact_statistics"][4]),
               "max_tool_penetration_m": float(info["material_contact_statistics"][3]),
               "max_material_overlap_m": float(info["material_contact_statistics"][6]), "finite": finite}
        trace.append(row)
        if (step + 1) % 5 == 0 or step == steps - 1:
            material.append(scene.debug_material_state())
            snapshot_times.append(scene.time_s)
        if (step + 1) % 50 == 0:
            print(f"feedback={config.feedback} step={step+1}/{steps} t={scene.time_s:.2f} "
                  f"F={row['material_force_norm_N']:.3g}N detF={row['min_detF']:.3f} "
                  f"error={row['tool_position_error_m']*1000:.3f}mm", flush=True)
        if not finite or row["min_detF"] < .05 or row["max_speed_m_s"] > 5 or row["min_z_m"] < -.013:
            failed = "Nonfinite, inverted/extreme F, speed>5m/s, or table penetration>3mm"
            if snapshot_times[-1] != scene.time_s:
                material.append(scene.debug_material_state()); snapshot_times.append(scene.time_s)
            break
    final = scene.debug_material_state()
    count, component = components(scene.bed, final["codes"])
    states_q = np.array([s["q"] for s in states])
    tools = np.array([s["tool_pose"] for s in states])
    times = np.array([s["timestamp_s"] for s in states])
    force = np.array([d["next_material_wrench_world"] for d in diagnostics])
    applied = np.array([d["applied_previous_material_wrench_world"] for d in diagnostics])
    reaction = np.array([d["current_material_reaction_impulse_world"] for d in diagnostics])
    np.savez_compressed(output / "robot_state.npz", time_s=times, q=states_q,
        qdot=np.array([s["qdot"] for s in states]), body_pose=body_poses, tool_pose_xyzw=tools,
        desired_tool_pose_xyzw=np.array([reference_pose(t, config.cut_height_m) for t in times]),
        next_material_wrench_world=force, applied_previous_material_wrench_world=applied,
        material_reaction_impulse_world=reaction)
    from check_robot_control import export_replay
    replay = export_replay(scene.robot, times, np.array(body_poses), tools, output / "dual_franka_replay.usda")
    np.savez_compressed(output / "physics_state.npz", time_s=snapshot_times,
        positions=np.array([s["positions"] for s in material]), velocities=np.array([s["velocities"] for s in material]),
        codes=np.array([s["codes"] for s in material]), mass_kg=scene.masses,
        reference_positions=scene.points, reference_indices=scene.lattice,
        final_F=final["F"], final_Fpi=final["Fpi"], final_cut_progress=final["cut_progress"],
        final_plastic_strain=final["plastic_strain"], component_ids=component,
        component_provenance="final conservative MPM reference-stencil compatibility graph, including derivative support")
    vertices, faces = wedge_mesh()
    np.savez_compressed(output / "tool_geometry.npz", vertices_tool=vertices, triangle_indices=faces)
    write_json(output / "diagnostics.json", diagnostics)
    with (output / "trace.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0])); writer.writeheader(); writer.writerows(trace)
    labels = np.array([s["codes"] for s in material])
    digit_changed = bool(((labels[:-1] != 0) & (labels[1:] != labels[:-1])).any())
    masses = np.bincount(component, weights=scene.masses)
    component_stats = []
    for index, mass in enumerate(masses):
        selected = component == index
        com = np.average(np.array([s["positions"][selected] for s in material]), axis=1,
                         weights=scene.masses[selected])
        withdrew = np.flatnonzero(np.array(snapshot_times) >= 6.4 - 1e-8)
        component_stats.append({"id": index, "particles": int(selected.sum()), "mass_kg": float(mass),
            "initial_com_m": com[0], "final_com_m": com[-1],
            "post_withdraw_motion_m": float(np.linalg.norm(com[-1] - com[withdrew[0]])) if len(withdrew) else None})
    impulse_error = float(np.max(np.abs(applied[1:] * config.control_dt - reaction[:-1]))) if config.feedback else None
    active_depth = tools[(times >= 2.) & (times <= 5.4), 2]
    material_work = float(sum(d["material_contact_statistics"][2] for d in diagnostics))
    robot_work = float(sum(d["applied_tool_wrench_work_j"] for d in diagnostics))
    result = {"status": "FAIL" if failed else "PASS_NUMERICAL_COUPLING_PROBE",
        "duration_s": scene.time_s, "wall_s": time.perf_counter() - start,
        "scope": "experimental partitioned articulated robot/TLMPM, not Isaac Lab",
        "particle_count": len(scene.points), "initial_mechanical_components": initial_components,
        "final_mechanical_components": count, "components": component_stats,
        "persistent_split_observed": initial_components == 1 and count >= 2 and not digit_changed,
        "existing_side_digit_changed": digit_changed,
        "minimum_detF": min(t["min_detF"] for t in trace),
        "maximum_speed_m_s": max(t["max_speed_m_s"] for t in trace),
        "max_tool_tracking_error_m": max(t["tool_position_error_m"] for t in trace),
        "active_cut_depth_range_m": [float(active_depth.min()), float(active_depth.max())] if len(active_depth) else None,
        "maximum_material_force_N": float(np.linalg.norm(force[:, :3], axis=1).max()),
        "feedback_enabled": config.feedback, "feedback_delay_s": config.control_dt,
        "applied_impulse_previous_interval_max_error": impulse_error,
        "mean_rigid_physics_wall_s": float(np.mean([d["physics_wall_s"] for d in diagnostics])),
        "mean_ik_wall_s": float(np.mean([d["ik_wall_s"] for d in diagnostics])),
        "mean_material_physics_wall_s": float(np.mean([d["material_wall_s"] for d in diagnostics])),
        "mass_kg": float(scene.masses.sum()), "mass_exchange": "none; immutable IDs/masses, no deletion/spawn",
        "proprioception_keys": list(states[0]), "guard_failure": failed,
        "material_geometry_matches_robot_wedge": True,
        "replay": replay,
        "interface_work_ledger": {"material_boundary_work_j": material_work,
            "robot_applied_material_wrench_work_j": robot_work,
            "sum_interface_work_j": material_work + robot_work,
            "scope": "boundary-work diagnostic under explicit lag; not a converged total-energy conservation claim"},
        "material_calibration": "provisional_uncalibrated", "physics_proven": False,
        "limitations": ["20ms explicit feedback delay; not monolithic, no converged energy balance",
            "Torque impulse accumulated about moving tool points; inter-interval lever-arm approximation",
            "No Isaac Lab runtime or cameras in this isolated probe",
            "Material contact resolution and cutting constitutive law remain experimental"]}
    result["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.npz")}
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=8.)
    parser.add_argument("--variant", choices=["D0", "D1", "D2", "D3"], default="D0")
    parser.add_argument("--benchmark-recipe", action="store_true", help="Explicit BenchmarkSpec recipe, including its supported material values")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-feedback", action="store_true")
    mode.add_argument("--compare-feedback", action="store_true")
    args = parser.parse_args()
    output = new_output(args.output_dir)
    config = CouplingConfig(device=args.device, feedback=not args.no_feedback, variant=args.variant,
        benchmark_spec=BenchmarkSpec.generate(args.seed, args.variant) if args.benchmark_recipe else None)
    result = run(output, config, args.seed, args.duration)
    print(json.dumps({k: v for k, v in result.items() if k != "components"}, indent=2), flush=True)
    failed = result["status"] == "FAIL"
    if args.compare_feedback:
        from dataclasses import replace
        other = new_output(output / "feedback_disabled")
        control = run(other, replace(config, feedback=False), args.seed, args.duration)
        a, b = np.load(output / "robot_state.npz"), np.load(other / "robot_state.npz")
        same = a["q"].shape == b["q"].shape
        difference = float(np.max(np.abs(a["q"] - b["q"]))) if same else None
        comparison = {"status": "PASS" if same and difference > 1e-8 and result["maximum_material_force_N"] > 0 else "FAIL",
            "scope": "matched commands/seed with material reaction feedback toggled; no training",
            "q_max_difference_rad": difference,
            "tool_position_max_difference_m": float(np.linalg.norm(a["tool_pose_xyzw"][:, :3] - b["tool_pose_xyzw"][:, :3], axis=1).max()) if same else None,
            "enabled_final_components": result["final_mechanical_components"],
            "disabled_final_components": control["final_mechanical_components"]}
        write_json(output / "feedback_comparison.json", comparison)
        print(json.dumps(comparison, indent=2), flush=True)
        failed |= comparison["status"] == "FAIL"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
