#!/usr/bin/env python3
"""Run the single frozen 2mm-grid TRIM physics-only branch and retain failures."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from robotic_cutting_sim.coupled_robot import CouplingConfig
from robotic_cutting_sim.cut_topology import mechanical_components
from robotic_cutting_sim.cutting_surface import ComponentSurface, wedge_mesh
from robotic_cutting_sim.live_material import LiveCuttingMaterial
from robotic_cutting_sim.recording import command, manifest, new_output, write_json
from robotic_cutting_sim.tasks import make_task

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/trim_grid_2mm_v1.json")
    parser.add_argument("--device", choices=["cuda:0"], default="cuda:0")
    args = parser.parse_args()
    output = new_output(args.output_dir)
    protocol = json.loads(args.protocol.read_text())
    protocol_sha = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    (output/"protocol.json").write_bytes(args.protocol.read_bytes())
    write_json(output/"manifest.json", manifest({"protocol_sha256": protocol_sha, "device": args.device}, "physics_only_live_material_grid_diagnostic"))
    write_json(output/"gpu_before.json", command(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader"]))
    implementation = output/"implementation"
    implementation.mkdir()
    for path in [Path(__file__), args.protocol, ROOT/"configs/cutting_tasks_v1.json"] + [ROOT/"src/robotic_cutting_sim"/name for name in
            ("live_material.py", "tlmpm.py", "fixture_contacts.py", "tasks.py", "material_config.py", "coupled_robot.py",
             "cut_topology.py", "cutting_surface.py", "scene_validation.py", "benchmark.py", "scene_frames.py", "affine_surface_controls.py")]:
        (implementation/path.name).write_bytes(path.read_bytes())
    write_json(output/"result.json", {"status": "NOT_RUN", "protocol_sha256": protocol_sha})
    start = time.perf_counter()
    try:
        assert hashlib.sha256((ROOT/"configs/cutting_tasks_v1.json").read_bytes()).hexdigest() == protocol["original_task_protocol_sha256"]
        original = make_task(protocol["task_id"], seed=protocol["seed"])
        task = replace(original, **protocol["only_explicit_change"])
        assert task.particle_spacing_m == protocol["particle_spacing_m"]
        assert task.control_dt_s == protocol["control_dt_s"] and task.duration_s == protocol["duration_s"]
        assert np.isclose(task.target_origin_local_m[2]+task.edge_clearance_m, protocol["cut_plane_local_z_m"], atol=1e-12, rtol=0)
        changed_fields = [name for name in asdict(original) if asdict(original)[name] != asdict(task)[name]]
        assert changed_fields == ["grid_dx_m"]
        material = LiveCuttingMaterial(CouplingConfig(device=args.device, dx=task.grid_dx_m,
            control_dt=task.control_dt_s, material_dt=protocol["material_dt_s"]), task=task)
        times = np.arange(protocol["steps"]+1)*task.control_dt_s
        poses = np.stack([task.reference_edge_pose_world(t) for t in times])
        reset = material.reset(poses[0], seed=protocol["seed"])
        bed, count = material.bed, len(material.points)
        original_uploaded_mass = bed.mass.numpy().copy()
        nodes, weights, gradients = bed.nodes.numpy(), bed.weights.numpy(), bed.gradients.numpy()
        _, fixture_centers, fixture_halves, table_plane = bed.fixture_data
        blade_vertices, blade_triangles = wedge_mesh(task.blade_length_m, task.blade_height_m, task.blade_half_width_m)
        np.savez_compressed(output/"fixture_tool_geometry.npz", fixture_centers_local_m=fixture_centers.numpy(),
            fixture_half_extents_local_m=fixture_halves.numpy(), table_plane_local=np.asarray(list(table_plane)),
            blade_vertices_tool_m=blade_vertices, blade_triangles=blade_triangles, particle_contact_radius_m=bed.radius)
        np.savez_compressed(output/"reference.npz", reference_positions=material.world_points,
            reference_positions_local=material.points, mass_kg=material.masses, uploaded_mass_kg=original_uploaded_mass,
            reference_indices=material.lattice, reference_stencil_nodes=nodes,
            reference_stencil_weights=weights, reference_stencil_gradients=gradients,
            measured_tool_pose_world_xyzw=poses, time_s=times)
        write_json(output/"setup.json", {"original_task": original.to_dict(), "actual_task": task.to_dict(),
            "changed_fields": changed_fields, "material_reset": reset, "actual_material_config": asdict(material.material_config),
            "nominal_bulk_particles_per_reference_cell": int(round((task.grid_dx_m/task.particle_spacing_m)**3)),
            "grid_origin_local_m": bed.grid_origin, "grid_dimensions": bed.grid_dims,
            "reference_node_count": bed.node_count, "derived_dx_scales_m": protocol["derived_dx_scales_m"],
            "zero_robot_dynamics_steps": True, "zero_isaac_runtime_steps": True,
            "adapter_feedback_fields_scope": "Generic previous-wrench fields are proposals for a caller; no robot exists here and no wrench is actually applied to an articulation"})
        positions = np.lib.format.open_memmap(output/"positions.npy", mode="w+", dtype=np.float32, shape=(len(times), count, 3))
        velocities = np.lib.format.open_memmap(output/"velocities.npy", mode="w+", dtype=np.float32, shape=positions.shape)
        codes = np.lib.format.open_memmap(output/"codes.npy", mode="w+", dtype=np.int32, shape=(len(times), count))
        initial = material.debug_state()
        positions[0], velocities[0], codes[0] = initial["positions"], initial["velocities"], initial["codes"]
        initial_components, _ = mechanical_components(nodes, weights, codes[0], 1, gradients=gradients)
        trace, diagnostics, guard_failure, completed = [], [], None, 0
        for step in range(1, len(times)):
            tick = time.perf_counter()
            diagnostic = material.advance(poses[step], float(times[step]), cutting=True)
            data = material.debug_state()
            positions[step], velocities[step], codes[step] = data["positions"], data["velocities"], data["codes"]
            determinant = np.linalg.det(data["F"])
            speed = np.linalg.norm(data["velocities"], axis=-1)
            finite = bool(all(np.isfinite(data[key]).all() for key in data)
                          and np.isfinite(diagnostic["current_material_reaction_impulse_world"]).all()
                          and np.isfinite(diagnostic["material_contact_statistics"]).all())
            row = {"step": step, "time_s": float(times[step]), "finite": finite,
                   "minimum_det_total_F": float(determinant.min()) if np.isfinite(determinant).all() else None,
                   "maximum_det_total_F": float(determinant.max()) if np.isfinite(determinant).all() else None,
                   "maximum_speed_m_s": float(speed.max()) if np.isfinite(speed).all() else None,
                   "code_counts": np.bincount(data["codes"], minlength=3),
                   "particle_count": len(data["positions"]), "reaction_impulse_world_Ns_Nms": diagnostic["current_material_reaction_impulse_world"],
                   "reaction_wrench_world_N_Nm": diagnostic["next_material_wrench_world"],
                   "contact_statistics": diagnostic["material_contact_statistics"],
                   "cut_index": diagnostic["cut_index"], "control_wall_s": time.perf_counter()-tick}
            trace.append(row); diagnostics.append(diagnostic); completed = step
            if step % 25 == 0:
                print(f"PHYSICS_GRID2 {step}/{protocol['steps']} t={times[step]:.2f} codes={row['code_counts'].tolist()} Jmin={row['minimum_det_total_F']}", flush=True)
                write_json(output/"trace_partial.json", trace)
                write_json(output/"phase.json", {"phase": "PHYSICS_ONLY_STEPPING", **row})
                positions.flush(); velocities.flush(); codes.flush()
            if not finite or determinant.min() <= 0 or speed.max() > protocol["abort_guards"]["maximum_speed_m_s"]:
                guard_failure = "Nonfinite state, nonpositive total-F Jacobian or frozen5m/s abort guard"
                break
        positions.flush(); velocities.flush(); codes.flush()
        final = material.debug_state()
        np.savez_compressed(output/"final_state.npz", **final, final_uploaded_mass_kg=bed.mass.numpy(),
                            actual_final_particle_count=bed.n, sampled_control_boundaries=completed+1)
        write_json(output/"trace.json", trace)
        write_json(output/"diagnostics.json", diagnostics)
        actual_times, actual_codes = times[:completed+1], codes[:completed+1]
        actual_positions = positions[:completed+1]
        final_components, labels = mechanical_components(nodes, weights, final["codes"], 1, gradients=gradients)
        np.savez_compressed(output/"final_components.npz", mechanical_component_ids=labels,
                            component_mass_kg=np.bincount(labels, weights=material.masses))
        cache = {final["codes"].tobytes(): final_components}
        post = np.flatnonzero(actual_times >= protocol["fully_withdrawn_time_s"]-1e-9)
        post_counts = []
        for index in post:
            key = actual_codes[index].tobytes()
            if key not in cache:
                cache[key] = mechanical_components(nodes, weights, actual_codes[index], 1, gradients=gradients)[0]
            post_counts.append(cache[key])
        persistence = 0. if len(post) < 2 else float(actual_times[post[-1]]-actual_times[post[0]])
        recording = dict(time_s=actual_times, positions=actual_positions, codes=actual_codes,
            reference_positions=material.world_points, mass_kg=material.masses,
            reference_stencil_nodes=nodes, reference_stencil_weights=weights, reference_stencil_gradients=gradients)
        if guard_failure is None:
            surface = ComponentSurface(task.particle_spacing_m, args.device, task.particle_spacing_m/2, profile="current_volume_v1")
            vertices, triangles, info = surface.extract(bed)
            face_labels = np.full(len(triangles), -1, np.int32)
            for r in info["component_ranges"]:
                face_labels[r["first_triangle"]:r["first_triangle"]+r["triangles"]] = r["mechanical_component"]
            np.savez_compressed(output/"final_surface.npz", vertices_world=vertices, triangles=triangles,
                triangle_component_ids=face_labels, physics_timestamp_s=actual_times[-1])
            write_json(output/"final_surface.json", info)
            recording.update(surface_vertices_world=vertices, surface_triangles=triangles,
                surface_triangle_component_ids=face_labels, surface_component_provenance=info["partition_rule"],
                surface_timestamp_s=actual_times[-1])
        evaluation = task.evaluate_oracle(recording)
        write_json(output/"task_evaluation.json", evaluation)
        removed = evaluation["metrics"].get("removed_reference_above_plane_mass_fraction")
        criteria = protocol["physical_candidate_criteria"]
        checks = {"complete_original_8s_episode": completed == protocol["steps"],
            "initial_actual_stencil_continuity": initial_components == criteria["initial_actual_stencil_components"],
            "actual_stencil_disconnected": final_components >= criteria["minimum_final_actual_stencil_components"],
            "above_plane_mass_detached": removed is not None and removed >= criteria["minimum_detached_initial_above_plane_mass_fraction"],
            "disconnected_topology_persistent_after_full_withdrawal": bool(post_counts) and min(post_counts) >= 2 and persistence >= criteria["minimum_post_withdraw_topology_persistence_s"],
            "all_control_boundary_states_finite": all(row["finite"] for row in trace),
            "positive_total_F_Jacobian": all(row["minimum_det_total_F"] is not None and row["minimum_det_total_F"] > 0 for row in trace),
            "no_existing_digit_rewrites": evaluation["metrics"]["nonzero_digit_rewrites"] == 0,
            "fixed_particle_count": all(row["particle_count"] == count for row in trace),
            "uploaded_mass_vector_exactly_unchanged": bool(np.array_equal(original_uploaded_mass, bed.mass.numpy())),
            "no_abort_guard_failure": guard_failure is None}
        passed = all(checks.values())
        result = {"status": "PASS_PHYSICS_ONLY_CANDIDATE" if passed else "FAIL_PHYSICS_ONLY_CANDIDATE",
            "protocol_sha256": protocol_sha, "checks": checks, "physical_candidate": passed,
            "full_unchanged_task_v1_status": evaluation["status"], "guard_failure": guard_failure,
            "steps": completed, "simulation_s": float(actual_times[-1]), "particle_count": count,
            "initial_components": initial_components, "final_components": final_components,
            "detached_reference_above_plane_mass_fraction": removed,
            "minimum_post_withdraw_component_count": min(post_counts) if post_counts else None,
            "post_withdraw_persistence_measured_s": persistence,
            "minimum_det_total_F": min(row["minimum_det_total_F"] for row in trace if row["minimum_det_total_F"] is not None),
            "maximum_speed_m_s": max(row["maximum_speed_m_s"] for row in trace if row["maximum_speed_m_s"] is not None),
            "maximum_material_reaction_N": float(max(np.linalg.norm(row["reaction_wrench_world_N_Nm"][:3]) for row in trace)),
            "post_full_withdraw_tool_contact_substep_pairs": int(sum(row["contact_statistics"][1] for row in trace if row["time_s"] > protocol["fully_withdrawn_time_s"]+1e-9)),
            "wall_s": time.perf_counter()-start, "exclusive_timing": False,
            "actual_isaac_runtime_episode": False, "robot_dynamics_steps": 0, "sensor_frames": 0,
            "bidirectional_articulation_coupling": False,
            "runtime_or_default_promoted": False, "negative_grid_branch_stopped": not passed,
            "component_provenance": "Actual reference eight-node compatibility including code-zero bridges and nonzero stress gradients",
            "scope": protocol["runtime_scope"], "derived_dx_scale_caveat": protocol["derived_scale_boundary"]}
        write_json(output/"result.json", result)
        write_json(output/"phase.json", {"phase": "COMPLETE_RETAINED", "status": result["status"]})
        print(json.dumps(result, indent=2), flush=True)
        return int(not passed)
    except Exception:
        write_json(output/"result.json", {"status": "ERROR_PRESERVED", "protocol_sha256": protocol_sha,
                   "wall_s": time.perf_counter()-start, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
