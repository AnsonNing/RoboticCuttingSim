#!/usr/bin/env python3
"""Frozen initial-flatbar renderer control using the unchanged whole-task scorer.

No material dynamics, knife, trimming, boundary exclusion or task promotion.
The predeclared 3-particle interior ROI is descriptive and cannot replace the
whole fixed-footprint v1 result.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

import numpy as np

from robotic_cutting_sim.affine_surface_controls import current_volume_radii
from robotic_cutting_sim.recording import command, manifest, new_output, write_json
from robotic_cutting_sim.scene_frames import WorkpieceTarget
from robotic_cutting_sim.scene_validation import mesh_heightfield_metrics
from robotic_cutting_sim.tasks import make_task


# Committed before the first numerical output of this exact flatbar control.
PROTOCOL = {
    "protocol_id": "initial_flatbar_whole_footprint_v1_20261028",
    "seed": 42,
    "dimensions_m": [0.16, 0.03, 0.02],
    "sample_source": "CUT-01 exact full-cell local sampler; no bump or dynamics",
    "particle_spacing_m": .001,
    "density_kg_m3": 1500.,
    "target_height_m": .020,
    "total_deformation_gradient": "identity for every particle",
    "surface_profile": "current_volume_v1 H2; no changes or tuning",
    "surface_parameters": {"voxel_size": .0005, "kernel_radius": .003,
                           "threshold": .5, "kernel_scale": .5,
                           "smooth_lambda": 0., "anisotropic": False,
                           "field_smooth_iterations": 0, "mesh_smooth_iterations": 0},
    "measurement_spacing_m": .001,
    "measurement_footprint": "whole [-.080,.080] x [-.015,.015] fixed rectangle, cell-centred rays",
    "descriptive_interior_margin_over_particle_spacing": 3.,
    "tail_counts_absolute_error_thresholds_m": [.0005, .001],
    "v1_surface_limits": {
        "maximum_substrate_surface_residual_height_m": .001,
        "surface_p95_absolute_plane_error_m": .0005,
        "surface_rms_plane_error_m": .0005,
        "maximum_gouge_depth_m": .001,
        "maximum_overcut_volume_fraction_of_initial_base": .01,
        "maximum_missing_surface_fraction": .01,
    },
    "unchanged_task_protocol_sha256": "feb3a8ddb99504ccabee122615b841c844fd8a0527fc1eec713cd70c9d7013a1",
    "unchanged_affine_profile_protocol_sha256": "b9e65ca4679c3d217c1a0af83eae8656362d6f988c0b192252e523b2e83384db",
    "flatness_pass_is_not_full_task_success": True,
    "missing_rays_keep_full_target_height_deficit": True,
    "all_valid_error_tails_retained": True,
}
ROOT = Path(__file__).resolve().parents[1]


def analytic_closed_box(dimensions):
    length, width, height = dimensions
    vertices = np.array([[-length/2, -width/2, 0], [length/2, -width/2, 0],
                         [length/2, width/2, 0], [-length/2, width/2, 0],
                         [-length/2, -width/2, height], [length/2, -width/2, height],
                         [length/2, width/2, height], [-length/2, width/2, height]])
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int32)
    return vertices, faces


def sample_summary(height, valid, mask, xy):
    selected, selected_valid = mask, mask & valid
    errors = height[selected_valid]-PROTOCOL["target_height_m"]
    conservative_deficit = np.maximum(PROTOCOL["target_height_m"]-np.where(valid, height, 0.), 0.)
    finite_error = abs(errors)
    return {"sampled_rays": int(selected.sum()), "valid_rays": int(selected_valid.sum()),
            "missing_rays": int(np.sum(mask & ~valid)),
            "missing_fraction": float(np.sum(mask & ~valid)/mask.sum()),
            "p95_absolute_plane_error_m": float(np.quantile(finite_error, .95)) if len(errors) else None,
            "rms_plane_error_m": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            "maximum_absolute_plane_error_m": float(np.max(finite_error)) if len(errors) else None,
            "maximum_gouge_with_missing_ray_deficit_m": float(np.max(conservative_deficit[selected])),
            "valid_absolute_error_gt_0_5mm_rays": int(np.sum(finite_error > .0005)),
            "valid_absolute_error_gt_1mm_rays": int(np.sum(finite_error > .001)),
            "ray_xy_bounds_m": [xy[selected].min(0), xy[selected].max(0)],
            "error_trimming_applied": False}


def evaluate(vertices, triangles, target, geometry, output, name):
    measured = mesh_heightfield_metrics(vertices, triangles, target, geometry, PROTOCOL["measurement_spacing_m"])
    height, valid = measured.pop("measured_heightfield_local_m_debug"), measured.pop("valid_heightfield_debug")
    nx, ny = measured["grid_shape_xy"]
    sx, sy = measured["measurement_spacing_xy_m"]
    x, y = (np.arange(nx)+.5)*sx-geometry.length_m/2, (np.arange(ny)+.5)*sy-geometry.width_m/2
    xy = np.stack(np.meshgrid(x, y, indexing="xy"), -1)
    margin = PROTOCOL["descriptive_interior_margin_over_particle_spacing"]*PROTOCOL["particle_spacing_m"]
    interior = ((abs(xy[..., 0]) <= geometry.length_m/2-margin)
                & (abs(xy[..., 1]) <= geometry.width_m/2-margin))
    criteria = {name: measured[key] is not None and measured[key] <= limit for name, key, limit in
                [(key, "overcut_volume_fraction_of_initial_base" if key == "maximum_overcut_volume_fraction_of_initial_base"
                  else "missing_surface_fraction" if key == "maximum_missing_surface_fraction" else key, value)
                 for key, value in PROTOCOL["v1_surface_limits"].items()]}
    deficit = np.maximum(PROTOCOL["target_height_m"]-np.where(valid, height, 0.), 0.)
    maximum = np.max(deficit)
    worst = np.argwhere(deficit == maximum)
    tails = np.argwhere(valid & (abs(height-PROTOCOL["target_height_m"]) > .001))
    def ray(row, column):
        return {"row": int(row), "column": int(column), "xy_m": xy[row, column],
                "measured_height_m": float(height[row, column]) if valid[row, column] else None,
                "valid": bool(valid[row, column]), "conservative_gouge_m": float(deficit[row, column]),
                "distance_to_nearest_outer_edge_m": float(min(geometry.length_m/2-abs(xy[row, column, 0]),
                                                                geometry.width_m/2-abs(xy[row, column, 1])))}
    receipt = {"name": name, "whole_footprint_metrics": measured,
               "unchanged_v1_surface_checks": criteria,
               "unchanged_v1_surface_status": "PASS_SURFACE_ONLY" if all(criteria.values()) else "FAIL_SURFACE_ONLY",
               "whole_footprint_full_tail_summary": sample_summary(height, valid, np.ones_like(valid), xy),
               "interior_3_spacing_descriptive_only": sample_summary(height, valid, interior, xy),
               "interior_roi_promotes_task": False,
               "worst_gouge_ray_tie_count": int(len(worst)),
               "first_worst_gouge_ray": ray(*worst[0]),
               "all_valid_error_gt_1mm_rays": [ray(i, j) for i, j in tails]}
    np.savez_compressed(output/(name+".npz"), vertices_m=vertices, triangles=triangles,
                        measured_height_m=height, valid_mask=valid, ray_xy_m=xy,
                        interior_roi_debug=interior, conservative_gouge_m=deficit)
    write_json(output/(name+".json"), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda:0", choices=["cuda:0"])
    args = parser.parse_args()
    output = new_output(args.output_dir)
    protocol_sha = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
    write_json(output/"protocol.json", PROTOCOL)
    write_json(output/"manifest.json", manifest({"protocol_sha256": protocol_sha, "device": args.device}, "initial_flatbar_surface_geometry_control"))
    write_json(output/"gpu_before.json", command(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"]))
    implementation = output/"implementation"
    implementation.mkdir()
    for path in (Path(__file__), ROOT/"configs/cutting_tasks_v1.json", ROOT/"configs/particle_surface_affine_v1.json",
                 ROOT/"src/robotic_cutting_sim/scene_validation.py", ROOT/"src/robotic_cutting_sim/cutting_surface.py",
                 ROOT/"src/robotic_cutting_sim/affine_surface_controls.py", ROOT/"src/robotic_cutting_sim/tasks.py",
                 ROOT/".deps/newton/newton/_src/geometry/particle_surface.py",
                 ROOT/".deps/newton/newton/_src/geometry/particle_surface_kernels.py"):
        (implementation/path.name).write_bytes(path.read_bytes())
    write_json(output/"result.json", {"status": "NOT_RUN", "protocol_sha256": protocol_sha})
    start = time.perf_counter()
    try:
        import warp as wp
        from newton.geometry import ParticleSurface

        assert importlib.metadata.version("newton") == "1.6.1"
        assert importlib.metadata.version("warp-lang") == "1.17.0"
        for file, field in (("cutting_tasks_v1.json", "unchanged_task_protocol_sha256"),
                            ("particle_surface_affine_v1.json", "unchanged_affine_profile_protocol_sha256")):
            assert hashlib.sha256((ROOT/"configs"/file).read_bytes()).hexdigest() == PROTOCOL[field]
        task = make_task("CUT-01", seed=PROTOCOL["seed"])
        geometry = task.geometry
        assert np.array_equal([geometry.length_m, geometry.width_m, geometry.base_height_m], PROTOCOL["dimensions_m"])
        assert not geometry.bumps
        target = WorkpieceTarget(height_m=PROTOCOL["target_height_m"], edge_clearance_m=0.)
        points, mass, lattice = task.sample_particles()
        total_F = np.broadcast_to(np.eye(3), (len(points), 3, 3))
        radii = current_volume_radii(PROTOCOL["particle_spacing_m"], total_F).astype(np.float32)
        np.savez_compressed(output/"initial_particles.npz", positions_local_m=points, mass_kg=mass,
                            lattice=lattice, total_F=total_F, radii_m=radii)
        parameters = PROTOCOL["surface_parameters"]
        wp.init()
        with wp.ScopedDevice(args.device):
            surface = ParticleSurface(**parameters, device=args.device)
            tick = time.perf_counter()
            vertices, triangles, _ = surface.extract(wp.array(points, dtype=wp.vec3, device=args.device),
                                                     wp.array(radii, dtype=float, device=args.device))
            wp.synchronize_device(args.device)
            extraction_s = time.perf_counter()-tick
            vertices, triangles = vertices.numpy().copy(), triangles.numpy().reshape(-1, 3).copy()
        h2 = evaluate(vertices, triangles, target, geometry, output, "h2_initial_flatbar")
        analytic_vertices, analytic_faces = analytic_closed_box(PROTOCOL["dimensions_m"])
        analytic = evaluate(analytic_vertices, analytic_faces, target, geometry, output, "analytic_exact_flatbar")
        result = {"status": "GEOMETRY_DIAGNOSTIC_COMPLETED", "protocol_sha256": protocol_sha,
                  "h2_surface_v1_status": h2["unchanged_v1_surface_status"],
                  "analytic_surface_v1_status": analytic["unchanged_v1_surface_status"],
                  "particles": len(points), "initial_mass_kg": float(mass.sum()),
                  "particle_spacing_m": PROTOCOL["particle_spacing_m"], "target_height_m": target.height_m,
                  "minimum_radius_m": float(radii.min()), "maximum_radius_m": float(radii.max()),
                  "extraction_wall_s_including_compile": extraction_s, "total_wall_s": time.perf_counter()-start,
                  "exclusive_timing": False, "knife_present": False, "material_dynamics_steps": 0,
                  "initial_mechanical_continuum_prescribed_by_control": True,
                  "actual_trim_task_promoted": False, "full_footprint_task_gates_unchanged": True,
                  "scope": "Initial flatbar renderer boundary floor, not physical knife gouging or trimming"}
        write_json(output/"result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception:
        write_json(output/"result.json", {"status": "ERROR_PRESERVED", "protocol_sha256": protocol_sha,
                   "wall_s": time.perf_counter()-start, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
