#!/usr/bin/env python3
"""Build a cut-independent frozen represented target and negative scorer controls."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from robotic_cutting_sim.affine_surface_controls import current_volume_radii
from robotic_cutting_sim.benchmark import HeightFieldSpec
from robotic_cutting_sim.recording import command, manifest, new_output, write_json
from robotic_cutting_sim.represented_target import compare_represented_columns
from robotic_cutting_sim.scene_frames import WorkpieceTarget
from robotic_cutting_sim.scene_validation import mesh_heightfield_metrics
from robotic_cutting_sim.surface_controls import box_particles

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--protocol", type=Path, default=ROOT/"configs/represented_flatbar_target_v1.json")
    p.add_argument("--device", default="cuda:0", choices=["cuda:0"])
    args = p.parse_args()
    output = new_output(args.output_dir)
    protocol = json.loads(args.protocol.read_text())
    sha = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    (output/"protocol.json").write_bytes(args.protocol.read_bytes())
    write_json(output/"manifest.json", manifest({"protocol_sha256": sha, "device": args.device}, "independent_represented_flatbar_target"))
    write_json(output/"gpu_before.json", command(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"]))
    source = output/"implementation"; source.mkdir()
    for path in [Path(__file__), args.protocol]+[ROOT/"src/robotic_cutting_sim"/name for name in
            ("represented_target.py", "scene_validation.py", "surface_controls.py", "affine_surface_controls.py")]:
        (source/path.name).write_bytes(path.read_bytes())
    start = time.perf_counter()
    cfg, measurement = protocol["target"], protocol["measurement"]
    assert importlib.metadata.version("newton") == protocol["newton_version"]
    assert importlib.metadata.version("warp-lang") == protocol["warp_version"]
    import warp as wp
    from newton.geometry import ParticleSurface

    dimensions = cfg["dimensions_m"]
    case = {"dimensions_m": dimensions, "euler_xyz_rad": [0., 0., 0.], "translation_m": [0., 0., dimensions[2]/2]}
    points, masses = box_particles(case, cfg["particle_spacing_m"], cfg["density_kg_m3"])
    F = np.broadcast_to(np.eye(3), (len(points), 3, 3))
    radii = current_volume_radii(cfg["particle_spacing_m"], F).astype(np.float32)
    wp.init()
    with wp.ScopedDevice(args.device):
        surface = ParticleSurface(**cfg["surface_parameters"], device=args.device)
        tick = time.perf_counter()
        vertices, faces, _ = surface.extract(wp.array(points, dtype=wp.vec3, device=args.device),
                                            wp.array(radii, dtype=float, device=args.device))
        wp.synchronize_device(args.device)
        extraction_s = time.perf_counter()-tick
        vertices, faces = vertices.numpy().copy(), faces.numpy().reshape(-1, 3).copy()
    geometry = HeightFieldSpec(*dimensions, bumps=())
    target = WorkpieceTarget(height_m=dimensions[2], edge_clearance_m=0.)
    plane = mesh_heightfield_metrics(vertices, faces, target, geometry, measurement["spacing_m"])
    height, valid = plane.pop("measured_heightfield_local_m_debug"), plane.pop("valid_heightfield_debug")
    assert plane["grid_shape_xy"] == measurement["grid_shape_xy"] and height.size == measurement["columns"]
    path = output/"fixed_target.npz"
    np.savez_compressed(path, positions_local_m=points, mass_kg=masses, total_F=F, radii_m=radii,
                        vertices_local_m=vertices, triangles=faces, height_local_m=height, valid_mask=valid)
    write_json(output/"original_analytic_plane_metrics.json", plane)
    kwargs = dict(spacing_xy_m=plane["measurement_spacing_xy_m"], initial_base_volume_m3=float(np.prod(dimensions)),
                  limits=protocol["shape_limits"])
    cases = {"exact_target": (height.copy(), valid.copy())}
    for name in ("positive_vertical_offset_m", "negative_vertical_offset_m"):
        cases[name] = (height+protocol["synthetic_controls"][name], valid.copy())
    missing = protocol["synthetic_controls"]["one_actual_missing_ray"]
    missing_mask = valid.copy()
    assert missing_mask[missing["row"], missing["column"]]
    missing_mask[missing["row"], missing["column"]] = False
    cases["one_actual_missing_ray"] = (height.copy(), missing_mask)
    controls = {}
    for name, (actual, actual_valid) in cases.items():
        receipt, debug = compare_represented_columns(actual, actual_valid, height, valid, **kwargs)
        controls[name] = receipt
        np.savez_compressed(output/(name+".npz"), **debug)
    passed = (controls["exact_target"]["status"] == "PASS_REPRESENTED_SHAPE"
              and all(value["status"] == "FAIL_REPRESENTED_SHAPE" for key, value in controls.items() if key != "exact_target"))
    write_json(output/"result.json", {"status": "PASS_FROZEN_GEOMETRY_CONTROLS" if passed else "FAIL_GEOMETRY_CONTROLS",
        "protocol_sha256": sha, "target_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "target_particles": len(points), "mass_kg": float(masses.sum()), "original_analytic_plane_metrics": plane,
        "controls": controls, "extraction_wall_s": extraction_s, "wall_s": time.perf_counter()-start,
        "exclusive_timing": False, "cut_output_used_for_target": False, "physical_task_success_claimed": False,
        "target_alignment_fitted": False, "actual_runtime_or_v1_promoted": False})
    print(json.dumps({"controls_passed": passed, "target_missing_columns": int((~valid).sum()),
                      "target_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2), flush=True)
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
