#!/usr/bin/env python3
"""Run frozen affine/current-volume ParticleSurface controls, without cut data."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import warp as wp

from robotic_cutting_sim.recording import manifest,new_output,write_json
from robotic_cutting_sim.affine_surface_controls import affine_particles,affine_mesh_measurements,current_volume_radii,tlmpm_total_F_receipt


def main():
    from newton.geometry import ParticleSurface
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir",type=Path)
    p.add_argument("--protocol",type=Path,default=Path("configs/particle_surface_affine_v1.json"))
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    output=new_output(args.output_dir)
    protocol=json.loads(args.protocol.read_text())
    protocol_hash=hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    (output/"protocol.json").write_bytes(args.protocol.read_bytes())
    assert importlib.metadata.version("newton")==protocol["newton_version"]
    assert importlib.metadata.version("warp-lang")==protocol["warp_version"]
    write_json(output/"manifest.json",manifest({"device":args.device,"protocol_sha256":protocol_hash},"independent_affine_current_volume_surface"))
    source=output/"implementation";source.mkdir()
    for path in [Path(__file__),Path("src/robotic_cutting_sim/surface_controls.py"),
                 Path("src/robotic_cutting_sim/affine_surface_controls.py"),Path("src/robotic_cutting_sim/tlmpm.py"),
                 Path(".deps/newton/newton/_src/geometry/particle_surface.py"),
                 Path(".deps/newton/newton/_src/geometry/particle_surface_kernels.py")]:
        (source/path.name).write_bytes(path.read_bytes())
    wp.init()
    total_F_receipt=tlmpm_total_F_receipt(args.device)
    write_json(output/"total_F_semantics.json",total_F_receipt)
    rows=[]
    for case in protocol["cases"]:
        for spacing in protocol["spacing_m"]:
            positions,masses,matrix=affine_particles(case,spacing,protocol["density_kg_m3"])
            jacobian=float(np.linalg.det(matrix))
            for hypothesis,recipe in protocol["recipes"].items():
                parameters={**protocol["shared_recipe"],"threshold":recipe["threshold"],"kernel_scale":recipe["kernel_scale"],
                    "voxel_size":recipe.get("voxel_size_m",recipe.get("voxel_size_over_spacing",0)*spacing),
                    "kernel_radius":recipe.get("kernel_radius_m",recipe.get("kernel_radius_over_spacing",0)*spacing)}
                radius=float(current_volume_radii(spacing,matrix)) if recipe["radius_mode"]=="current_volume_from_total_F" else spacing/2
                start=time.perf_counter()
                with wp.ScopedDevice(args.device):
                    surface=ParticleSurface(**parameters,device=args.device)
                    v,f,_=surface.extract(wp.array(positions,dtype=wp.vec3,device=args.device),
                        wp.full(len(positions),radius,dtype=float,device=args.device))
                    wp.synchronize_device(args.device)
                    elapsed=time.perf_counter()-start
                    if v is None or f is None:
                        raise RuntimeError(f"Empty geometry {case['id']}/{hypothesis}/{spacing}")
                    vertices,triangles=v.numpy().copy(),f.numpy().reshape(-1,3).copy()
                metrics=affine_mesh_measurements(vertices,triangles,case,spacing,masses.sum()/protocol["density_kg_m3"])
                c=protocol["criteria"]
                checks={"finite":metrics["finite"],"closed":metrics["closed_oriented_manifold"],
                    "components":metrics["mesh_components"]==c["mesh_components"],"euler":metrics["euler_characteristic"]==c["euler_characteristic"],
                    "volume_analytic":metrics["relative_volume_error_vs_analytic"]<=c["mesh_volume_relative_error_vs_analytic_max"],
                    "volume_mass":metrics["relative_volume_error_vs_mass"]<=c["mesh_volume_relative_error_vs_mass_over_density_max"],
                    "full_area_p95":metrics["full_area_sdf"]["p95_absolute_m"]<=spacing*c["full_mesh_area_weighted_p95_analytic_sdf_absolute_over_spacing_max"],
                    "full_vertex_max":metrics["full_vertex_sdf"]["max_absolute_m"]<=spacing*c["full_mesh_vertex_max_analytic_sdf_absolute_over_spacing_max"],
                    "central_face_bias":all(q["signed_mean_m"] is not None and abs(q["signed_mean_m"])<=spacing*c["central_face_offset_absolute_over_spacing_max"] for q in metrics["central_faces"])}
                row={"case_id":case["id"],"split":case["split"],"spacing_m":spacing,"hypothesis":hypothesis,
                    "particles":len(positions),"parameters":parameters,"particle_radius_m":radius,
                    "radius_mode":recipe["radius_mode"],"det_total_F":jacobian,"current_density_kg_m3":protocol["density_kg_m3"]/jacobian,"extract_wall_s":elapsed,"measurements":metrics,
                    "checks":checks,"status":"PASS" if all(checks.values()) else "FAIL"}
                rows.append(row)
                name=f"{case['id']}_{spacing:.4f}_{hypothesis}"
                np.savez_compressed(output/(name+".npz"),positions_m=positions,mass_kg=masses,vertices_m=vertices,triangles=triangles,total_F=matrix,particle_radius_m=radius)
                write_json(output/"partial.json",rows)
                print(f"{name}: {row['status']} volume_error={metrics['relative_volume_error_vs_analytic']:.4f} p95={metrics['full_area_sdf']['p95_absolute_m']*1e3:.3f}mm",flush=True)
    passed=all(r["status"]=="PASS" for r in rows if r["hypothesis"]=="H2")
    write_json(output/"result.json",{"protocol_id":protocol["protocol_id"],"protocol_sha256":protocol_hash,
        "status":"PASS_CURRENT_VOLUME_GEOMETRY_PROFILE" if passed else "FAIL_CURRENT_VOLUME_GEOMETRY_PROFILE",
        "total_F_semantics":total_F_receipt,"profile_promoted_for_cut_results":False,"cut_physics_changed":False,"rows":rows,
        "scope":"Exact affine-box renderer controls with actual total-F semantics probe; no TASK score or cutting evidence"})
    return int(not passed)


if __name__=="__main__":
    raise SystemExit(main())
