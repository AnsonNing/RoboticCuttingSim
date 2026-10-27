#!/usr/bin/env python3
"""Run frozen whole-closed-volume ParticleSurface controls, without cut data."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import warp as wp

from robotic_cutting_sim.recording import manifest,new_output,write_json
from robotic_cutting_sim.surface_controls import box_particles,mesh_measurements


def main():
    from newton.geometry import ParticleSurface
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir",type=Path)
    p.add_argument("--protocol",type=Path,default=Path("configs/particle_surface_geometry_v1.json"))
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    output=new_output(args.output_dir)
    protocol=json.loads(args.protocol.read_text())
    protocol_hash=hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    (output/"protocol.json").write_bytes(args.protocol.read_bytes())
    assert importlib.metadata.version("newton")==protocol["newton_version"]
    assert importlib.metadata.version("warp-lang")==protocol["warp_version"]
    write_json(output/"manifest.json",manifest({"device":args.device,"protocol_sha256":protocol_hash},"independent_particle_surface_geometry"))
    source=output/"implementation";source.mkdir()
    for path in [Path(__file__),Path("src/robotic_cutting_sim/surface_controls.py"),
                 Path(".deps/newton/newton/_src/geometry/particle_surface.py"),
                 Path(".deps/newton/newton/_src/geometry/particle_surface_kernels.py")]:
        (source/path.name).write_bytes(path.read_bytes())
    wp.init()
    rows=[]
    for case in protocol["cases"]:
        for spacing in protocol["spacing_m"]:
            positions,masses=box_particles(case,spacing,protocol["density_kg_m3"])
            for hypothesis,recipe in protocol["recipes"].items():
                parameters={**protocol["shared_recipe"],"threshold":recipe["threshold"],"kernel_scale":recipe["kernel_scale"],
                    "voxel_size":recipe.get("voxel_size_m",recipe.get("voxel_size_over_spacing",0)*spacing),
                    "kernel_radius":recipe.get("kernel_radius_m",recipe.get("kernel_radius_over_spacing",0)*spacing)}
                start=time.perf_counter()
                with wp.ScopedDevice(args.device):
                    surface=ParticleSurface(**parameters,device=args.device)
                    v,f,_=surface.extract(wp.array(positions,dtype=wp.vec3,device=args.device),
                        wp.full(len(positions),spacing/2,dtype=float,device=args.device))
                    wp.synchronize_device(args.device)
                    elapsed=time.perf_counter()-start
                    if v is None or f is None:
                        raise RuntimeError(f"Empty geometry {case['id']}/{hypothesis}/{spacing}")
                    vertices,triangles=v.numpy().copy(),f.numpy().reshape(-1,3).copy()
                metrics=mesh_measurements(vertices,triangles,case,spacing,masses.sum()/protocol["density_kg_m3"])
                c=protocol["criteria"]
                checks={"finite":metrics["finite"],"closed":metrics["closed_oriented_manifold"],
                    "components":metrics["mesh_components"]==c["mesh_components"],"euler":metrics["euler_characteristic"]==c["euler_characteristic"],
                    "volume_analytic":metrics["relative_volume_error_vs_analytic"]<=c["mesh_volume_relative_error_vs_analytic_max"],
                    "volume_mass":metrics["relative_volume_error_vs_mass"]<=c["mesh_volume_relative_error_vs_mass_over_density_max"],
                    "full_area_p95":metrics["full_area_sdf"]["p95_absolute_m"]<=spacing*c["full_mesh_area_weighted_p95_analytic_sdf_absolute_over_spacing_max"],
                    "full_vertex_max":metrics["full_vertex_sdf"]["max_absolute_m"]<=spacing*c["full_mesh_vertex_max_analytic_sdf_absolute_over_spacing_max"],
                    "central_face_bias":all(q["signed_mean_m"] is not None and abs(q["signed_mean_m"])<=spacing*c["central_face_offset_absolute_over_spacing_max"] for q in metrics["central_faces"])}
                row={"case_id":case["id"],"split":case["split"],"spacing_m":spacing,"hypothesis":hypothesis,
                    "particles":len(positions),"parameters":parameters,"extract_wall_s":elapsed,"measurements":metrics,
                    "checks":checks,"status":"PASS" if all(checks.values()) else "FAIL"}
                rows.append(row)
                name=f"{case['id']}_{spacing:.4f}_{hypothesis}"
                np.savez_compressed(output/(name+".npz"),positions_m=positions,mass_kg=masses,vertices_m=vertices,triangles=triangles)
                write_json(output/"partial.json",rows)
                print(f"{name}: {row['status']} volume_error={metrics['relative_volume_error_vs_analytic']:.4f} p95={metrics['full_area_sdf']['p95_absolute_m']*1e3:.3f}mm",flush=True)
    passed=all(r["status"]=="PASS" for r in rows if r["hypothesis"]=="H1")
    write_json(output/"result.json",{"protocol_id":protocol["protocol_id"],"protocol_sha256":protocol_hash,
        "status":"PASS_INDEPENDENT_GEOMETRY_PROFILE" if passed else "FAIL_INDEPENDENT_GEOMETRY_PROFILE",
        "profile_promoted_for_cut_results":False,"cut_physics_changed":False,"rows":rows,
        "scope":"Analytic closed-volume renderer controls only; no TASK score or cutting evidence"})
    return int(not passed)


if __name__=="__main__":
    raise SystemExit(main())
