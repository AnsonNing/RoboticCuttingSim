#!/usr/bin/env python3
"""Descriptive bidirectional whole-substrate/target mesh distances, without gates.

Exact nearest-triangle queries at every referenced vertex and triangle centroid.
Centroid area quadrature and sampled maxima are not continuous Hausdorff bounds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from robotic_cutting_sim.recording import new_output, write_json
from robotic_cutting_sim.scene_frames import WorkpieceTarget
from robotic_cutting_sim.surface_controls import weighted_quantile


@wp.kernel
def nearest_triangle_distance(mesh_id: wp.uint64, points: wp.array(dtype=wp.vec3), distance: wp.array(dtype=float)):
    i = wp.tid()
    point = points[i]
    query = wp.mesh_query_point_no_sign(mesh_id, point, 10.)
    distance[i] = -1.
    if query.result:
        nearest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distance[i] = wp.length(point-nearest)


def closed_volume(vertices, faces):
    triangles = np.asarray(vertices, float)[faces]
    volume = float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()/6)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    signs = np.where(edges[:, 0] < edges[:, 1], 1., -1.)
    balance = np.bincount(inverse, weights=signs, minlength=len(unique))
    return {"signed_volume_m3": volume, "absolute_volume_m3": abs(volume),
            "closed_oriented_manifold": bool(np.all(counts == 2) and np.all(balance == 0)),
            "boundary_edges": int(np.sum(counts == 1)), "nonmanifold_edges": int(np.sum(counts > 2)),
            "referenced_vertices": int(len(np.unique(faces))), "triangles": int(len(faces))}


def directed(vertices, faces, other_vertices, other_faces):
    points = np.asarray(vertices, np.float32)
    tri = points[faces].astype(float)
    active = np.unique(faces)
    centroids = tri.mean(axis=1).astype(np.float32)
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=-1)/2
    samples = np.concatenate([points[active], centroids])
    mesh = wp.Mesh(wp.array(np.asarray(other_vertices, np.float32), dtype=wp.vec3, device="cpu"),
                   wp.array(np.asarray(other_faces, np.int32).ravel(), dtype=int, device="cpu"))
    output = wp.empty(len(samples), dtype=float, device="cpu")
    wp.launch(nearest_triangle_distance, len(samples), [mesh.id,
              wp.array(samples, dtype=wp.vec3, device="cpu"), output], device="cpu")
    distances = output.numpy().astype(float)
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise RuntimeError("At least one closest-triangle query failed; no sample was discarded")
    vertex_distance, centroid_distance = distances[:len(active)], distances[len(active):]
    result = {"all_referenced_vertices": len(active), "all_triangle_centroids": len(faces),
              "surface_area_m2": float(area.sum()),
              "centroid_area_weighted_p95_distance_m": weighted_quantile(centroid_distance, area, .95),
              "centroid_area_weighted_rms_distance_m": float(np.sqrt(np.sum(area*centroid_distance**2)/area.sum())),
              "maximum_vertex_distance_m": float(vertex_distance.max()),
              "maximum_centroid_distance_m": float(centroid_distance.max()),
              "maximum_all_sampled_distance_m": float(distances.max()),
              "sampled_maximum_is_not_continuous_Hausdorff_bound": True,
              "nearest_query": "exact triangle closest point, Warp CPU mesh_query_point_no_sign",
              "error_trimming_or_roi_applied": False}
    return result, {"active_vertex_ids": active, "vertex_distance_m": vertex_distance,
                    "triangle_centroid_distance_m": centroid_distance, "triangle_area_m2": area}


def describe_meshes(actual_vertices, actual_faces, target_vertices, target_faces):
    wp.init()
    actual_to_target, a = directed(actual_vertices, actual_faces, target_vertices, target_faces)
    target_to_actual, b = directed(target_vertices, target_faces, actual_vertices, actual_faces)
    av, tv = closed_volume(actual_vertices, actual_faces), closed_volume(target_vertices, target_faces)
    difference = av["absolute_volume_m3"]-tv["absolute_volume_m3"]
    result = {"status": "DESCRIPTIVE_ONLY_NO_ACCEPTANCE_GATE", "actual_to_target": actual_to_target,
              "target_to_actual": target_to_actual, "actual_volume": av, "fixed_target_volume": tv,
              "signed_absolute_volume_difference_m3": difference,
              "absolute_volume_difference_fraction_of_fixed_target": abs(difference)/tv["absolute_volume_m3"],
              "volume_comparison_has_closed_oriented_inputs": av["closed_oriented_manifold"] and tv["closed_oriented_manifold"],
              "new_pass_thresholds": None, "prior_heightmap_or_v1_failures_overwritten": False}
    return result, {**{"actual_to_target_"+key: value for key, value in a.items()},
                    **{"target_to_actual_"+key: value for key, value in b.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("comparison", type=Path)
    p.add_argument("fixed_target", type=Path)
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()
    output = new_output(args.output_dir)
    actual_path, target_path = args.comparison/"column_comparison.npz", args.fixed_target/"fixed_target.npz"
    episode = Path(json.loads((args.comparison/"result.json").read_text())["episode"])
    task_path = episode/"task.json"
    task = json.loads(task_path.read_text()) if task_path.exists() else json.loads((episode/"setup.json").read_text())["task"]
    frame = WorkpieceTarget(tuple(task["world_from_workpiece_xyzw"]))
    with np.load(actual_path) as a, np.load(target_path) as t:
        faces = a["all_triangles"][a["substrate_triangle_selection"]]
        result, debug = describe_meshes(frame.points_to_local(a["substrate_vertices_world"]), faces, t["vertices_local_m"], t["triangles"])
    result["scope"] = "All currently attributed substrate triangles/vertices versus the independently fixed target; no chip/ROI relabeling, no fitted alignment"
    result["world_from_workpiece_xyzw"] = task["world_from_workpiece_xyzw"]
    result["input_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [actual_path, target_path, Path(__file__)]}
    np.savez_compressed(output/"all_distance_samples.npz", **debug)
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write_json(output/"result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
