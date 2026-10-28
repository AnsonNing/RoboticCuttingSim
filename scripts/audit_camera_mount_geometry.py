#!/usr/bin/env python3
"""Audit visual camera boxes against the official hand's actual triangles.

Frozen geometric criteria: zero triangle/box intersections for each housing;
report bracket intersections and central workpiece-ray occlusion separately.
This checks visual geometry, not contact dynamics or full trajectory clearance.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from robotic_cutting_sim.camera_rig import make_rig
from robotic_cutting_sim.recording import ROOT, manifest, new_output, write_json


def triangle_box_overlap(triangles, transform, dimensions):
    """Exact triangle/OBB separating-axis test, with a 10nm numeric tolerance."""
    transform = np.asarray(transform, float)
    triangles = (np.asarray(triangles)-transform[:3, 3]) @ transform[:3, :3]
    half = np.asarray(dimensions, float)/2
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edges = (v1-v0, v2-v1, v0-v2)
    axes = [np.broadcast_to(a, v0.shape) for a in np.eye(3)]
    axes += [np.cross(edges[0], edges[1])]
    axes += [np.cross(e, a) for e in edges for a in np.eye(3)]
    overlap = np.ones(len(triangles), bool)
    for axis in axes:
        projection = np.sum(triangles*axis[:, None, :], axis=2)
        bound = np.abs(axis) @ half
        eps = 1e-8*np.linalg.norm(axis, axis=1)
        overlap &= (projection.min(1) <= bound+eps) & (projection.max(1) >= -bound-eps)
    return overlap


def ray_first_hit(triangles, origin, direction):
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1, e2 = b-a, c-a
    h = np.cross(np.broadcast_to(direction, e2.shape), e2)
    det = np.sum(e1*h, axis=1)
    ok = np.abs(det) > 1e-12
    inv = np.divide(1., det, out=np.zeros_like(det), where=ok)
    s = origin-a
    u = inv*np.sum(s*h, axis=1)
    q = np.cross(s, e1)
    v = inv*(q@direction)
    t = inv*np.sum(e2*q, axis=1)
    ok &= (u >= 0) & (v >= 0) & (u+v <= 1) & (t > 1e-6)
    return float(t[ok].min()) if np.any(ok) else None


def official_hand_triangles(asset_path):
    """Actual visual hand and fingers, with the runtime's 40mm finger joints."""
    from pxr import Gf, Usd, UsdGeom
    stage = Usd.Stage.Open(str(Path(asset_path).resolve()))
    hand = next(p for p in stage.Traverse() if p.GetName() == "panda_hand")
    for child in hand.GetChildren():
        if child.GetName() not in ("panda_leftfinger", "panda_rightfinger"):
            continue
        xf = UsdGeom.Xformable(child)
        matrix = np.asarray(xf.GetLocalTransformation()).T.copy()
        matrix[1, 3] += .04 if child.GetName() == "panda_leftfinger" else -.04
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))
    inverse = np.linalg.inv(np.asarray(UsdGeom.Xformable(hand).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T)
    meshes = []
    for prim in Usd.PrimRange(hand, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        transform = inverse @ np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        points = np.asarray(mesh.GetPointsAttr().Get(), float)
        points = points @ transform[:3, :3].T + transform[:3, 3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), int)
        faces, cursor = [], 0
        for count in counts:
            poly = indices[cursor:cursor+count]
            faces.extend((poly[0], poly[j], poly[j+1]) for j in range(1, count-1))
            cursor += count
        meshes.append((str(prim.GetPath()), points[np.asarray(faces)]))
    return meshes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--specs", type=Path, help="Optional frozen rig JSON containing a specs object")
    args = p.parse_args()
    output = new_output(args.output_dir)
    specs = json.loads(args.specs.read_text())["specs"] if args.specs else make_rig().specs
    write_json(output/"protocol.json", {"hypothesis": __doc__, "criteria": {
        "housing_triangle_intersections": 0, "tolerance_m": 1e-8, "finger_joints_m": [.04, .04]}})
    write_json(output/"input_specs.json", {"specs": specs})
    asset = ROOT / "artifacts/workcell_asset_cache/Isaac/IsaacLab/Robots/FrankaEmika/franka_panda.usda"
    meshes = official_hand_triangles(asset)
    all_triangles = np.concatenate([tri for _, tri in meshes])
    rows = {}
    for name, spec in specs.items():
        if spec["parent_hand_index"] is None:
            continue
        boxes = []
        for obb in spec["visual_obbs"]:
            hits = {path: int(triangle_box_overlap(tri, obb["T_hand_obb"], obb["dimensions_m"]).sum())
                    for path, tri in meshes}
            boxes.append({"name": obb["name"], "intersecting_triangles": sum(hits.values()),
                          "per_mesh": hits, "dimensions_m": obb["dimensions_m"],
                          "T_hand_obb": obb["T_hand_obb"]})
        mount = np.asarray(spec["T_hand_optical_cv"])
        hit = ray_first_hit(all_triangles, mount[:3, 3], mount[:3, 2])
        rows[name] = {"boxes": boxes, "central_forward_ray_hand_first_hit_m": hit,
                      "housing_triangle_clear": all(b["intersecting_triangles"] == 0 for b in boxes if b["name"] == "Housing")}
    result = {"status": "PASS_HOUSING_GEOMETRY" if all(r["housing_triangle_clear"] for r in rows.values()) else "FAIL_HOUSING_GEOMETRY",
              "scope": "Static official hand visual triangle/OBB intersections; no contact, other-link, table, material, or full-trajectory clearance claim",
              "criteria": {"housing_triangle_intersections": 0, "tolerance_m": 1e-8, "finger_joints_m": [.04, .04]},
              "camera_mounts": rows, "official_hand_visual_triangle_count": len(all_triangles)}
    write_json(output/"result.json", result)
    write_json(output/"manifest.json", manifest({"asset": str(asset), "specs_source": str(args.specs)}, "camera_mount_visual_geometry"))
    print(result["status"], {name: [(b["name"], b["intersecting_triangles"]) for b in row["boxes"]] for name, row in rows.items()})
    return int(result["status"].startswith("FAIL"))


if __name__ == "__main__":
    raise SystemExit(main())
